#include "recon.hpp"
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/imgproc/imgproc.hpp"

#include <set>
#include <getopt.h>
#include <stdio.h>
using namespace cv;

Configuration::Configuration(int argc, char** argv)
{
	char *inFileName=NULL;
	outFileName = (char*)"output.obj";
	verbosity = 0;
	doEstimateExposure = false;
	
	iterationCount = 2;
	sceneResolution = 1;
	scalingFactor = 1;
	skipFrames = 1;
	while (1) {
		int option_index = 0;
		static struct option long_options[] = {
			{"input",   required_argument, 0,  'i' },
			{"output",  required_argument, 0,  'o' },
			{"estimate-exposure", no_argument, 0,  'e' },
			{"iterations", required_argument, 0, 'n' },
			{"scale", required_argument, 0, 's' },
			{"skip-frames", required_argument, 0, 'k' },
			{"verbose", no_argument,       0,  'v' },
			{"help",    no_argument,       0,  'h' },
			{0,         0,                 0,  0 }
		};
		
		char c = getopt_long(argc, argv, "i:o:en:s:k:vh", long_options, &option_index);
		if (c == -1)
			break;
		
		switch (c) {
			case 'i':
				inFileName = optarg;
				break;
			
			case 'o':
				outFileName = optarg;
				break;
		
			case 'e':
				doEstimateExposure = true;
				break;
			
			case 'n':
				iterationCount = atoi(optarg);
				break;
			
			case 's':
				scalingFactor = atof(optarg);
				break;
			
			case 'k':
				skipFrames = atoi(optarg);
				break;
			
			case 'v':
				verbosity = 99;
				break;
			
			case 'h':
			case 0:
			default:
				printf("Usage: recon [OPTIONS] [INPUT_FILE]\n");
				printf("Reconstructs dense geometry from given YAML scene calibration and video\n\n");
				printf("  -e, --estimate-exposure   try to normalize exposure over time (default: false)\n");
				printf("  -h, --help                print this message and exit\n");
				printf("  -i, --input               input configuration file name (.yaml, usually exported from Blender; default: output.obj)\n");
				printf("  -k, --skip-frames=i       use only every n-th frame of the sequence (default: 1)\n");
				printf("  -n, --iterations=i        maximal iteration count of surface reconstruction (default: 2)\n");
				printf("  -o, --output              output mesh file name (.obj)\n");
				printf("  -s, --scale=f             downsample the input video by a given factor (default: 1.0)\n");
				printf("  -v, --verbose             print out messages during computation\n");
				exit(0);
				break;
		}
	}
	
	if (optind < argc) {
		inFileName = argv[optind];
	}
		
	FileStorage fs((inFileName ? inFileName : "tracks/koberec-.yaml"), FileStorage::READ);
	if (!fs.isOpened()) {
		printf("Cannot read file %s, exiting.\n", inFileName);
		exit(1);
	}
	
	FileNode nodeClip = fs["clip"];
 	string clipPath;
	nodeClip["width"] >> width;
	nodeClip["height"] >> height;
	if (scalingFactor != 1 && scalingFactor != 0) {
		width /= scalingFactor;
		height /= scalingFactor;
	}
	
	nodeClip["path"] >> clipPath;
	nodeClip["center-x"] >> centerX;
	nodeClip["center-y"] >> centerY;
	centerX += 0.5; // conversion from grid to grid, seems to help
	centerY -= 0.5;
	nodeClip["distortion"] >> lensDistortion;
	
	VideoCapture clip(clipPath);
	int frameCount = clip.get(CV_CAP_PROP_FRAME_COUNT);

	FileNode tracks = fs["tracks"];
	bundles = Mat(0, 4, CV_32FC1);
	for (FileNodeIterator it = tracks.begin(); it != tracks.end(); it++){
		Mat bundle;
		(*it)["bundle"] >> bundle;
		vector<int> enabledFrames;
		(*it)["frames-enabled"] >> enabledFrames;
		for (int i=0; i<enabledFrames.size(); i++)
			enabledFrames[i] -= 1;
		std::set<int> enabledFramesSet(enabledFrames.begin(), enabledFrames.end());
		bundlesEnabled.push_back(enabledFramesSet);
		bundle = bundle.t();
		bundles.push_back(bundle);
	}
	//bundles = bundles.t();

	FileNode camera = fs["camera"];
	cameras.resize(frameCount);
	nearVals.resize(frameCount);
	farVals.resize(frameCount);
	int trackedFrameCount = -1;
	for (FileNodeIterator cit = camera.begin(); cit != camera.end(); cit ++)	{
		int fi;
		(*cit)["frame"] >> fi;
		assert (fi > 0 && fi <= frameCount);
		fi -= 1;
		if (fi % skipFrames)
			continue;
		fi /= skipFrames;
		(*cit)["near"] >> nearVals[fi];
		(*cit)["far"] >> farVals[fi];
		(*cit)["projection"] >> cameras[fi];
		if (trackedFrameCount <= fi)
			trackedFrameCount = fi+1;
	}
	for (int i=0; i<trackedFrameCount/skipFrames; i++) {
		assert (nearVals[i] > 0 && farVals[i] > 0);
	}
	cameras.resize(trackedFrameCount);
	nearVals.resize(trackedFrameCount);
	farVals.resize(trackedFrameCount);
	
	frames.resize(trackedFrameCount);
	// read and cache the whole clip
	for (int fi = 0; fi < trackedFrameCount; fi++) {
		Mat frame;
		clip.read(frame);
		// todo: undistort!
		if (frame.rows != height || frame.cols != width)
			cv::resize(frame, frames[fi], cv::Size(width, height), CV_INTER_AREA);
		else
			frame.copyTo(frames[fi]);
		for(int skip=fi*skipFrames+1; skip < (fi+1)*skipFrames && skip < frameCount; skip++)
			clip.read(frame);
	}
	
	if (doEstimateExposure)
		estimateExposure();
}

void cameraToScreen(Mat points, const vector<float> lensDistortion, float aspect)
// expects cartesian 3D points in rows
{
	for (int i=0; i < points.rows; i++) {
		float *p = points.ptr<float>(i);
		float radSquared = (p[0]*p[0] + p[1]*p[1]*aspect*aspect)/4;
		float k = 1 + radSquared * (lensDistortion[0] + radSquared * lensDistortion[1]); 
		points.row(i) *= k;
	}
}

const Mat Configuration::reprojectPoints(const int frameNo) {
	Mat projectedPoints = (camera(frameNo) * bundles.t()).t();
	Mat cartesianPoints = dehomogenize(projectedPoints);
	cameraToScreen(cartesianPoints, lensDistortion, (float)height/(float)width);
	return cartesianPoints;
}

void Configuration::estimateExposure()
//TODO: je potřeba to převést na poctivou korekci bílé. Možná dokonce CAM?
//TODO: convert from camera color space to linear
 // OR try to estimate the gamma? Would be super-cool :)
{
	int frameCount = cameras.size(), pointCount = bundles.rows;
	Mat brightness(frameCount, pointCount, CV_32FC1); // measured brightness in linear space. rows: frames, cols: points
	Mat weight(frameCount, pointCount, CV_32FC1); // reliability of each measurement
	//TODO: weight sum across points and frames can be precalculated, no need to save two matrices
	
	float *br = brightness.ptr<float>(0), *we = weight.ptr<float>(0);
	for (int i=0; i<frameCount; i++) {
		Mat image;
		frames[i].copyTo(image);
		Mat reprojected = reprojectPoints(i);
		float *re = reprojected.ptr<float>(0);
		for (int j=0; j<pointCount; j++) {
			//if it is enabled in this frame:
			float imageX = centerX + re[j*3]*width*0.5 , imageY = height - centerY - re[j*3 + 1]*height*0.5;
			float sample;
			if (bundlesEnabled[j].count(i) && (sample = sampleImage(image, 1, imageX, imageY)) > 0) {
				br[i*pointCount + j] = sample;
				we[i*pointCount + j] = 1.;
			} else {
				br[i*pointCount + j] = 1.;
				we[i*pointCount + j] = 0.;
			}
		}
	}

	if (verbosity >= 1)
		printf("Estimating exposure values...\n");
	vector<float> exposure(frameCount, 1.0), pointColor(pointCount, 1.0); // brightness (should)= exposure * pointColors.t()
	float change;
	do {
		change = 0.;
		//imagine that exposure is correct
		for (int j=0; j<pointCount; j++) {
			float sum = 0., weightSum = 0.;
			for (int i=0; i<frameCount; i++) {
				if (exposure[i] > 0) {
					sum += br[i*pointCount + j] * we[i*pointCount + j] / exposure[i];
					weightSum += we[i*pointCount + j];
				}
			}
			//pointColor[j] = weightedAvg(brightness[i, j]/exposure[i] over all i)
			if (weightSum > 0)
				pointColor[j] = sum / weightSum;
			else
				pointColor[j] = 0.;
		}
		//imagine that point colors are correct
		for (int i=0; i<frameCount; i++) {
			float sum = 0., weightSum = 0.;
			for (int j=0; j<pointCount; j++) {
				if (pointColor[j] > 0) {
					sum += br[i*pointCount + j] * we[i*pointCount + j] / pointColor[j];
					weightSum += we[i*pointCount + j];
				}
			}
			//exposure[i] = weightedAvg(brightness[i, j]/pointColor[j] over all j)
			if (weightSum > 0) {
				float newExposure = sum / weightSum, diff = exposure[i] - newExposure;
				change += diff*diff;
				exposure[i] = newExposure;
			} else
				exposure[i] = 0.;
		}
	} while (change/frameCount > 1e-10);
	
	//save exposure somewhere (TODO: or multiply each frame directly?)
	if (verbosity >= 3) {
		FILE *exlog = fopen("exposure.tab", "w+");
		for (int i=0; i<frameCount; i++) {
			float stddev = 0., weightSum = 0.;
			for (int j=0; j<pointCount; j++) {
				float difference = br[i*pointCount + j] - exposure[i] * pointColor[j];
				stddev += (difference * difference) * we[i*pointCount + j];
				weightSum += we[i*pointCount + j];
			}
			stddev = sqrt(stddev / weightSum);
			fprintf(exlog, "%f\t%f\n", exposure[i], stddev);
		}
		fclose(exlog);
	}
	for (int i=0; i<frameCount; i++) {
		frames[i] /= exposure[i];
	}
}

Configuration::~Configuration()
{
	
}

Mat Configuration::reconstructedPoints()
{
	return bundles;
}

const Mat Configuration::frame(int frameNo) const
{
	return frames[frameNo];
}

const Mat Configuration::camera(int frameNo) const
{
	return cameras[frameNo];
}

const std::vector<Mat> Configuration::allCameras() const
{
	return vector<Mat> (cameras);
}

const float Configuration::near(int frameNo)
{
	return nearVals[frameNo];
}

const float Configuration::far(int frameNo)
{
	return farVals[frameNo];
}

const int Configuration::frameCount()
{
	return frames.size();
}
