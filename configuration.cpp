#include "recon.hpp"
#include <opencv2/highgui/highgui.hpp>

#include <set>
#include <getopt.h>
#include <stdio.h>
using namespace cv;

Configuration::Configuration(int argc, char** argv)
{
	char *inFileName=NULL, *outFileName=NULL;
	verbosity = 0;
	doEstimateExposure = false;
	while (1) {
		int option_index = 0;
		static struct option long_options[] = {
			{"estimate-exposure", no_argument, 0,  'e' },
			{"help",    no_argument,       0,  'h' },
			{"input",   required_argument, 0,  'i' },
			{"output",  required_argument, 0,  'o' },
			{"verbose", no_argument,       0,  'v' },
			{0,         0,                 0,  0 }
		};
		
		char c = getopt_long(argc, argv, "ehi:o:v", long_options, &option_index);
		if (c == -1)
			break;
		
		switch (c) {
			case 'e':
				doEstimateExposure = true;
				break;
			
			case 'i':
				inFileName = optarg;
				break;
			
			case 'o':
				outFileName = optarg;
				break;
		
			case 'v':
				verbosity = 99;
				break;
			
			case 'h':
			case 0:
			default:
				printf("Usage: recon [OPTIONS] [INPUT_FILE]\n");
				printf("Reconstructs dense geometry from given YAML scene calibration and video\n\n");
				printf("  -e, --estimate-exposure   try to normalize exposure over time\n");
				printf("  -i, --input               input YAML file name (usually exported from Blender)\n");
				printf("  -o, --output              output Wavefront OBJ file name (.obj)\n");
				printf("  -v, --verbose             print out messages during computation\n");
				printf("  -h, --help                print this message and exit\n");
				exit(0);
				break;
		}
	}
	
	if (optind < argc) {
		inFileName = argv[optind];
	}
		
	FileStorage fs((inFileName ? inFileName : "tracks/rotunda.yaml"), FileStorage::READ);
	
	FileNode nodeClip = fs["clip"];
 	string clipPath;
	nodeClip["width"] >> width;
	nodeClip["height"] >> height;
	nodeClip["path"] >> clipPath;
	nodeClip["center-x"] >> centerX;
	nodeClip["center-y"] >> centerY;
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
		int frame;
		(*cit)["frame"] >> frame;
		assert (frame > 0 && frame <= frameCount);
		frame -= 1;
		(*cit)["near"] >> nearVals[frame];
		(*cit)["far"] >> farVals[frame];
		//undistort?
		(*cit)["projection"] >> cameras[frame];
		if (trackedFrameCount <= frame)
			trackedFrameCount = frame+1;
	}
	for (int i=0; i<trackedFrameCount; i++) {
		assert (nearVals[i] > 0 && farVals[i] > 0);
	}
	cameras.resize(trackedFrameCount);
	nearVals.resize(trackedFrameCount);
	farVals.resize(trackedFrameCount);
	
	frames.resize(trackedFrameCount);
	// read and cache the whole clip
	for (int fi = 0; fi < trackedFrameCount; fi++) {
		Mat frame;
		//Mat *value = new Mat;
		clip.read(frame);
		//clip.read(frames[fi]);
		frame.copyTo(frames[fi]);
	}
	
	if (doEstimateExposure)
		estimateExposure();
}

// BEGIN MESSY SHIT BLOCK
inline const float invertPoly(const float k1, const float k2, const float q)
{
	float x = q, dx;
	for (int i=0; i<1000; i++) {
		dx = (q - x*(1 + x*(k1 + x*k2)));
		if (dx < 1.e-6 && dx > -1.e-6)
			break;
		dx /= (1 + x*(2*k1 + 3*x*k2));
		x += dx;
	}
	return x;
}

const Mat screenToCamera(const Mat points, const vector<float> lensDistortion)
// expects cartesian 3D points in rows
{
	const Mat mneg(Matx22f(-1, 0, 0, -1));
	Mat rp = mneg * points.colRange(0, 2).t();
	/*Mat rad = Mat::ones(1, 2, CV_32FC1) * rp.mul(rp);
	float *radius = rad.ptr<float>(0);
	for (int i=0; i < rad.cols; i++) {
		float k = sqrt(invertPoly(lensDistortion[0], lensDistortion[1], radius[i]) / radius[i]);
		rp.col(i) *= k;
	}*/
	return rp.t();
}

const Mat Configuration::reprojectPoints(const int frameNo) {
	Mat projectedPoints = (camera(frameNo) * bundles.t()).t();
	Mat cartesianPoints = dehomogenize(projectedPoints);
	Mat undistortedPoints = screenToCamera(cartesianPoints, lensDistortion);
	return undistortedPoints;
}
// END MESSY SHIT BLOCK

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
			float imageX = re[j*2]/2*width + centerX, imageY = height - (re[j*2 + 1]/2*height + centerY);
			float sample;
			if (bundlesEnabled[j].count(i) && (sample = sampleImage(image, 1, imageX, imageY)) > 0) {
				br[i*pointCount + j] = sample;
				we[i*pointCount + j] = 1.;
			} else {
				br[i*pointCount + j] = 0.;
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
				sum += br[i*pointCount + j] * we[i*pointCount + j] / exposure[i];
				weightSum += we[i*pointCount + j];
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
				sum += br[i*pointCount + j] * we[i*pointCount + j] / pointColor[j];
				weightSum += we[i*pointCount + j];
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

const Mat Configuration::frame(int frameNo)
{
	return frames[frameNo];
}

const Mat Configuration::camera(int frameNo)
{
	return cameras[frameNo];
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
