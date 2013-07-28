#include "recon.hpp"
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <set>
#include <getopt.h>
#include <cstdio>
#include <libgen.h> // dirname(char*)
const char dirDelimiter = '/';
using namespace cv;

Configuration::Configuration(int argc, char** argv)
{
	char *inFileName=NULL;
	inMeshFile=NULL;
	outFileName = (char*)"output.obj";
	verbosity = 0;
	doEstimateExposure = false;
	useFarneback = false;
	
	iterationCount = 2;
	sceneResolution = 1;
	cameraThreshold = 10.;
	scalingFactor = 1.;
	skipFrames = 1;
	while (1) {
		int option_index = 0;
		static struct option long_options[] = {
			{"input",   required_argument, 0,  'i' },
			{"initial-mesh",   required_argument, 0,  'm' },
			{"output",  required_argument, 0,  'o' },
			{"camera-threshold", required_argument, 0,  'c' },
			{"estimate-exposure", no_argument, 0,  'e' },
			{"iterations", required_argument, 0, 'n' },
			{"scale", required_argument, 0, 's' },
			{"skip-frames", required_argument, 0, 'k' },
			{"farneback",   no_argument, 0,  'f' },
			{"verbose", no_argument,       0,  'v' },
			{"hyper-verbose", no_argument,       0,  'V' },
			{"help",    no_argument,       0,  'h' },
			{0,         0,                 0,  0 }
		};
		
		char c = getopt_long(argc, argv, "i:m:o:c:en:s:k:fvVh", long_options, &option_index);
		if (c == -1)
			break;
		
		switch (c) {
			case 'i':
				inFileName = optarg;
				break;
			
			case 'm':
				inMeshFile = optarg;
				break;
			
			case 'o':
				outFileName = optarg;
				break;
		
			case 'c':
				cameraThreshold = atof(optarg);
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
			
			case 'f':
				useFarneback = true;
				break;
			
			case 'v':
				if (verbosity < 2) verbosity = 2;
				break;
			
			case 'V':
				verbosity = 99;
				break;
			
			case 'h':
			case 0:
			default:
				printf("Usage: recon [OPTIONS] [INPUT_FILE]\n");
				printf("Reconstructs dense geometry from given YAML scene calibration and video\n\n");
				printf("  -c, --camera-threshold=f  use given threshold for camera selection (default: 10)\n");
				printf("  -e, --estimate-exposure   try to normalize exposure over time (default: false)\n");
				printf("  -f, --farneback           use Farneback's algorithm for optical flow, intsead of Horn & Schunck's (default: false)\n");
				printf("  -h, --help                print this message and exit\n");
				printf("  -i, --input=s             input configuration file name (.yaml, usually exported from Blender; default: output.obj)\n");
				printf("  -k, --skip-frames=i       use only every n-th frame of the sequence (default: 1)\n");
				printf("  -m, --input-mesh=s        load initial scene estimate from given file (.obj, by default not set)\n");
				printf("  -n, --iterations=i        maximal iteration count of surface reconstruction (default: 2)\n");
				printf("  -o, --output=s            output mesh file name (.obj)\n");
				printf("  -s, --scale=f             downsample the input video by a given factor (default: 1.0)\n");
				printf("  -v, --verbose             print current task and summarize its results during computation\n");
				printf("  -V, --hyper-verbose       print out what comes to mind, and save all images at hand\n");
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
	nodeClip["width"] >> width;
	nodeClip["height"] >> height;
	
 	string clipPathRel;
	nodeClip["path"] >> clipPathRel;
	string clipPath(dirname(inFileName));
	clipPath.push_back(dirDelimiter);
	clipPath.append(clipPathRel);
	nodeClip["center-x"] >> centerX;
	nodeClip["center-y"] >> centerY;
	if (scalingFactor != 1 && scalingFactor != 0) {
		width /= scalingFactor;
		height /= scalingFactor;
		centerX /= scalingFactor;
		centerY /= scalingFactor;
	}
	nodeClip["distortion"] >> lensDistortion;
	
	VideoCapture clip(clipPath);
	if (!clip.isOpened()) {
		printf("Cannot read clip %s, exiting.\n", clipPath.c_str());
		exit(1);
	}
	int frameCount = clip.get(CV_CAP_PROP_FRAME_COUNT);

	FileNode tracks = fs["tracks"];
	bundles = Mat(0, 4, CV_32FC1);
	for (FileNodeIterator it = tracks.begin(); it != tracks.end(); it++){
		Mat bundle;
		(*it)["bundle"] >> bundle;
		vector<int> enabledFrames;
		(*it)["frames-enabled"] >> enabledFrames;
		{
			int writeIndex = 0;
			for (int i=0; i<enabledFrames.size(); i++) {
				if ((enabledFrames[i] - 1)%skipFrames == 0) {
					enabledFrames[writeIndex] = (enabledFrames[i] - 1) / skipFrames;
					writeIndex += 1;
				}
			}
			enabledFrames.resize(writeIndex);
		}
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
	
	if (doEstimateExposure) {
		estimateExposure();
	} else {
		for (int i=0; i<frames.size(); i++)
			cv::cvtColor(frames[i], frames[i], CV_BGR2GRAY);
	}
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
{
	if (verbosity >= 1)
		printf("Estimating exposure values...\n");
	
	int frameCount = cameras.size(), pointCount = bundles.rows;
	char ch = frames[0].channels();
	Mat sampledColor(frameCount*pointCount, ch, CV_32FC1); // measured brightness in linear space. rows: (frames x points via sampleIds), columns: channels
	Mat sampleIds(-1 * Mat::ones(frameCount, pointCount, CV_32SC1)); // row index of given point, given frame in sampledColor or -1 if invalid
	std::vector<Mat> validSamples; // submatrices prepared for the linear system
	validSamples.reserve(frameCount);
	
	int32_t rowId=0;
	int matOffset=0;
	for (int i=0; i<frameCount; i++) {
		Mat image = frames[i];
		assert(image.channels() == ch);
		Mat reprojected = reprojectPoints(i);
		for (int j=0; j<pointCount; j++) {
			if (bundlesEnabled[j].count(i)) {
				float *re = reprojected.ptr<float>(j);
				float imageX = centerX + re[0]*width*0.5,
				      imageY = height - centerY - re[1]*height*0.5;
				bool valid = true;
				//if it is enabled in this frame:
				float *sc = sampledColor.ptr<float>(rowId);
				float sample;
				for (char c=0; c<ch; c++) {
					if ((sample = sampleImage(image, 16, imageX, imageY, c)) == -1) {
						valid = false;
						break;
					}
					sc[c] = sample;
				}
				if (valid) {
					sampleIds.at<int32_t>(i, j) = rowId;
					rowId ++; // otherwise, the data get overwritten
				} else {
					sampleIds.at<int32_t>(i, j) = -1;
				}
			} else {
				sampleIds.at<int32_t>(i, j) = -1;
			}
		}
		if (rowId-matOffset < ch) {
			// TODO: retry taking all values
			assert(false);
		}
		validSamples.push_back(sampledColor.rowRange(matOffset, rowId));
		matOffset = rowId;
	}
	sampledColor.resize(rowId);

	// for normalization
	double sumBrightness = 0;
	for (int j=0; j<pointCount; j++) {
		float sum = 0.;
		int weightSum = 0;
		for (int i=0; i<frameCount; i++) {
			int32_t rowId = sampleIds.at<int32_t>(i, j);
			if (rowId == -1)
				continue;
			float *sc = sampledColor.ptr<float>(rowId);
			for (char c=0; c < ch; c++)
				sumBrightness += sc[c];
		}
	}
	sumBrightness *= 1./ch;
	// sampledColor[frame][point] . exposure[frame] (should)= pointBrightness[point]
	Mat exposure(1./ch * Mat::ones(ch, frameCount, CV_32FC1)), pointBrightness(Mat::ones(pointCount, 1, CV_32FC1));
	int iteration = 0;
	double error;
	for (int iteration=0; iteration<100; iteration++) {
		error = 0;
		double currentSumBrightness = 0;
		//imagine that exposure is correct
		for (int j=0; j<pointCount; j++) {
			float sum = 0.;
			int weightSum = 0;
			for (int i=0; i<frameCount; i++) {
				int32_t rowId = sampleIds.at<int32_t>(i, j);
				if (rowId == -1)
					continue;
				weightSum += 1;
				float *sc = sampledColor.ptr<float>(rowId);
				for (char c=0; c < ch; c++) {
					sum += sc[c] * exposure.at<float>(c, i);
				}
			}
			currentSumBrightness += sum;
			//pointColor[j] = avg(sampledColor[i, j] * exposure[i] over all i)
			float *pb = pointBrightness.ptr<float>(0);
			if (weightSum > 0)
				pointBrightness.at<float>(j) = sum / weightSum;
			else
				pointBrightness.at<float>(j) = 0.; // TODO: such points should be just discarded
		}
		
		// normalize brightness to original scale
		pointBrightness *= sumBrightness / currentSumBrightness;
		
		//imagine that point colors are correct
		for (int i=0; i<frameCount; i++) {
			Mat oldExposure;
			exposure.col(i).copyTo(oldExposure);
			//exposure[i][*] = validSamples[i]^-1 . pointBrightness[*]
			Mat validPointBrightness(0, 1, CV_32FC1);
			for (int j=0; j<pointCount; j++) {
				if (sampleIds.at<int32_t>(i, j) >= 0)
					validPointBrightness.push_back(pointBrightness.at<float>(j));
			}
			assert(validSamples[i].rows == validPointBrightness.rows);
			// extremely strongly overrelax
			float omega = 0.8;
			exposure.col(i) = validSamples[i].inv(cv::DECOMP_SVD) * validPointBrightness * (1+omega) - oldExposure * omega;
			error += cv::norm(validSamples[i]*exposure.col(i) - validPointBrightness) / validPointBrightness.rows;
		}
		if (error/frameCount < 0.1)
			break;
	}

	//save exposure somewhere (TODO: or multiply each frame directly?)
	if (verbosity >= 3) {
		FILE *exlog = fopen("exposure.tab", "w+");
		for (int i=0; i<frameCount; i++) {
			float stddev = 0.;
			int weightSum = 0;
			for (int j=0; j<pointCount; j++) {
				int32_t rowId = sampleIds.at<int32_t>(i, j);
				if (rowId == -1)
					continue;
				float *sc = sampledColor.ptr<float>(rowId);
				for (char c=0; c<ch; c++) {
					float difference = sc[c] - exposure.at<float>(c,i) * pointBrightness.at<float>(j);
					stddev += (difference * difference);
					weightSum += 1;
				}
			}
			stddev = sqrt(stddev / weightSum);
			fprintf(exlog, "%f\t%f\t%f\t%f\n", exposure.at<float>(0,i), exposure.at<float>(1,i), exposure.at<float>(2,i), stddev);
		}
		fclose(exlog);
	}
	for (int i=0; i<frameCount; i++) {
		std::vector<Mat> channels;
		cv::split(frames[i], channels);
		frames[i] = Mat::zeros(frames[i].rows, frames[i].cols, CV_8UC1);
		for (char c=0; c<ch; c++) {
			frames[i] += channels[c] * exposure.at<float>(c, i);
		}
		/*
		// DEBUG
		Mat painted;
		cv::cvtColor(frames[i], painted, CV_GRAY2BGR);
		Mat reprojected = reprojectPoints(i);
		for (int j=0; j<pointCount; j++) {
			float *re = reprojected.ptr<float>(j);
			float imageX = centerX + re[0]*width*0.5,
			      imageY = height - centerY - re[1]*height*0.5;
			float brightness = pointBrightness.at<float>(j);
			cv::Scalar color = cv::Scalar(brightness, brightness, brightness);
			cv::Scalar mark = cv::Scalar(0,0,255);
			cv::circle(painted, cv::Point(imageX, imageY), 3, color, -1, 8);
			cv::circle(painted, cv::Point(imageX, imageY), 1, mark, -1, 8);
		}
		cv::namedWindow("w");
		cv::imshow("w", painted);
		cvWaitKey(1000);
		char filename[300];
		snprintf(filename, 300, "frame%i.png", i);
		cv::imwrite(filename, frames[i]);
		*/
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
