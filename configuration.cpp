#include "recon.hpp"
#include <opencv2/highgui/highgui.hpp>
#include <set>
using namespace cv;

Configuration::Configuration(int argc, char** argv)
{
	FileStorage fs(((argc>1) ? argv[1]: "tracks/rotunda.yaml"), FileStorage::READ);
	
	FileNode nodeClip = fs["clip"];
	int width, height;
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
	std::vector< std::set<int> > bundlesEnabled;
	for (FileNodeIterator it = tracks.begin(); it != tracks.end(); it++){
		Mat bundle;
		(*it)["bundle"] >> bundle;
		vector<int> enabledFrames;
		(*it)["frames-enabled"] >> enabledFrames;
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
