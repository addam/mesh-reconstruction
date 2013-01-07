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
	int i = 0;
	for (FileNodeIterator cit = camera.begin(); cit != camera.end(); cit ++)	{
		int frame;
		(*cit)["frame"] >> frame;
		//undistort?
		Mat camera;
		(*cit)["projection"] >> camera;
		cameras.push_back(camera); //FIXME: cameras[frame-1] = camera;
	}
	
	// read and cache the whole clip
	VideoCapture clip(clipPath);
	int frameCount = clip.get(CV_CAP_PROP_FRAME_COUNT);
	frames.resize(frameCount);
	for (int fi = 0; fi < frameCount; fi++) {
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

const Mat Configuration::frame(int number)
{
	return frames[number];
}

const Mat Configuration::camera(int number)
{
	return cameras[number];
}

const int Configuration::frameCount()
{
	return frames.size();
}
