#include "opencv2/core/core.hpp"
#include "opencv2/video/tracking.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"

#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <set>

using namespace cv;
using namespace std;

static bool readCameraMatrix(const string& filename,
                             Mat& cameraMatrix, Mat& distCoeffs,
                             Size& calibratedImageSize )
{
    FileStorage fs(filename, FileStorage::READ);
    fs["image_width"] >> calibratedImageSize.width;
    fs["image_height"] >> calibratedImageSize.height;
    fs["distortion_coefficients"] >> distCoeffs;
    fs["camera_matrix"] >> cameraMatrix;

    if( distCoeffs.type() != CV_64F )
        distCoeffs = Mat_<double>(distCoeffs);
    if( cameraMatrix.type() != CV_64F )
        cameraMatrix = Mat_<double>(cameraMatrix);

    return true;
}

static bool readModelViews( const string& filename, vector<Point3f>& box,
                           vector<string>& imagelist,
                           vector<Rect>& roiList, vector<Vec6f>& poseList )
{
    imagelist.resize(0);
    roiList.resize(0);
    poseList.resize(0);
    box.resize(0);
    
    FileStorage fs(filename, FileStorage::READ);
    if( !fs.isOpened() )
        return false;
    fs["box"] >> box;
    
    FileNode all = fs["views"];
    if( all.type() != FileNode::SEQ )
        return false;
    FileNodeIterator it = all.begin(), it_end = all.end();
    
    for(; it != it_end; ++it)
    {
        FileNode n = *it;
        imagelist.push_back((string)n["image"]);
        FileNode nr = n["roi"];
        roiList.push_back(Rect((int)nr[0], (int)nr[1], (int)nr[2], (int)nr[3]));
        FileNode np = n["pose"];
        poseList.push_back(Vec6f((float)np[0], (float)np[1], (float)np[2],
                                 (float)np[3], (float)np[4], (float)np[5]));
    }
    
    return true;
}

int main(int argc, char** argv) {
	const char defaultFile[] = "tracks/rotunda.yaml", windowName[] = "Reader";
	FileStorage fs(((argc>1) ? argv[1]: defaultFile), FileStorage::READ);
	if (!fs.isOpened())
		return 1;
	FileNode nodeClip = fs["clip"];
	int width, height;
	float centerX, centerY;
	string clipPath;
	nodeClip["width"] >> width;
	nodeClip["height"] >> height;
	nodeClip["path"] >> clipPath;
	nodeClip["center-x"] >> centerX;
	nodeClip["center-y"] >> centerY;
	vector<float> lensDistortion;
	nodeClip["distortion"] >> lensDistortion;
	//cout << "Distortion: " << lensDistortion[0] << ", " << lensDistortion[1] << endl;
	
	FileNode camera = fs["camera"];
	Mat projection;
	
	FileNode tracks = fs["tracks"];
	Mat bundle, bundles(0, 4, CV_32FC1);
	int bundleCount = 0;
	vector< set<int> > bundlesEnabled;
	for (FileNodeIterator it = tracks.begin(); it != tracks.end(); it++){
		(*it)["bundle"] >> bundle;
		vector<int> enabledFrames;
		(*it)["frames-enabled"] >> enabledFrames;
		set<int> enabledFramesSet(enabledFrames.begin(), enabledFrames.end());
		bundlesEnabled.push_back(enabledFramesSet);
		bundle = bundle.t();
		bundles.push_back(bundle);
		bundleCount += 1;
	}
	bundles = bundles.t();
	cout << "Loaded "<< bundleCount <<" bundles."<<endl;


	VideoCapture clip(clipPath);
	int delay = 1000/clip.get(CV_CAP_PROP_FPS), frameCount = clip.get(CV_CAP_PROP_FRAME_COUNT);
  Size subPixWinSize(10,10), winSize(31,31);
  const int MAX_COUNT = 500;
  TermCriteria termcrit(CV_TERMCRIT_ITER|CV_TERMCRIT_EPS,20,0.03);
	Mat frame, nextFrame, gray[3];
	//points[1] are current, points[0] and [2] are tracked to the previous and the next frame, respectively
	vector<Point2f> points[3];
	FileNodeIterator cit=camera.begin();
	cit ++;
	clip >> nextFrame;
  cvtColor(nextFrame, gray[1], CV_BGR2GRAY); 
	clip >> nextFrame;
  cvtColor(nextFrame, gray[2], CV_BGR2GRAY); 

	namedWindow(windowName, CV_WINDOW_AUTOSIZE);
	for (int i=3; i<frameCount && cit != camera.end(); i++) {
		// gray[(i+2)%3] is the current frame, (i+1)%3 previous, i%3 the next one (but we must load it right now)
		nextFrame.copyTo(frame);
		clip >> nextFrame;
	  cvtColor(nextFrame, gray[i%3], CV_BGR2GRAY); 
		goodFeaturesToTrack(gray[(i+2)%3], points[1], MAX_COUNT, 0.01, 10, Mat(), 3, 0, 0.04);
		cornerSubPix(gray[(i+2)%3], points[1], subPixWinSize, Size(-1,-1), termcrit);
		
		vector<uchar> status[2];
		vector<float> err[2];
		calcOpticalFlowPyrLK(gray[(i+2)%3], gray[(i+1)%3], points[1], points[0], status[0], err[0], winSize, 3, termcrit, 0, 0, 0.001);
		calcOpticalFlowPyrLK(gray[(i+2)%3], gray[i%3], points[1], points[2], status[1], err[1], winSize, 3, termcrit, 0, 0, 0.001);
		int frameNo;
		sscanf((*cit).name().c_str(), "frame-%d", &frameNo);
		while (i < frameNo && i < frameCount) {
			clip >> frame;
			i ++;
		}
		(*cit) >> projection;
		Mat projected = projection*bundles;
		for (int col=0; col<bundleCount; col++) {
			if (bundlesEnabled[col].count(i)) {
				float pointX = projected.at<float>(0,col)/projected.at<float>(3,col), pointY = projected.at<float>(1,col)/projected.at<float>(3,col);
				/*float radiusSquared = pointX*pointX + pointY*pointY, distortionFactor = (1 + lensDistortion[0]*radiusSquared + lensDistortion[1]*radiusSquared*radiusSquared);
				pointX /= distortionFactor;
				pointY /= distortionFactor;*/
				circle(frame, Point(pointX*width*0.5 + centerX, height - pointY*height*0.5 - centerY), 1, Scalar(0,255,0), -1, 8);
			}
		}
		
		for (int i=0; i<points[1].size(); i++){
			if (status[0][i])
				line(frame, points[0][i], points[1][i], Scalar(255,255,0), 1, CV_AA, 0);
			if (status[1][i])
				line(frame, points[1][i], points[2][i], Scalar(255,255,0), 1, CV_AA, 0);
		}
		for (char i=0; i<2; i++){
			points[i].clear();
			status[i].clear();
			err[i].clear();
		}
		points[2].clear();
		
		imshow(windowName, frame);
		cit ++;
		char c = cvWaitKey(delay);
	  if (c == 27)
		  break;
		else if (c > 0)
			c = cvWaitKey(10000);
	  if (c == 27)
		  break;
	}
	return 0;
}
