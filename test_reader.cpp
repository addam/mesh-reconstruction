#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"

#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>

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
	float center_x, center_y;
	string clipPath;
	nodeClip["width"] >> width;
	nodeClip["height"] >> height;
	nodeClip["path"] >> clipPath;
	nodeClip["center-x"] >> center_x;
	nodeClip["center-y"] >> center_y;
	
	FileNode camera = fs["camera"];
	Mat projection;
	camera["frame-200"] >> projection;
	//cout << projection<< endl;
	
	FileNode tracks = fs["tracks"];
	Mat bundle;
	tracks[0]["bundle"] >> bundle;
	VideoCapture clip(clipPath);
	int delay = 1000/clip.get(CV_CAP_PROP_FPS), frameCount = clip.get(CV_CAP_PROP_FRAME_COUNT);
	Mat frame;
	namedWindow(windowName, CV_WINDOW_AUTOSIZE);
	FileNodeIterator it = camera.begin();
	cout << bundle << endl;
	for (int i=1; i<frameCount; i++) {
		clip >> frame;
		(*it) >> projection;
		Vec4f projected = Mat_<Vec4f>(projection*bundle);
		circle(frame, Point(projected[0]*width*0.5/projected[3] + center_x, height - projected[1]*height*0.5/projected[3] - center_y), 3, Scalar(0,255,0), -1, 8);
		imshow(windowName, frame);
		it ++;
		char c = cvWaitKey(delay);
	  if (c == 27)
		  break;
	}
	return 0;
}
