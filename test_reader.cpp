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
#include <fstream>
#include <set>

#include "recon.hpp"

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

const Mat removeProjectionZ(const Mat projection)
{
	Mat result = Mat(projection.rowRange(0,2));
	result.push_back(projection.row(3));
	return result;
}

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

const Mat screenToCamera(const Mat points, const float width, const float height, const vector<float> lensDistortion)
{
	const Mat mscale(Matx22f(2.0/width, 0, 0, 2.0/height)), mneg(Matx22f(1, 0, 0, -1));
	Mat rp = points.reshape(1).t();
	rp = mneg * (mscale * rp - 1);
	Mat rad = Mat::ones(1, 2, CV_32FC1) * rp.mul(rp);
	float *radius = rad.ptr<float>(0);
	float avg = 0.0;
	for (int i=0; i < rad.cols; i++) {
		float k = sqrt(invertPoly(lensDistortion[0], lensDistortion[1], radius[i]) / radius[i]);
		rp.col(i) *= k;
		avg += k;
	}
	if (rad.cols > 0)
		cout << "Distortion " << avg/rad.cols << " on average" << endl;
	return rp;
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
	cout << "Loaded "<< bundleCount <<" bundles."<<endl;


	VideoCapture clip(clipPath);
	int delay = 1000/clip.get(CV_CAP_PROP_FPS), frameCount = clip.get(CV_CAP_PROP_FRAME_COUNT);
	int frameStep = 2;
  Size subPixWinSize(10,10), winSize(31,31);
  const int MAX_COUNT = 500;
  TermCriteria termcrit(CV_TERMCRIT_ITER|CV_TERMCRIT_EPS,20,0.03);
	Mat frame, gray[2];
	Mat projection[2];
	Mat reconstructedPoints(0, 3, CV_32FC1);
	FileNode camera = fs["camera"];
	FileNodeIterator cit=camera.begin();
	
	int frameNo;
	(*cit)["frame"] >> frameNo;
	clip.set(CV_CAP_PROP_POS_FRAMES, frameNo-1);
	clip >> frame;
  cvtColor(frame, gray[1], CV_BGR2GRAY); 
	Mat xyzwProjection;
	(*cit)["projection"] >> xyzwProjection;
	projection[1] = removeProjectionZ(xyzwProjection);
  cit += frameStep;

	namedWindow(windowName, CV_WINDOW_AUTOSIZE);
	for (int i=0; i<frameCount && cit != camera.end(); i++) {
		clip >> frame;
		int frameNo;
		(*cit)["frame"] >> frameNo;
		clip.set(CV_CAP_PROP_POS_FRAMES, frameNo-1);
		clip >> frame;
	  cvtColor(frame, gray[i%2], CV_BGR2GRAY); 
		Mat xyzwProjection;
		(*cit)["projection"] >> xyzwProjection;
		projection[i%2] = removeProjectionZ(xyzwProjection);
		
		Mat points, prevPoints;
		goodFeaturesToTrack(gray[i%2], points, MAX_COUNT, 0.01, 10);
		cornerSubPix(gray[i%2], points, subPixWinSize, Size(-1,-1), termcrit);
		vector<uchar> status;
		vector<float> err;
		calcOpticalFlowPyrLK(gray[i%2], gray[(i+1)%2], points, prevPoints, status, err, winSize, 3, termcrit, 0, 0.001);
		int pointCount = points.rows, putIndex = 0;
		double minError;
		minMaxIdx(err, &minError, NULL);
		cout << "Error: " << minError << endl;
		for (int readIndex = 0; readIndex < pointCount; readIndex ++){
			if (status[readIndex] && err[readIndex] <= 1.5*minError){ 
				if (putIndex != readIndex) {
					points.row(readIndex).copyTo(points.row(putIndex));
					prevPoints.row(readIndex).copyTo(prevPoints.row(putIndex));
					err[putIndex] = err[readIndex];
					status[putIndex] = status[readIndex];
				}
				putIndex ++;
			}
		}
		if (putIndex < pointCount){
			points.resize(putIndex);
			prevPoints.resize(putIndex);
			err.resize(putIndex);
			status.resize(putIndex);
			pointCount = putIndex;
		}
		
		Mat rp = screenToCamera(points, width, height, lensDistortion), prp = screenToCamera(prevPoints, width, height, lensDistortion);
		if (pointCount > 0) {
			Mat triangulated;
			triangulatePoints(projection[i%2], projection[(i+1)%2], rp, prp, triangulated);
			triangulated = triangulated.t();
	
			Mat reconstructedPointsRaw = Mat(triangulated.rows, 3, CV_32FC1);
			convertPointsFromHomogeneous(triangulated.reshape(4), reconstructedPointsRaw.reshape(3));
			reconstructedPoints.push_back(reconstructedPointsRaw);
			cout << reconstructedPoints.rows << " points" << endl;
			frameStep += 1;
		} else if (frameStep > 1) {
			frameStep -= 1;
		}
		cout << "Step: " << frameStep << endl;
		
		Mat projected = bundles * projection[i%2].t();
		for (int j=0; j<bundleCount; j++) {
			if (bundlesEnabled[j].count(frameNo)) {
				float *point = projected.ptr<float>(j);
				float pointX = point[0]/point[2], pointY = point[1]/point[2];
				float radiusSquared = pointX*pointX + pointY*pointY, distortionFactor = (1 + lensDistortion[0]*radiusSquared + lensDistortion[1]*radiusSquared*radiusSquared);
				pointX /= distortionFactor;
				pointY /= distortionFactor;
				circle(frame, cv::Point(width*(0.5 + pointX*0.5), height * (0.5 - pointY*0.5)), 3, Scalar(0,0,255), -1, 8);
			}
		}
		
		for (int j=0; j < points.rows; j++){
			if (status[j] && err[j] <= 1.5*minError) {
				float ax = rp.at<float>(0,j), ay = rp.at<float>(1,j), bx = prp.at<float>(0,j), by = prp.at<float>(1,j);
				float radiusSquared = ax*ax + ay*ay, distortionFactor = (1 + lensDistortion[0]*radiusSquared + lensDistortion[1]*radiusSquared*radiusSquared);
				ax = width * (0.5 + ax*0.5/distortionFactor); ay = height * (0.5 - ay*0.5/distortionFactor);
				radiusSquared = bx*bx + by*by; distortionFactor = (1 + lensDistortion[0]*radiusSquared + lensDistortion[1]*radiusSquared*radiusSquared);
				bx = width * (0.5 + bx*0.5/distortionFactor); by = height * (0.5 - by*0.5/distortionFactor);
				line(frame, cv::Point(ax, ay), cv::Point(bx, by), Scalar(255,255,0), 1, CV_AA, 0);
			}
		}

		const double pyr_scale = 0.5, poly_sigma = 1.5;
		const int levels = 5, winsize = 3, iterations = 100, poly_n = 7, flags = 0;
		Mat flow;
		cv::calcOpticalFlowFarneback(gray[0], gray[1], flow, pyr_scale, levels, winsize, iterations, poly_n, poly_sigma, flags);
		Mat bgr(flow.rows, flow.cols, CV_32FC3);
		int from_to[] = {-1,0, 0,1, 1,2};
		mixChannels(&flow, 1, &bgr, 1, from_to, 3);
		imshow(windowName, bgr);
		cit += frameStep;
		char c = cvWaitKey(delay);
	  if (c == 27)
		  break;
		else if (c > 0)
			c = cvWaitKey(100000);
	  if (c == 27)
		  break;
	}
	
	Mat indices = alphaShapeIndices(reconstructedPoints);
	cout << reconstructedPoints.rows << " points converted to " << indices.rows << " facets" << endl;
	std::ofstream os("test_reader.obj");
	for(int i=1; i <= reconstructedPoints.rows; i ++) {
		const float* row = reconstructedPoints.ptr<float>(i-1);
		os << "v " << row[0] << ' ' << row[1] << ' ' << row[2] << std::endl;
	}
	for (int i=0; i < indices.rows; i++){
		const int32_t* row = indices.ptr<int32_t>(i);
		os << "f " << row[0]+1 << ' ' << row[1]+1 << ' ' << row[2]+1 << std::endl;
	}
	os.close();
	return 0;
}
