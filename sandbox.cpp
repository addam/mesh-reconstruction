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

void r(const std::vector<cv::Point2f> &patternPoints, std::vector<cv::Point2f> &rectifiedPatternPoints)
{
  Mat homography = Mat(Matx43f(1,0,0,
												   0,1,0,
												   0,0,1,
												   1,1,1));
  Mat rectifiedPointsMat;
  transform(patternPoints, rectifiedPointsMat, homography);
  rectifiedPatternPoints.clear();
  convertPointsFromHomogeneous(rectifiedPointsMat, rectifiedPatternPoints);
}

int main() {
	Matx41f v1(4.f,6.f,8.f,2.f), v2(6.f,16.f,14.f,4.f);
	Mat avg(v1.t());
	avg.push_back(Mat(v2.t()));// avg.push_back(v2); avg.push_back(v1);
	cout << avg << endl << avg.size[0] << " " << avg.size[1] << endl;
	/*Mat src(3, 0, CV_32F);
	Mat tmp;
	tmp = Mat(v1);
	tmp = tmp.t();
	src.push_back(tmp);src.push_back(tmp);src.push_back(tmp);*/
	/*avg.push_back(v1);
	avg.push_back(v2);
	avg.push_back(v1);
	avg.push_back(v2);
	transform(avg, src, Mat::eye(4, 4, CV_32F));*/
	//avg = avg.t();
	/*cout << src << endl;
	cout << src.depth() << endl << CV_32F << CV_32S << endl;*/
	//r(avg, res);
	cout << "orig: " << Mat(avg.row(0)) << endl << "      " << Mat(avg.row(1)) << endl;
	Mat res = Mat_<float>(avg.size[0],3);
	convertPointsFromHomogeneous(avg.reshape(4), res.reshape(3));
	cout << res.row(0) << endl << res.row(1) << endl;
	return 0;
}
	
