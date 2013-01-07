#include <opencv2/video/tracking.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "recon.hpp"

Mat calculateFlow(Mat prev, Mat next)
{
	const double pyr_scale = 0.5, poly_sigma = 1.5;
	const int levels = 5, winsize = 3, iterations = 100, poly_n = 7, flags = 0;
	Mat flow;
	Mat prev_gray, next_gray;
	cv::cvtColor(prev,prev_gray,CV_RGB2GRAY);
	cv::cvtColor(next,next_gray,CV_RGB2GRAY);
	cv::calcOpticalFlowFarneback(prev_gray, next_gray, flow, pyr_scale, levels, winsize, iterations, poly_n, poly_sigma, flags);
	return flow;
}
