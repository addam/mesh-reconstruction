#include <opencv2/video/tracking.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "recon.hpp"

Mat calculateFlow(Mat prev, Mat next)
{
	double pyr_scale = 0.5, poly_sigma = 1.5;
	int levels = 3, winsize = 30, iterations = 5, poly_n = 9, flags = 0;
	Mat flow;
	Mat prev_gray, next_gray;
	cv::cvtColor(prev,prev_gray,CV_RGB2GRAY);
	cv::cvtColor(next,next_gray,CV_RGB2GRAY);
	cv::calcOpticalFlowFarneback(prev_gray, next_gray, flow, pyr_scale, levels, winsize, iterations, poly_n, poly_sigma, flags);
	flags = cv::OPTFLOW_USE_INITIAL_FLOW | cv::OPTFLOW_FARNEBACK_GAUSSIAN;
	pyr_scale = 0.75;
	iterations = 50;
	levels = 3;
	poly_sigma = 1.1;
	poly_n = 2;
	winsize = 6;
	cv::calcOpticalFlowFarneback(prev_gray, next_gray, flow, pyr_scale, levels, winsize, iterations, poly_n, poly_sigma, flags);
	return flow;
}
