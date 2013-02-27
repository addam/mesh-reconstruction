#include <opencv2/video/tracking.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "recon.hpp"

Mat calculateFlow(Mat prev, Mat next)
{
	double pyr_scale = 0.5, poly_sigma = 1.5;
	int levels = 3, winsize = 50, iterations = 150, poly_n = 9, flags = cv::OPTFLOW_FARNEBACK_GAUSSIAN;
	Mat flow;
	Mat prev_gray, next_gray;
	cv::cvtColor(prev,prev_gray,CV_RGB2GRAY);
	cv::cvtColor(next,next_gray,CV_RGB2GRAY);
	cv::calcOpticalFlowFarneback(prev_gray, next_gray, flow, pyr_scale, levels, winsize, iterations, poly_n, poly_sigma, flags);
	flags = cv::OPTFLOW_USE_INITIAL_FLOW | cv::OPTFLOW_FARNEBACK_GAUSSIAN;
	pyr_scale = 0.5;
	iterations = 50;
	levels = 1;
	poly_sigma = 1.3;
	poly_n = 3;
	winsize = 20;
//	cv::calcOpticalFlowFarneback(prev_gray, next_gray, flow, pyr_scale, levels, winsize, iterations, poly_n, poly_sigma, flags);
	return flow;
}
/*Mat calculateFlowSF(Mat prev, Mat next)
{
	Mat flow;
	int layers = 8, blocksize=30, maxflow=3;
	calcOpticalFlowSF(prev, next, flow, layers, blocksize, maxflow);
	return flow;
}*/
