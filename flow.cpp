#include <opencv2/video/tracking.hpp>
#include "recon.hpp"

void calculateFlow(Mat prev, Mat next, Mat flows)
{
	const double pyr_scale = 0.5, poly_sigma = 1.5;
	const int levels = 5, winsize = 3, iterations = 100, poly_n = 7, flags = 0;
	Mat flow;
	calcOpticalFlowFarneback(prev, next, flow, pyr_scale, levels, winsize, iterations, poly_n, poly_sigma, flags);
	addChannel(flows, flow);
}
