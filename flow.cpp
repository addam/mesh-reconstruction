#include <opencv2/video/tracking.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <vector>

#ifdef TEST_BUILD
	#include <iostream>
	#include <fstream>
	#include <opencv2/core/core.hpp>
	#include <opencv2/highgui/highgui.hpp>
	typedef cv::Mat Mat;
#else
	#include "recon.hpp"
#endif

#ifndef TEST_BUILD
Mat calculateFlow(Mat prev, Mat next)
{
	double pyr_scale = 0.8, poly_sigma = (prev.rows+prev.cols)/1000.0;
	int levels = 100, winsize = (prev.rows+prev.cols)/100, iterations = 7, poly_n = (poly_sigma<1.5?5:7), flags = 0;//cv::OPTFLOW_FARNEBACK_GAUSSIAN;
	Mat flow;
	cv::calcOpticalFlowFarneback(prev, next, flow, pyr_scale, levels, winsize, iterations, poly_n, poly_sigma, flags);
	flags = cv::OPTFLOW_USE_INITIAL_FLOW | cv::OPTFLOW_FARNEBACK_GAUSSIAN;
	pyr_scale = 0.5;
	iterations = 50;
	levels = 1;
	poly_sigma = 1.3;
	poly_n = 3;
	winsize = 20;
	//	cv::calcOpticalFlowFarneback(prev_gray, next_gray, flow, pyr_scale, levels, winsize, iterations, poly_n, poly_sigma, flags);
	Mat certanity = compare(prev, flowRemap(flow, next));
	
	//certanity = 1 - (1/(certanity + 1));
	Mat mixInput[] = {flow, certanity};
	Mat mixed(flow.rows, flow.cols, CV_32FC4); // opencv does weird things if channel count is not 4...
	int fromTo[] = {0,0, 1,1, 2,2, -1,3};
	cv::mixChannels(mixInput, 2, &mixed, 1, fromTo, 4);
	return mixed;
}
/*Mat calculateFlowSF(Mat prev, Mat next)
{
	Mat flow;
	int layers = 8, blocksize=30, maxflow=3;
	calcOpticalFlowSF(prev, next, flow, layers, blocksize, maxflow);
	return flow;
}*/

#else //ifdef TEST_BUILD

int main(int argc, char **argv)
{
	if (argc <= 2) {
		printf("Usage: flow <IMAGE1> <IMAGE2> [(l|w|i|n|p|s)<NUMBER>|g<NUMBER>]...\n");
		exit(0);
	}
	Mat prev = cv::imread(argv[1]),
	    next = cv::imread(argv[2]);
	double pyr_scale = 0.5, poly_sigma = 1.5;
	int levels = 4, winsize = 20, iterations = 30, poly_n = 5, flags = 0;
	for (int i=3; i<argc; i++) {
		switch(argv[i][0]) {
			case 'l':
				levels = atoi(argv[i]+1); break;
			case 'w':
				winsize = atoi(argv[i]+1); break;
			case 'i':
				iterations = atoi(argv[i]+1); break;
			case 'n':
				poly_n = atoi(argv[i]+1); break;
			case 'p':
				pyr_scale = atof(argv[i]+1); break;
			case 's':
				poly_sigma = atof(argv[i]+1); break;
			case 'g':
				flags = cv::OPTFLOW_FARNEBACK_GAUSSIAN; break;
			default:
				fprintf(stderr, "Unrecognized option: %s\n", argv[i]);
		}
	}
	//printf("Calculating optflow between %s and %s.\n", argv[1], argv[2]);
	printf("Levels: %i; winsize: %i; iterations: %i; polyexpansion size: %i; pyramid scale: %g; sigma: %g; Gaussian: %s\n", levels, winsize, iterations, poly_n, pyr_scale, poly_sigma, (flags?"TRUE":"FALSE"));
	Mat prev_gray, next_gray, flow;
	cv::cvtColor(prev,prev_gray,CV_BGR2GRAY);
	cv::cvtColor(next,next_gray,CV_BGR2GRAY);
	cv::calcOpticalFlowFarneback(prev_gray, next_gray, flow, pyr_scale, levels, winsize, iterations, poly_n, poly_sigma, flags);
	
	Mat mixed(flow.rows, flow.cols, CV_32FC3);
	int fromTo[] = {0,0, 1,1, -1,2};
	cv::mixChannels(&flow, 1, &mixed, 1, fromTo, 3);
	mixed = mixed*10 + 127;
	cv::imwrite("flow.png", mixed);
	
	for (int x=0; x < flow.cols; x++)
		flow.col(x) += cv::Scalar(x, 0);
	for (int y=0; y < flow.rows; y++)
		flow.row(y) += cv::Scalar(0, y);
	Mat remapped;
	cv::remap(next, remapped, flow, Mat(), CV_INTER_CUBIC);
	cv::imwrite("remap.png", remapped);
	cv::imwrite("diff.png", cv::abs(prev-remapped)*10);
	printf("Diff sum: %g\n", cv::norm(cv::sum(cv::abs(prev-remapped))));
}
#endif
