// flow.cpp: wrapper for optical flow calculation

#include <opencv2/video/tracking.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/optflow.hpp>
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
Mat calculateFlow(Mat prev, Mat next, bool use_farneback)
{
	cv::Ptr<cv::DenseOpticalFlow> algo;
	if (use_farneback) {
		// Calculate flow using Farnebäck's algorithm and some parameters that seem to work the best
		double pyr_scale = 0.8, poly_sigma = (prev.rows+prev.cols)/1000.0;
		int levels = 10, winsize = (prev.rows+prev.cols)/100, iterations = 7, poly_n = (poly_sigma<1.5?5:7), flags = 0;
		algo = cv::FarnebackOpticalFlow::create(levels, pyr_scale, false, winsize, iterations, poly_n, poly_sigma, flags);
	} else {
		// calculate flow using the Horn&Schunck scheme
		algo = cv::optflow::createVariationalFlowRefinement();
	}
	Mat flow(prev.rows, prev.cols, CV_32FC2);
	algo->calc(prev, next, flow);
	// estimate the variance in each pixel
	Mat variance = compare(prev, flowRemap(flow, next));
	
	// combine all the values into a single matrix
	Mat mixInput[] = {flow, variance};
	Mat mixed(flow.rows, flow.cols, CV_32FC4); // opencv does weird things if channel count is not 4...
	int fromTo[] = {0,0, 1,1, 2,2, -1,3};
	cv::mixChannels(mixInput, 2, &mixed, 1, fromTo, 4);
	return mixed;
}

#else //ifdef TEST_BUILD

Mat flowRemap(Mat flow, const Mat image)
{
	for (int x=0; x < flow.cols; x++)
		flow.col(x) += cv::Scalar(x, 0);
	for (int y=0; y < flow.rows; y++)
		flow.row(y) += cv::Scalar(0, y);
	Mat remapped;
	cv::remap(image, remapped, flow, Mat(), CV_INTER_CUBIC);
	return remapped;
}

Mat calculateFlowHS(Mat prev, Mat next, int iterations, double smoothness)
{
	CvMat *velx = cvCreateMat(prev.rows, prev.cols, CV_32FC1), *vely = cvCreateMat(prev.rows, prev.cols, CV_32FC1);
	Mat flow(prev.rows, prev.cols, CV_32FC2);
	CvMat *prev_gray = cvCreateMat(prev.rows, prev.cols, CV_8UC1), *next_gray = cvCreateMat(prev.rows, prev.cols, CV_8UC1);
	cv::cvtColor(prev,Mat(prev_gray),CV_BGR2GRAY);
	cv::cvtColor(next,Mat(next_gray),CV_BGR2GRAY);
	double epsilon = 1e-10;
	CvTermCriteria crit = {CV_TERMCRIT_ITER+CV_TERMCRIT_EPS, iterations, epsilon};
	cvCalcOpticalFlowHS(prev_gray, next_gray, false, velx, vely, smoothness, crit);
	cvReleaseMat(&prev_gray);
	cvReleaseMat(&next_gray);
	int fromTo[] = {0,0, 1,1};
	Mat combi[] = {Mat(velx), Mat(vely)};
	cv::mixChannels(combi, 2, &flow, 1, fromTo, 2);
	cvReleaseMat(&velx);
	cvReleaseMat(&vely);
	return flow;
}

int main(int argc, char **argv)
{
	if (argc <= 2) {
		printf("Usage: flow <IMAGE1> <IMAGE2> [(l|w|i|n|p|s)<NUMBER>|g|h]...\n");
		exit(0);
	}
	Mat prev = cv::imread(argv[1]),
	    next = cv::imread(argv[2]);
	double pyr_scale = 0.5, poly_sigma = 1.5;
	int levels = 4, winsize = 20, iterations = 30, poly_n = 5, flags = 0;
	bool use_hornschunck = false;
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
			case 'h':
				use_hornschunck = true; break;
			default:
				fprintf(stderr, "Unrecognized option: %s\n", argv[i]);
		}
	}
	//printf("Calculating optflow between %s and %s.\n", argv[1], argv[2]);
	Mat flow;
	if (use_hornschunck) {
		printf("lambda: %g; iterations: %i;\n", poly_sigma, iterations);
		flow = calculateFlowHS(prev, next, iterations, 1./poly_sigma);
	} else {
		printf("Levels: %i; winsize: %i; iterations: %i; polyexpansion size: %i; pyramid scale: %g; sigma: %g; Gaussian: %s\n", levels, winsize, iterations, poly_n, pyr_scale, poly_sigma, (flags?"TRUE":"FALSE"));
		Mat prev_gray, next_gray;
		cv::cvtColor(prev,prev_gray,CV_BGR2GRAY);
		cv::cvtColor(next,next_gray,CV_BGR2GRAY);
		cv::calcOpticalFlowFarneback(prev_gray, next_gray, flow, pyr_scale, levels, winsize, iterations, poly_n, poly_sigma, flags);
	}

	printf("Done.\n");
	Mat mixed(flow.rows, flow.cols, CV_32FC3);
	int fromTo[] = {0,0, 1,1, -1,2};
	cv::mixChannels(&flow, 1, &mixed, 1, fromTo, 3);
	mixed = mixed*10 + 127;
	cv::imwrite("flow.png", mixed);
	
	Mat remapped = flowRemap(flow, next);
	cv::imwrite("remap.png", remapped);
	cv::imwrite("diff.png", cv::abs(prev-remapped)*10);
	printf("Diff sum: %g\n", cv::norm(cv::sum(cv::abs(prev-remapped))));
}
#endif
