#include "recon.hpp"
#include <stdio.h> //DEBUG
#include <string.h> //DEBUG
#include <opencv2/imgproc/imgproc.hpp>

int main(int argc, char ** argv) {
	//načti všechny body
	Configuration config = Configuration(argc, argv);
	if (config.verbosity >= 2)
		printf(" Loaded configuration and video clip\n");
	Heuristic hint(&config);
	Render *render = spawnRender(hint);
	Mat points = config.reconstructedPoints();
	if (config.verbosity >= 2)
		printf(" Loaded %i points\n", points.rows);
	
	while (hint.notHappy(points)) {
		//sestav z nich alphashape
		float alpha;
		if (config.verbosity >= 1)
			printf("Calculating alpha shape...\n");
		Mat indices = alphaShapeIndices(points, &alpha);
		if (config.verbosity >= 2)
			printf(" %i faces.\n", indices.rows);
		hint.logAlpha(alpha);
		if (config.verbosity >= 3)
			saveMesh(points, indices, "recon_orig.obj");
		render->loadMesh(points, indices);

		hint.chooseCameras(points, indices, config.allCameras());
		if (config.verbosity >= 1)
			printf("Tracking the whole clip...\n");
		for (int fa = hint.beginMain(); fa != Heuristic::sentinel; fa = hint.nextMain()) {
			//vyber náhodné dva snímky
			//promítni druhý do kamery prvního
			Mat originalImage = config.frame(fa);
			/*
			Mat projectedPoints = config.camera(fa) * points.t();
			for (int i=0; i<projectedPoints.cols; i++) {
				float pointW = projectedPoints.at<float>(3, i), pointX = projectedPoints.at<float>(0, i)/pointW, pointY = projectedPoints.at<float>(1, i)/pointW, pointZ = projectedPoints.at<float>(2, i)/pointW;
				cv::Scalar color = (pointZ <= 1 && pointZ >= -1) ? cv::Scalar(128*(1-pointZ), 128*(pointZ+1), 0) : cv::Scalar(0, 0, 255);
				cv::circle(originalImage, cv::Point(originalImage.cols*(0.5 + pointX*0.5), originalImage.rows * (0.5 - pointY*0.5)), 3, color, -1, 8);
			}*/
			Mat depth = render->depth(config.camera(fa));
			if (config.verbosity >= 3) {
				char filename[300];
				snprintf(filename, 300, "frame%i.png", fa);
				saveImage(originalImage, filename);
				snprintf(filename, 300, "depth-frame%i.png", fa);
				saveImage(depth, filename, true);
			}
			MatList flows, cameras;
			for (int fb = hint.beginSide(fa); fb != Heuristic::sentinel; fb = hint.nextSide(fa)) {
				Mat projectedImage = render->projected(config.camera(fa), config.frame(fb), config.camera(fb));
				mixBackground(projectedImage, originalImage, depth);
				Mat flow = calculateFlow(originalImage, projectedImage);
				//mixBackground(flow, Mat::zeros(flow.rows, flow.cols, CV_32FC4), depth);
				flow += cv::Scalar(0,0,0,1);
				if (config.verbosity >= 3) {
					char filename[300];
					//nahrubo ulož výsledek
					snprintf(filename, 300, "project-frame%ifrom%i.png", fa, fb);
					saveImage(projectedImage, filename);
					snprintf(filename, 300, "flow-frame%ifrom%i.png", fa, fb);
					saveImage(flow, filename, true);
					Mat remapped = flowRemap(flow, projectedImage);
					snprintf(filename, 300, "frame%ifrom%i-remapped.png", fa, fb);
					saveImage(remapped, filename);
					snprintf(filename, 300, "frame%ifrom%i-remap-error.png", fa, fb);
					saveImage(compare(originalImage, remapped), filename, true);
				}
				
				flows.push_back(flow);
				cameras.push_back(config.camera(fb));
			}
			//trianguluj všechny pixely
			points.push_back(triangulatePixels(flows, config.camera(fa), cameras, depth));
			if (config.verbosity >= 2)
				printf(" After processing main frame %i: %i points\n", fa, points.rows);
		}
		hint.filterPoints(points);
		if (config.verbosity >= 2)
			printf(" %i filtered points\n", points.rows);
	}
	delete render;
	//vysypej triangulované body jako obj
	if (config.verbosity >= 1)
		printf("Calculating final alpha shape...\n");
	Mat indices = alphaShapeIndices(points);
	if (config.verbosity >= 2)
		printf(" %i faces\n", indices.rows);
	saveMesh(points, indices, "triangulated.obj");
	if (config.verbosity >= 2)
		printf(" Saved, done.\n");
	return 0;
}
