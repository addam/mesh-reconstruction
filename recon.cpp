#include "recon.hpp"
#include <stdio.h>
#include <string.h>
#include <opencv2/imgproc/imgproc.hpp>

#define logprint(config, level, ...) {if ((config).verbosity >= (level)) printf(__VA_ARGS__);}

int main(int argc, char ** argv) {
	//načti všechny body
	Configuration config = Configuration(argc, argv);
	logprint(config, 2, " Loaded configuration and video clip\n");
	Heuristic hint(&config);
	Render *render = spawnRender(hint);
	Mat points = config.reconstructedPoints();
	logprint(config, 2, " Loaded %i points\n", points.rows);
	Mat normals(Mat::zeros(points.rows, 3, CV_32FC1));
	
	while (hint.notHappy(points)) {
		//sestav z nich alphashape
		float alpha;
		logprint(config, 1, "Meshing...\n");
		Mesh mesh = hint.tessellate(points, normals);
		logprint(config, 2, " %i faces.\n", mesh.faces.rows);
		if (config.verbosity >= 3)
			saveMesh(mesh, "recon_orig.obj");
		render->loadMesh(mesh);

		logprint(config, 1, "Choosing cameras...\n");
		int cameraCount = hint.chooseCameras(mesh, config.allCameras());
		if (cameraCount == 0) {
			printf(" Heuristic has chosen no cameras, which is an error. However, we have got nothing more to do.\n");
			exit(1);
		}
		if (config.verbosity >= 2) {
			for (int fa = hint.beginMain(); fa != Heuristic::sentinel; fa = hint.nextMain()) {
				printf("  main camera %i, side cameras ", fa);
				for (int fb = hint.beginSide(fa); fb != Heuristic::sentinel; fb = hint.nextSide(fa)) {
					printf("%i, ", fb);
				}
				printf("\n");
			}
		}

		logprint(config, 1, "Tracking the whole clip...\n");
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
				projectedImage = mixBackground(projectedImage, originalImage, depth);
				Mat flow = calculateFlow(originalImage, projectedImage);
				//mixBackground(flow, Mat::zeros(flow.rows, flow.cols, CV_32FC4), depth);
				if (config.verbosity >= 3) {
					char filename[300];
					//nahrubo ulož výsledek
					snprintf(filename, 300, "project-frame%ifrom%i.png", fa, fb);
					Mat mask;
					cv::compare(depth, backgroundDepth, mask, cv::CMP_EQ);
					projectedImage.setTo(0, mask);
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
			Mat triangData = triangulatePixels(flows, config.camera(fa), cameras, depth);
			points.push_back(triangData.colRange(0,4));
			normals.push_back(triangData.colRange(4,7));
			cameras.push_back(config.camera(fa));
			logprint(config, 2, " After processing main frame %i: %i points\n", fa, points.rows);
		}
		if (config.verbosity >= 3)
			saveMesh(Mesh(points, Mat()), "purepoints.obj");
		hint.filterPoints(points, normals);
		logprint(config, 2, " %i filtered points\n", points.rows);
	}
	delete render;
	//vysypej triangulované body jako obj
	if (config.verbosity >= 3)
		saveMesh(Mesh(points, Mat()), "filteredpoints.obj");
	logprint(config, 1, "Calculating final mesh...\n");
	Mesh mesh = hint.tessellate(points, normals);
	logprint(config, 2, " %i faces\n", mesh.faces.rows);
	saveMesh(mesh, config.outFileName);
	logprint(config, 2, " Saved, done.\n");
	return 0;
}
