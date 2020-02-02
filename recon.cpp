// recon.cpp: main file of the program

// main header 
#include "recon.hpp"

// purely for debugging purposes (when executed with -v or -V)
#include <stdio.h>
#include <string.h>
#define logprint(config, level, ...) {if ((config).verbosity >= (level)) printf(__VA_ARGS__);}

// application entry point 
int main(int argc, char ** argv) {
	// loads the reconstruction parameters from command-line parameters and the video+calibration from external files 
	Configuration config = Configuration(argc, argv);
	logprint(config, 2, " Loaded configuration and video clip\n");

	// initializes heuristic algorithms from the supplied configuration 
	Heuristic hint(&config);

	// intitialize off-screen rendering (create OpenGL context, ...) 
	Render *render = spawnRender(hint);

	// store the points from the initial reconstruction 
	Mat points = config.reconstructedPoints();
	logprint(config, 2, " Loaded %i points\n", points.rows);

	// initialize normals to zero vectors 
	Mat normals(Mat::zeros(points.rows, 3, CV_32FC1));
	
	// iterate until the heuristic is happy with the precission
	while (hint.notHappy(points)) {

		// construct polygonized mesh 
		float alpha;
		logprint(config, 1, "Meshing...\n");	
		Mesh mesh = hint.tessellate(points, normals);
		logprint(config, 2, " %i faces.\n", mesh.faces.rows);
		if (config.verbosity >= 3)
			saveMesh(mesh, "recon_orig.obj");

		// feed the mesh into the rendering pipeline 
		render->loadMesh(mesh);

		// choose the bundles of cameras with each containing one main camera and some number of side cameras 
		logprint(config, 1, "Choosing cameras...\n");
		int cameraCount = hint.chooseCameras(mesh, config.allCameras(), *render);
		if (cameraCount == 0) {
			printf(" Heuristic has chosen no cameras, which is an error. However, we have got nothing more to do.\n");
			exit(1);
		}

		// print debug information about the selected cameras 
		if (config.verbosity >= 2) {
			for (int fa = hint.beginMain(); fa != Heuristic::sentinel; fa = hint.nextMain()) {
				printf("  main camera %i, side cameras ", fa);
				for (int fb = hint.beginSide(fa); fb != Heuristic::sentinel; fb = hint.nextSide(fa)) {
					printf("%i, ", fb);
				}
				printf("\n");
			}
		}

		// construct an improved version of the point cloud 
		logprint(config, 1, "Tracking the whole clip...\n");
		for (int fa = hint.beginMain(); fa != Heuristic::sentinel; fa = hint.nextMain()) {
			// * we now have one main camera with the index fa * 

			// load main camera's image and calculate its depth map 
			Mat originalImage = config.frame(fa);
			Mat depth = render->depth(config.camera(fa));
			if (config.verbosity >= 3) {
				char filename[300];
				snprintf(filename, 300, "frame%i.png", fa);
				saveImage(originalImage, filename);
				snprintf(filename, 300, "depth-frame%i.png", fa);
				saveImage(depth, filename, true);
			}

			// calculate optical between the main camera and each side view reprojected by our method
			MatList flows, cameras;
			for (int fb = hint.beginSide(fa); fb != Heuristic::sentinel; fb = hint.nextSide(fa)) {
				// * we now have main camera and a side view * 

				// calculate prediction frame from the side camera 
				Mat projectedImage = render->projected(config.camera(fa), config.frame(fb), config.camera(fb));
				projectedImage = mixBackground(projectedImage, originalImage, depth);

				// calculate the flow 
				Mat flow = calculateFlow(originalImage, projectedImage, config.useFarneback);
				if (config.verbosity >= 3) {
					char filename[300];
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
				
				// insert the result so that we can use it in the triangulation part 
				// note that i-th element of the flows vector corresponds to the i-th element of the cameras vector
				flows.push_back(flow);
				cameras.push_back(config.camera(fb)); 
			}

			// triangulate all the pixels 
			// note that the resulting matrix contains rows of the form (x, y, z, w, nx, ny, nz)
			Mat triangData = triangulatePixels(flows, config.camera(fa), cameras, depth);
			points.push_back(triangData.colRange(0,4));
			normals.push_back(triangData.colRange(4,7));
			cameras.push_back(config.camera(fa));
			logprint(config, 2, " After processing main frame %i: %i points\n", fa, points.rows);
		}
		// end of the for cycle going through all main cameras 

		// select a reliable subset of the points  
		if (config.verbosity >= 3)
			saveMesh(Mesh(points, Mat()), "purepoints.obj");
		hint.filterPoints(points, normals);
		logprint(config, 2, " %i filtered points\n", points.rows);
	}

	// release resources 
	delete render;

	// output the polygonized result 
	if (config.verbosity >= 3)
		saveMesh(Mesh(points, Mat()), "filteredpoints.obj");
	logprint(config, 1, "Calculating final mesh...\n");
	Mesh mesh = hint.tessellate(points, normals);
	logprint(config, 2, " %i faces\n", mesh.faces.rows);
	saveMesh(mesh, config.outFileName);
	logprint(config, 2, " Saved, done.\n");
	return 0;
}

