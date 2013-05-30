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
	
	Mat debugIndices = alphaShapeIndices(points);
	while (hint.notHappy(points)) {
		//sestav z nich alphashape
		float alpha;
		Mat indices = alphaShapeIndices(points, &alpha);
		if (config.verbosity >= 2)
			printf(" %i faces.\n", indices.rows);
		hint.logAlpha(alpha);
		if (config.verbosity >= 3)
			saveMesh(points, indices, "recon_orig.obj");
		render->loadMesh(points, indices);

		if (config.verbosity >= 1)
			printf("Tracking the whole clip...\n");
		int fa = config.frameCount()/2;
		std::vector<Mat> frames;
		MatList cameras;
		/*cv::Size size(0,0);
		float sigma = 0.0;*/
		printf("Picking side cameras: ");
		const int span = 50;
		for (int fb = (fa<span)?0:fa-span; fb < config.frameCount() && fb <= fa+span; fb += 7) {
			if (fb == fa)
				continue;
			printf("%i, ", fb);
			Mat tmp;
			/*cv::GaussianBlur(config.frame(fb), tmp, size, sigma);*/
			config.frame(fb).copyTo(tmp);
			frames.push_back(tmp);
			cameras.push_back(config.camera(fb));
		}
		printf("\n");
		Mat tmp;
		/*cv::GaussianBlur(config.frame(fa), tmp, size, sigma);*/
		config.frame(fa).copyTo(tmp);
		char filename[30];
		snprintf(filename, 30, "frame%i.png", fa);
		saveImage(tmp, filename);
		Mat newPoints = bruteTriangulation(tmp, config.camera(fa), frames, cameras);
		points.push_back(newPoints);
		if (config.verbosity >= 2)
			printf(" After processing main frame %i: %i points\n", fa, points.rows);

		hint.filterPoints(points);
		if (config.verbosity >= 2)
			printf(" %i filtered points\n", points.rows);
	}
	delete render;
	//vysypej triangulované body jako obj
	/*if (config.verbosity >= 1)
		printf("Calculating final alpha shape...\n");
	Mat indices = alphaShapeIndices(points);
	if (config.verbosity >= 2)
		printf(" %i faces\n", indices.rows);*/
	saveMesh(points, debugIndices, "triangulated.obj");
	if (config.verbosity >= 2)
		printf(" Saved, done.\n");
	return 0;
}
