#ifndef RECON_CPP
#define RECON_CPP

#include "recon.hpp"

int main(int argc, char ** argv) {
	//načti všechny body
	Configuration config = Configuration(argc, argv);
	Heuristic hint(config);
	Render render(hint);
	
	Mat points = config.reconstructedPoints();
	while (hint.notHappy(points)) {
		//sestav z nich alphashape
		Mat indices = alphaShapeIndices(points);
		
		hint.chooseCameras();
		for (int fa = hint.beginMain(); fa != Heuristic::sentinel; fa = hint.nextMain()) {
			//vyber náhodné dva snímky
			//promítni druhý do kamery prvního
			Mat originalImage = config.frame(fa);
			saveImage(originalImage, "frame20.png"); //DEBUG
			Mat depth = render.depth(config.camera(fa), points, indices);
			saveImage(depth, "frame20depth.png"); //DEBUG
			Mat flows, cameras;
			for (int fb = hint.beginSide(); fb != Heuristic::sentinel; fb = hint.nextSide()) {
				Mat projectedImage = render.projected(config.camera(fa), config.camera(fb), config.frame(fb), points, indices);
				//nahrubo ulož výsledek
				saveImage(projectedImage, "frame75to20.png"); //DEBUG
				//spočítej Farnebackův optical flow, pro začátek
				calculateFlow(originalImage, projectedImage, flows);
				addChannel(cameras, config.camera(fb));
			}
			//trianguluj všechny pixely
			points.push_back(triangulatePixels(flows, cameras, depth));
		}
		hint.filterPoints(points);
	}
	//vysypej triangulované body jako obj
	Mat indices = alphaShapeIndices(points);
	saveMesh(points, indices, "triangulated.obj");
	return 0;
}
#endif
