#include <opencv2/flann/flann.hpp>
#include "recon.hpp"

Heuristic::Heuristic(Configuration *iconfig)
{
	config = iconfig;
	iteration = 0;
}
bool Heuristic::notHappy(const Mat points)
{
	iteration ++;
	return (iteration <= 1);
}
/*void Heuristic::filterPointsAlternative(Mat points)
{

	int pointCount = points.rows;
	Mat points3(0, 3, CV_32F);
	for (int i=0; i<pointsCount; i++) {
		points3.push_back(points.row(i).col_range(0,3) / points.at<float>(i, 3));
	}
	
	flann::Index_<float> index (points3, cv::KDTreeIndexParams());
	float change = 0.0;
	const float radius = alphaVals.last();
	cv::SearchParams params();
	std::vector<float> density(pointCount, 1.0), distances(pointCount, 0.0), point(3, 0.0);
	std::vector<int> indices(pointCount, -1);
	while (change > points.rows*0.1) { //TODO: tohle je potřeba řešit nějakou řídkou maticí, nepočítat sousedy vždycky znova
		for (int i=0; i<points.rows; i++) {
			points3.rows(i).copyTo(point);
			index.radiusSearch(point, indices, distances, radius, params);
			// přičti za každého jeho význam vážený podle vzdálenosti
			float densityNew = 0.0
			for (int j=0; j<indices.size() && indices[j] >= 0; j++) {
				densityNew += density[indices[j]] * (1 - distances[j]/radius);
			}
			change += pow2(densityNew - density[i]);
			density[i] = densityNew;
		}
	}
	//setřiď body podle lokální hustoty a vyházej rovnou ty, co ji mají nesmyslně malou
	//projdi kandidáty od nejhustších, přidávej je do výstupu a dle libovůle snižuj hustotu okolním

}*/
void Heuristic::filterPoints(Mat points)
{
	/*int pointCount = points.rows;
	Mat points3(0, 3, CV_32F);
	for (int i=0; i<pointsCount; i++) {
		points3.push_back(points.row(i).col_range(0,3) / points.at<float>(i, 3));
	}
	Mat centers;
	cv::flann::hierarchicalClustering<float> (points3, centers, const cvflann::KMeansIndexParams& params, Distance d=Distance())*/
	for (int i=1; 10*i < points.rows; i++) {
		points.row(10*i).copyTo(points.row(i));
	}
	points.resize(points.rows/10);
	printf("%i filtered points\n", points.rows);
}
void Heuristic::chooseCameras()
{
	chosenCameras.clear();
	for (int i=0; i < config->frameCount(); i += 20) {
		std::vector<int> linked;
		if (i - 15 >= 0)
			linked.push_back(i-15);
		if (i - 5 >= 0)
			linked.push_back(i-5);
		if (i + 5 < config->frameCount())
			linked.push_back(i+5);
		if (i + 15 < config->frameCount())
			linked.push_back(i+15);
		chosenCameras.push_back(numberedVector(i, linked));
	}
}
int Heuristic::beginMain()
{ // initialize and return frame number for first main camera
	if (chosenCameras.size() == 0)
		return Heuristic::sentinel;
	else
		return chosenCameras[mainIdx = 0].first;
}
int Heuristic::nextMain()
{ // return frame number for next main camera
	if (++mainIdx < chosenCameras.size())
		return chosenCameras[mainIdx].first;
	else
		return Heuristic::sentinel;
}
int Heuristic::beginSide(int imain)
{ // initialize and return frame number for first side camera
	if (imain != chosenCameras[mainIdx].first || chosenCameras[mainIdx].second.size() == 0)
		return Heuristic::sentinel;
	else
		return chosenCameras[mainIdx].second[sideIdx = 0];
}
int Heuristic::nextSide(int imain)
{ // return frame number for next side camera
	if (imain != chosenCameras[mainIdx].first || ++sideIdx >= chosenCameras[mainIdx].second.size())
		return Heuristic::sentinel;
	else
		return chosenCameras[mainIdx].second[sideIdx];
}
void Heuristic::logAlpha(float alpha)
{
	alphaVals.push_back(alpha);
}
