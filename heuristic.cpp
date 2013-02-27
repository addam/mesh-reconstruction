#include <opencv2/flann/flann.hpp>
#include "recon.hpp"

typedef cv::flann::L2<float> Distance;

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

const inline float pow2(float x)
{
	return x * x;
}

void Heuristic::filterPoints(Mat& points)
{
	printf("Filtering...\n");
	int pointCount = points.rows;
	Mat points3(pointCount, 3, CV_32F);
	for (int i=0; i<pointCount; i++) {
		points.row(i).colRange(0,3).copyTo(points3.row(i));
		points3.row(i) /= points.at<float>(i, 3);
	}
	
	cv::flann::GenericIndex<Distance> index = cv::flann::GenericIndex<Distance>(points3, cvflann::KDTreeIndexParams());
	double change = 0.;
	const float radius = alphaVals.back()/8.;
	cvflann::SearchParams params = cvflann::SearchParams();
	std::vector<double> density(pointCount, 1.), densityNew(pointCount, 0.);
	std::vector<float> distances(pointCount, 0.), point(3, 0.);
	std::vector<int> indices(pointCount, -1), order(pointCount, -1);
	for (int i=0; i<order.size(); i++)
		order[i] = i;
	int debugNeighborsTotal = 0;
	do { //TODO: tohle je potřeba řešit nějakou řídkou maticí, nepočítat sousedy vždycky znova
		// nejlíp to uložit všechno do velkého vektoru, a každému bodu bude patřit jeho zarážka
		cv::randShuffle(order);
		double avg = 1.;
		for (int i=0; i<pointCount; i++) {
			int ord = order[i];
			points3.row(ord).copyTo(point);
			index.radiusSearch(point, indices, distances, radius, params);
			// přičti za každého jeho význam vážený podle vzdálenosti
			double densityTmp = 0.;
			for (int j=0; j<indices.size() && indices[j] >= 0; j++) {
				if (indices[j] != ord && distances[j] < radius) {
					densityTmp += density[indices[j]] * (1. - distances[j]/radius);
					debugNeighborsTotal ++;
				}
				indices[j] = -1;
			}
			densityNew[ord] = densityTmp;
			double coef = ((double)i)/((double)(i+1));
			avg = avg * coef + densityTmp / ((double)(i+1));
		}
		change = 0.;
		double coef = 1. / pointCount;
		for (int i=0; i<pointCount; i++) {
			densityNew[i] /= avg;
			change += pow2(density[i] - densityNew[i]) * coef;
			density[i] += 1.3*(densityNew[i] - density[i]); //overrelaxation
		}
		printf("Density iteration, change: %f\n", change);
	} while (change > 0.0002);
	printf("Neighbors total: %i\n", debugNeighborsTotal);
	double densityLimit = 0.0;
	for (int i=0; i<pointCount; i++) {
		densityNew[i] = density[i];
		if (density[i] > densityLimit)
			densityLimit = density[i];
	}
	densityLimit *= 0.5;
	printf("Density limit: %f\n", densityLimit);
	//projdi kandidáty od nejhustších, poznamenávej si je jako přidané a dle libovůle snižuj hustotu okolním
	cv::sortIdx(density, order, cv::SORT_DESCENDING);
	int writeIndex = 0;
	for (int i=0; i<pointCount; i++) {
		int ord = order[i];
		if (densityNew[ord] < densityLimit)
			continue;
		points3.row(ord).copyTo(point);
		index.radiusSearch(point, indices, distances, radius, params);
		// odečti hustotu, zbav se sousedů (původní hustotu, protože nová může být záporná)
		double localDensity = density[ord];
		for (int j=0; j<indices.size() && indices[j] >= 0; j++) {
			if (indices[j] != ord && distances[j] < radius)
				densityNew[indices[j]] -= localDensity * pow2(1 - distances[j]/radius);
			indices[j] = -1;
		}
		if (i > writeIndex)
			order[writeIndex] = order[i];
		writeIndex ++;
	}
	//fakticky vyfiltruj z matice všechny podhuštěné body
	std::sort(order.begin(), order.begin() + writeIndex);
	for (int i=0; i<writeIndex; i++) {
		if (order[i] > i)
			points.row(order[i]).copyTo(points.row(i)); //to můžu udělat díky tomu, že jsem indexy setřídil
	}
	points.resize(writeIndex);
}
/*
void Heuristic::filterPoints(Mat points)
{
	int pointCount = points.rows;
	Mat points3(0, 3, CV_32F);
	for (int i=0; i<pointCount; i++) {
		points3.push_back(points.row(i).colRange(0,3) / points.at<float>(i, 3));
	}
	//Mat centers;
	//cv::flann::hierarchicalClustering<float> (points3, centers, const cvflann::KMeansIndexParams& params, Distance d=Distance())
	for (int i=1; 10*i < points.rows; i++) {
		points.row(10*i).copyTo(points.row(i));
	}
	points.resize(points.rows/10);
	printf("%i filtered points\n", points.rows);
}
*/

void Heuristic::chooseCameras()
{
	chosenCameras.clear();
	for (int i=0; i < config->frameCount(); i += 33) {
		std::vector<int> linked;
		if (i - 25 >= 0)
			linked.push_back(i-25);
		if (i - 15 >= 0)
			linked.push_back(i-15);
		if (i + 15 < config->frameCount())
			linked.push_back(i+15);
		if (i + 25 < config->frameCount())
			linked.push_back(i+25);
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
