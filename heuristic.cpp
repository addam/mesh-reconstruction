#include <opencv2/flann/flann.hpp>
#include "recon.hpp"

typedef cvflann::L2_Simple<float> Distance;
typedef std::pair<int, float> Neighbor;

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

const inline float densityFn(float dist, float radius)
{
	return (1. - dist/radius);
}

void Heuristic::filterPoints(Mat& points)
{
	printf("Filtering: Preparing neighbor table...\n");
	int pointCount = points.rows;
	Mat points3 = dehomogenize(points);
	
	const float radius = alphaVals.back()/4.;
	std::vector<int> neighborBlocks(pointCount+1, 0);
	std::vector<Neighbor> neighbors;
	neighbors.reserve(pointCount);
	// Připrav tabulku sousedů
	{
		cv::flann::GenericIndex<Distance> index = cv::flann::GenericIndex<Distance>(points3, cvflann::KDTreeIndexParams());
		cvflann::SearchParams params;
		std::vector<float> distances(pointCount, 0.), point(3, 0.);
		std::vector<int> indices(pointCount, -1);
		for (int i=0; i<pointCount; i++) {
			points3.row(i).copyTo(point);
			index.radiusSearch(point, indices, distances, radius, params);
			int writeIndex = 0;
			neighborBlocks[i] = neighbors.size();
			int end;
			for (end = 0; end < indices.size() && indices[end] >= 0; end++);
			for (int j = 0; j < end; j++) {
				if (indices[j] < i && distances[j] <= radius) // pro zaručení symetrie se berou jen předcházející sousedi
					neighbors.push_back(Neighbor(indices[j], densityFn(distances[j], radius)));
				indices[j] = -1;
			}
		}
	}
	neighborBlocks[pointCount] = neighbors.size();
	printf(" Neighbors total: %lu, %f per point.\n", neighbors.size(), ((float)neighbors.size())/pointCount);
	
	printf("Estimating local density...\n");
	// Spočítej hustotu bodů v okolí každého (vlastní vektor pomocí power iteration)
	std::vector<float> density(pointCount, 1.), densityNew(pointCount, 0.);
	double change;
	int densityIterationNo = 0;
	do {
		for (int i=0; i<pointCount; i++) {
			densityNew[i] = 0.;
		}
		double sum = 0.;
		for (int i=0; i<pointCount; i++) {
			// přičti za každého souseda jeho význam vážený podle vzdálenosti
			float densityTemp = 0.0;
			for (int j = neighborBlocks[i]; j < neighborBlocks[i+1]; j++) {
				densityTemp += density[neighbors[j].first] * neighbors[j].second;
				densityNew[neighbors[j].first] += density[i] * neighbors[j].second;
				sum += (density[i] + density[neighbors[j].first]) * neighbors[j].second;
			}
			densityNew[i] += densityTemp;
		}
		float normalizer = pointCount / sum;
		change = 0.;
		for (int i=0; i<pointCount; i++) {
			float normalizedDensity = densityNew[i] * normalizer;
			if (normalizedDensity > 2.)
				normalizedDensity = 2.; // oříznutí -- jinak by maximální hustota dosahovala tisíců
			change += pow2(density[i] - normalizedDensity);
			density[i] = normalizedDensity;
		}
		change /= pointCount;
		densityIterationNo += 1;
	} while (change > 1e-6);
	float densityLimit = 0.0;
	for (int i=0; i<pointCount; i++) {
		densityNew[i] = density[i];
		if (density[i] > densityLimit)
			densityLimit = density[i];
	}
	densityLimit = 1.0;
	printf(" Density converged in %i iterations. Limit set to: %f\n", densityIterationNo, densityLimit);
	//projdi kandidáty od nejhustších, poznamenávej si je jako přidané a dle libovůle snižuj hustotu okolním
	std::vector<int> order(pointCount, -1);
	cv::sortIdx(density, order, cv::SORT_DESCENDING);
	int writeIndex = 0;
	for (int i=0; i<pointCount; i++) {
		int ord = order[i];
		if (densityNew[ord] < densityLimit)
			continue;
		// odečti hustotu, zbav se sousedů (odečti původní hustotu, protože nová může být záporná)
		double localDensity = density[ord];
		for (int j=neighborBlocks[ord]; j<neighborBlocks[ord+1]; j++) {
			densityNew[neighbors[j].first] -= localDensity * neighbors[j].second;
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
