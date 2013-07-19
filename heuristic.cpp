#include <opencv2/flann/flann.hpp>
#include "recon.hpp"
#include <map>

typedef cvflann::L2_Simple<float> Distance;
typedef std::pair<int, float> Neighbor;
const float focal = 0.5;

typedef struct {
	int index; // actual camera index in the video sequence
	float cosFromViewer, // cosine of this camera viewed from the given point
	      distance; // distance of the point to the camera, projected along the camera axis
	float viewX, viewY; // coordinates as viewed from the given point's camera
} CameraLabel;

const CameraLabel dummyLabel = {-1, 0, 0};

typedef std::vector< std::pair<CameraLabel, Mat> > LabelledCameras;

Heuristic::Heuristic(Configuration *iconfig)
{
	config = iconfig;
	iteration = 0;
}

bool Heuristic::notHappy(const Mat points)
{
	iteration ++;
	return (iteration <= config->iterationCount);
}

inline float const pow2(float x)
{
	return x * x;
}

inline unsigned const compact(unsigned short i, unsigned short j)
{
	return (unsigned(i) << sizeof(short)*CHAR_BIT) + unsigned(j);
}

inline float const densityFn(float dist, float radius)
{
	return (1. - dist/radius);
}

void Heuristic::filterPoints(Mat& points, Mat& normals)
{
	if (config->verbosity >= 1)
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
	if (config->verbosity >= 2)
		printf(" Neighbors total: %lu, %5.1g per point.\n", neighbors.size(), ((float)neighbors.size())/pointCount);
	
	if (config->verbosity >= 1)
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
	densityLimit = .7;
	if (config->verbosity >= 2)
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
			normals.row(order[i]).copyTo(normals.row(i));
	}
	points.resize(writeIndex);
	normals.resize(writeIndex);
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
float faceArea(Mat points, int ia, int ib, int ic)
{
	Mat a = points.row(ia),
	    b = points.row(ib),
	    c = points.row(ic);
	a = a.colRange(0,3) / a.at<float>(3);
	b = b.colRange(0,3) / b.at<float>(3);
	c = c.colRange(0,3) / c.at<float>(3);
	Mat e = b-a,
	    f = c-b;
	return cv::norm(e.cross(f))/2;
}

const Mat faceCamera(const Mesh mesh, int faceIdx, float far, float focal)
{
	const int32_t *vertIdx = mesh.faces.ptr<int32_t>(faceIdx);
	Mat a(mesh.vertices.row(vertIdx[0])),
	    b(mesh.vertices.row(vertIdx[1])),
	    c(mesh.vertices.row(vertIdx[2]));
	a = a.colRange(0,3) / a.at<float>(3);
	b = b.colRange(0,3) / b.at<float>(3);
	c = c.colRange(0,3) / c.at<float>(3);
	Mat normal((b-a).cross(c-b));
	float normalLength = cv::norm(normal);
	normal /= normalLength;

	// get a uniformly random camera center across the triangle
	float u1 = cv::randu<float>(), u2 = cv::randu<float>();
	if (u1 + u2 > 1) {
		u1 = 1-u1;
		u2 = 1-u2;
	}
	Mat center = a*u1 + b*u2 + c*(1-u1-u2);

	Mat RT;
	// ready, steady...
	float *n = normal.ptr<float>(0),
	      *ce = center.ptr<float>(0);
	float x = n[0], y = n[1], z = n[2];
	float xys = x*x + y*y, xy = sqrt(xys);
	// ...go!
	if (xy > 0) {
		RT = Mat(cv::Matx44f(
			z*x/xy, z*y/xy,  xy, -z*(ce[0]*x + ce[1]*y)/xy - ce[2]*xy,
			-y/xy,    x/xy,    0,  (ce[0]*y-ce[1]*x)/xy,
			-x,        -y,       z, ce[0]*x + ce[1]*y - ce[2]*z,
			0,        0,       0,  1));
	} else { // no need for rotation
		float s = (z > 0) ? 1 : -1;
		RT = Mat(cv::Matx44f(
			1, 0, 0, -ce[0],
			0, s, 0, -ce[1],
			0, 0, s, -ce[2],
			0, 0, 0, 1));
	}

	float near = 0.001;//normalLength/4; // just a value with length units...
	Mat K(cv::Matx44f(
		focal, 0, 0, 0,
		0, focal, 0, 0,
		0, 0, (near+far)/(far-near), 2*near*far/(near-far),
		0, 0, 1, 0));

	return K*RT;
}

int bisect(std::vector<float> list, float choice)
{
	//whatever.
	for (int i=0; i<list.size(); i++) {
		if (list[i] > choice)
			return i-1;
	}
	return list.size();
}

int myFind(std::vector<numberedVector> list, int index)
{
	//return -1 if index not in list
	//else return i: list[i].first == index
	for (int i=0; i<list.size(); i++) {
		if (list[i].first == index)
			return i;
	}
	return -1;
}

int myFind(std::vector<int> list, int index)
{
	for (int i=0; i<list.size(); i++) {
		if (list[i] == index)
			return i;
	}
	return -1;
}

LabelledCameras filterCameras(Mat viewer, Mat depth, const std::vector<Mat> cameras)
{
	LabelledCameras filtered;
	{int i=0; for (std::vector<Mat>::const_iterator camera=cameras.begin(); camera!=cameras.end(); camera++, i++) {
		CameraLabel label;
		label.index = i;
		Mat cameraFromViewer = viewer * extractCameraCenter(*camera);
		float *cfv = cameraFromViewer.ptr<float>(0);
		cameraFromViewer /= cfv[3];
		cfv = cameraFromViewer.ptr<float>(0);
		if (cfv[0] < -1 || cfv[0] > 1 || cfv[1] < -1 || cfv[1] > 1 || cfv[2] < -1) {
			//printf("  Failed test from viewer: %g, %g, %g\n", cfv[0], cfv[1], cfv[2]);
			continue;
		}
		label.viewX = cfv[0];
		label.viewY = cfv[1];
		
		int row = (cfv[1] + 1) * depth.rows / 2,
		    col = (cfv[0] + 1) * depth.cols / 2;
		float obstacleDepth = depth.at<float>(row, col);
		if (obstacleDepth != backgroundDepth && obstacleDepth <= cfv[2]) {
			//printf("  Failed depth test: %g >= %g\n", cfv[2], obstacleDepth);
			continue;
		}
		
		Mat viewerCenter = extractCameraCenter(viewer);
		Mat viewerFromCamera = *camera * viewerCenter;
		float *vfc = viewerFromCamera.ptr<float>(0);
		label.distance = vfc[3] / viewerCenter.at<float>(3);
		if (label.distance < 0)
			continue;
		viewerFromCamera /= vfc[3];
		vfc = viewerFromCamera.ptr<float>(0);
		if (vfc[0] < -1 || vfc[0] > 1 || vfc[1] < -1) {
			//printf("  Failed test from camera: %g, %g, %g\n", vfc[0], vfc[1], vfc[2]);
			continue;
		}
		
		label.cosFromViewer = sqrt(1 / (1 + (cfv[0]*cfv[0] + cfv[1]*cfv[1])/(focal*focal)));
		filtered.push_back(std::pair<CameraLabel, Mat>(label, *camera));
	}}
	//printf(" %i cameras passed visibility tests\n", filtered.size());
	return filtered;
}

const CameraLabel chooseMain(std::map<unsigned, float> &weights, LabelledCameras filteredCameras, float *outWeightSum)
{
	assert (filteredCameras.size() > 0);
	std::vector<float> weightSum(filteredCameras.size()+1, 0.);
	*outWeightSum = 0;
	{int i=0; for (LabelledCameras::const_iterator it = filteredCameras.begin(); it != filteredCameras.end(); it++, i++) {
		CameraLabel label = it->first;
		float weight = label.cosFromViewer/pow2(label.distance);
		*outWeightSum += weight; // use unmodified
		if (weights.count(compact(label.index, label.index)))
			weight *= 5 * filteredCameras.size();
		weightSum[i+1] = weightSum[i] + weight;
	}}
	float choice = cv::randu<float>() * weightSum.back();
	int index = bisect(weightSum, choice);
	//printf("           I shot at %g from %g and thus decided for main camera %i (at position %i), weight %g\n", choice, weightSum.back(), filteredCameras[index].first.index, index, weightSum[index+1] - weightSum[index]);
	return filteredCameras[index].first;
}

const CameraLabel chooseSide(std::map<unsigned, float> &weights, CameraLabel mainCamera, float threshold, LabelledCameras filteredCameras)
{
	assert (filteredCameras.size() > 1); // mainCamera is surely in filteredCameras and we cannot pick it
	std::vector<float> weightSum(filteredCameras.size(), 0.);
	std::vector<CameraLabel> labels;
	float actualWeightSum = 0;
	int i=0;
	for (LabelledCameras::const_iterator it = filteredCameras.begin(); it != filteredCameras.end(); it++) {
		CameraLabel label = it->first;
		if (label.index == mainCamera.index)
			continue;
		float parallax = sqrt(pow2(label.viewX - mainCamera.viewX) + pow2(label.viewY - mainCamera.viewY)) / focal;
		float weight = label.cosFromViewer * parallax / pow2(label.distance);
		actualWeightSum += weight;
		unsigned compactIndex = compact(mainCamera.index, label.index);
		if (weights.count(compactIndex) && weights[compactIndex] >= 1)
			weight *= filteredCameras.size(); // TODO: try *= filteredCameras.size() instead
		weightSum[i+1] = weightSum[i] + weight;
		labels.push_back(it->first);
		i++;
	}
	float choice = cv::randu<float>() * weightSum.back();
	int index = bisect(weightSum, choice);
	assert(index >= 0 && index < i);
	unsigned compactIndex = compact(mainCamera.index, labels[index].index);
	if (weights[compactIndex] >= 1) {
		//printf("  SKIPPED: I shot at %g from %g and thus decided for side camera %i (at position %i of %i) already picked (%g)\n", choice, weightSum.back(), labels[index].index, index, i, weights[compactIndex]);
		return dummyLabel;
	}
	weights[compact(mainCamera.index, mainCamera.index)] = 1; // just a mark
	float addWeight = (weightSum[index+1] - weightSum[index]) / (threshold * actualWeightSum);
	weights[compactIndex] += addWeight;
	float curWeight = weights[compactIndex];
	if (curWeight >= 1) {
		float parallax = sqrt(pow2(labels[index].viewX - mainCamera.viewX) + pow2(labels[index].viewY - mainCamera.viewY)) / focal;
		//printf("  PASSED:  I shot at %g from %g and thus decided for side camera %i (at position %i of %i), weight %g * %g, parallax %g\n", choice, weightSum.back(), labels[index].index, index, i, curWeight, threshold * weightSum.back(), parallax);
		return labels[index];
	} else {
		//printf("  FAILED:  I shot at %g from %g and thus decided for side camera %i (at position %i of %i), weight %g * %g\n", choice, weightSum.back(), labels[index].index, index, i, curWeight, threshold * weightSum.back());
		return dummyLabel;
	}
}

void Heuristic::chooseCameras(const Mesh mesh, const std::vector<Mat> cameras)
{
	chosenCameras.clear();
	std::vector<float> areaSum(mesh.faces.rows+1, 0.);
	for (int i=0; i<mesh.faces.rows; i++) {
		const int32_t *vertIdx = mesh.faces.ptr<int32_t>(i);
		areaSum[i+1] = areaSum[i] + faceArea(mesh.vertices, vertIdx[0], vertIdx[1], vertIdx[2]);
	}
	float totalArea = areaSum.back(),
	      average = totalArea / mesh.faces.rows;
	
	// TODO: guess from the sequence
	float samplingResolution = sqrt(cameras.size())*0.1*config->width*config->height/totalArea; // units: pixels per scene-space area
	std::vector<bool> used(false, mesh.faces.rows);
	cv::RNG random = cv::theRNG();
	Render *render = spawnRender(*this);
	render->loadMesh(mesh);
	std::vector<int> empty;
	int shotCount = 200;
	std::map<unsigned, float> weights; // indexed by calling compact(i,j) on two indices
	for (int i = 0; i < shotCount; i ++) {
		float choice = cv::randu<float>() * totalArea;
		int chosenIdx = bisect(areaSum, choice);
		//printf(" Projecting from face %i\n", chosenIdx);
		float far = 10; // FIXME: set to the farthest camera
		Mat viewer = faceCamera(mesh, chosenIdx, far, focal);
		Mat depth = render->depth(viewer);
		LabelledCameras filteredCameras = filterCameras(viewer, depth, cameras);
		if (filteredCameras.size() >= 2) {
			float mainWeightSum;
			CameraLabel mainCamera = chooseMain(weights, filteredCameras, &mainWeightSum);
			CameraLabel sideCamera = chooseSide(weights, mainCamera, shotCount * mainWeightSum/samplingResolution, filteredCameras);
			if (sideCamera.index == dummyLabel.index) {
				//printf("Missed (side).\n");
				continue;
			}
			int positionMain = myFind(chosenCameras, mainCamera.index);
			if (positionMain == -1) {
				chosenCameras.push_back(numberedVector(mainCamera.index, std::vector<int>(1, sideCamera.index)));
			} else if (myFind(chosenCameras[positionMain].second, sideCamera.index) == -1) {
				chosenCameras[positionMain].second.push_back(sideCamera.index);
			}
		} else {
			//printf(" Missed.\n");
		}
	}
	delete render;
	std::sort(chosenCameras.begin(), chosenCameras.end());
}
/*
void Heuristic::chooseCameras(const Mat points, const Mat indices)
{
	chosenCameras.clear();
	for (int i=0; i < config->frameCount(); i += 33) {
		std::vector<int> linked;
		if (i - 25 >= 0)
			linked.push_back(i-25);
		if (i - 20 >= 0)
			linked.push_back(i-20);
		if (i - 15 >= 0)
			linked.push_back(i-15);
		if (i - 10 >= 0)
			linked.push_back(i-10);
		if (i + 10 < config->frameCount())
			linked.push_back(i+10);
		if (i + 15 < config->frameCount())
			linked.push_back(i+15);
		if (i + 20 < config->frameCount())
			linked.push_back(i+20);
		if (i + 25 < config->frameCount())
			linked.push_back(i+25);
		chosenCameras.push_back(numberedVector(i, linked));
	}
}
*/
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
Mesh Heuristic::tesselate(const Mat points, const Mat normals)
{
	//TODO: add ifdefs for CGAL and PCL (and others..?)
	if (iteration <= 1) {
		float alpha;
		Mat faces = alphaShapeFaces(points, &alpha);
		alphaVals.push_back(alpha);
		return Mesh(points, faces);
	} else {
		Mesh result = poissonSurface(points, normals);
		alphaVals.push_back(alphaVals.back() / 2);
		return result;
	}
}

cv::Size Heuristic::renderSize()
{
	return cv::Size(config->width, config->height);
}
