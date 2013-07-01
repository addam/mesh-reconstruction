#ifndef RECON_HPP
#define RECON_HPP

#define WITH_CGAL
#define WITH_PCL

#include <opencv2/core/core.hpp>
#include <list>
#include <vector>
#include <set>
#include <utility>

#define IMIN(a,b) (((a)<(b)) ? (a) : (b))
#define IMAX(a,b) (((a)>(b)) ? (a) : (b))

typedef unsigned char uchar;
typedef cv::Mat Mat;
typedef struct Mesh{
	Mat vertices, faces;
	Mesh(Mat v, Mat f):vertices(v), faces(f) {};} Mesh;
typedef struct DensityPoint{
	Mat point; float density;
	DensityPoint(Mat p, float d):point(p), density(d) {};} DensityPoint;
typedef std::list<Mat> MatList;

class Configuration;
class Heuristic;

const float backgroundDepth = 1.0;

#ifdef WITH_CGAL
// alpha_shapes.cpp
Mat alphaShapeFaces(const Mat points);
Mat alphaShapeFaces(const Mat points, float *alpha); //'alpha' is currently just written to, not used
#endif

#ifdef WITH_PCL
// pcl.cpp
Mesh greedyProjection(const Mat points, const Mat normals);
Mesh poissonSurface(const Mat points, const Mat normals);
Mesh rbfSurface(const Mat points, const Mat normals);
#endif

// flow.cpp
Mat calculateFlow(const Mat prev, const Mat next);

// util.cpp
Mat extractCameraCenter(const Mat camera);
Mat triangulatePixels(const MatList flows, const Mat mainCamera, const MatList cameras, const Mat depth); //mělo by to jako poslední kanál zaznamenávat chybovou míru, aspoň nějak urvat
Mat averageNormals(const Mat points, MatList cameras);
Mat compare(const Mat prev, const Mat next);
Mat dehomogenize(Mat points);
Mat dehomogenize2D(const Mat points);
float sampleImage(const Mat image, float radius, const float x, const float y);
template <class T> T sampleImage(const Mat image, const float x, const float y); // linear sampling
void mixBackground(Mat image, const Mat background, const Mat depth);
Mat flowRemap(const Mat flow, const Mat image);
void saveImage(const Mat image, const char *fileName);
void saveImage(const Mat image, const char *fileName, bool normalize);
void saveMesh(const Mesh, const char *fileName);
Mat imageGradient(const Mat image);
void addChannel(MatList dest, const Mat src);

// configuration.cpp
class Configuration {
	public:
		Configuration(int argc, char** argv);
		~Configuration();
		Mat reconstructedPoints();
		const Mat frame(int frameNo) const;
		const Mat camera(int frameNo) const;
		const std::vector<Mat> allCameras() const;
		const float near(int frameNo);
		const float far(int frameNo);
		const int frameCount();
		int iterationCount;
		char verbosity;
	protected:
		const Mat reprojectPoints(int frame);
		void estimateExposure();
		std::vector <Mat> frames;
		std::vector <Mat> cameras;
		std::vector <float> nearVals, farVals;
		Mat bundles;
		std::vector< std::set<int> > bundlesEnabled;
		int width, height;
		std::vector <float> lensDistortion;
		float centerX, centerY;
		bool doEstimateExposure;
};

// render_<whatever>.cpp
class Render {
	public:
		virtual ~Render() {};
		virtual void loadMesh(const Mesh) = 0;
		virtual Mat projected(const Mat camera, const Mat frame, const Mat projector) = 0;
		virtual Mat depth(const Mat camera) = 0;
		//TODO: virtual Mat depth(const Mat camera, int width, int height) = 0;
};
Render *spawnRender(Heuristic hint);

// heuristic.cpp
typedef std::pair <int, std::vector <int> > numberedVector;
class Heuristic {
	public:
		Heuristic(Configuration *iconfig);
		void chooseCameras(const Mesh mesh, const std::vector<Mat> cameras);
		bool notHappy(const Mat points);
		int beginMain(); // initialize and return frame number for first main camera
		int nextMain(); // return frame number for next main camera
		int beginSide(int mainNumber); // initialize and return frame number for first side camera
		int nextSide(int mainNumber); // return frame number for next side camera
		void filterPoints(Mat& points, Mat& normals);
		Mesh tesselate(const Mat points, const Mat normals);
		static const int sentinel = -1;
	protected:
		Configuration *config;
		int iteration;
		int mainIdx, sideIdx;
		std::vector <numberedVector> chosenCameras;
		std::vector <float> alphaVals;
};
#endif
