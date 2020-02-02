// recon.hpp: main header file
// contains declarations of all public functions and classes used in the program

#ifndef RECON_HPP
#define RECON_HPP

#include <opencv2/core/core.hpp>
#include <list>
#include <vector>
#include <set>
#include <utility>

#define IMIN(a,b) (((a)<(b)) ? (a) : (b))
#define IMAX(a,b) (((a)>(b)) ? (a) : (b))

// convenience typedef's (instead of 'using namespace')
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

// == alpha_shapes.cpp ==
Mat alphaShapeFaces(const Mat points);
Mat alphaShapeFaces(const Mat points, float *alpha); //'alpha' is currently just written to, not used

// == either pcl.cpp or cgal_poisson.cpp ==
Mesh poissonSurface(const Mat points, const Mat normals);

// == flow.cpp ==
Mat calculateFlow(const Mat prev, const Mat next, bool useFarneback);

// == util.cpp ==
Mat extractCameraCenter(const Mat camera);
Mat triangulatePixels(const MatList flows, const Mat mainCamera, const MatList cameras, const Mat depth);
Mat compare(const Mat prev, const Mat next);
Mat dehomogenize(Mat points);
float sampleImage(const Mat image, float radius, const float x, const float y, char c);
template <class T> T sampleImage(const Mat image, const float x, const float y); // linear sampling
Mat mixBackground(const Mat image, const Mat background, Mat &depth);
Mat flowRemap(const Mat flow, const Mat image);
void saveImage(const Mat image, const char *fileName);
void saveImage(const Mat image, const char *fileName, bool normalize);
Mesh readMesh(const char *fileName);
void saveMesh(const Mesh, const char *fileName);
Mat imageGradient(const Mat image);

// == configuration.cpp ==
class Configuration {
	public:
		Configuration(int argc, char** argv);
		~Configuration();
		Mat reconstructedPoints();
		const Mat frame(int frameNo) const; // individual frames of the video clip
		const Mat camera(int frameNo) const; // individual cameras
		const std::vector<Mat> allCameras() const;
		const float near(int frameNo); // near camera values for each frame
		const float far(int frameNo);
		const int frameCount();
		int iterationCount;
		char verbosity;
		bool useFarneback; // switch between optflow algorithms by Farnebaeck and Horn&Schunck
		float cameraThreshold; // thresholding value for camera selection
		float sceneResolution; // a parameter to modify the density of the resulting mesh
		float scalingFactor; // downsample each frame
		unsigned skipFrames; // skip input frames, for testing
		int width, height;
		char *outFileName;
		char *inMeshFile; // filename to read initial mesh from
	protected:
		const Mat projectPoints(int frame);
		void estimateExposure();
		std::vector <Mat> frames;
		std::vector <Mat> cameras;
		std::vector <float> nearVals, farVals;
		Mat bundles;
		std::vector< std::set<int> > bundlesEnabled;
		std::vector <float> lensDistortion;
		float centerX, centerY;
		bool doEstimateExposure;
};

// == render_glx.cpp (or perhaps render_<whatever>.cpp in the future) ==
class Render {
	public:
		virtual ~Render() {};
		virtual void loadMesh(const Mesh) = 0;
		virtual Mat projected(const Mat camera, const Mat frame, const Mat projector) = 0;
		virtual Mat depth(const Mat camera) const = 0;
};
Render *spawnRender(Heuristic hint);

// == heuristic.cpp ==
typedef std::pair <int, std::vector <int> > numberedVector;
class Heuristic {
	public:
		Heuristic(Configuration *iconfig);
		int chooseCameras(const Mesh mesh, const std::vector<Mat> cameras, const Render&);
		bool notHappy(const Mat points);
		int beginMain(); // initialize and return frame number for the first main camera
		int nextMain(); // return frame number for the next main camera
		int beginSide(int mainNumber); // initialize and return frame number for the first side camera
		int nextSide(int mainNumber); // return frame number for the next side camera
		void filterPoints(Mat& points, Mat& normals);
		Mesh tessellate(const Mat points, const Mat normals);
		cv::Size renderSize();
		static const int sentinel = -1;
	protected:
		Configuration *config;
		int iteration;
		int mainIdx, sideIdx;
		std::vector <numberedVector> chosenCameras;
		std::vector <float> alphaVals;
};
#endif
