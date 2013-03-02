#ifndef RECON_HPP
#define RECON_HPP

#include <opencv2/core/core.hpp>
#include <list>
#include <vector>
#include <utility>

typedef cv::Mat Mat;
typedef std::list<Mat> MatList;

class Configuration;
class Heuristic;

// alpha_shapes.cpp
Mat alphaShapeIndices(const Mat points);
Mat alphaShapeIndices(const Mat points, float *alpha); //'alpha' is currently just written to, not used

// flow.cpp
Mat calculateFlow(const Mat prev, const Mat next);

// util.cpp
Mat triangulatePixels(const MatList flows, const Mat mainCamera, const MatList cameras, const Mat depth); //mělo by to jako poslední kanál zaznamenávat chybovou míru, aspoň nějak urvat
Mat dehomogenize(Mat points);
void mixBackground(Mat image, const Mat background, const Mat depth);
void saveImage(const Mat image, const char *fileName);
void saveImage(const Mat image, const char *fileName, bool normalize);
void saveMesh(const Mat points, const Mat indices, const char *fileName);
void addChannel(MatList dest, const Mat src);

// configuration.cpp
class Configuration {
	public:
		Configuration(int argc, char** argv);
		~Configuration();
		Mat reconstructedPoints();
		const Mat frame(int frameNo);
		const Mat camera(int frameNo);
		const float near(int frameNo);
		const float far(int frameNo);
		const int frameCount();
	protected:
		std::vector <Mat> frames;
		std::vector <Mat> cameras;
		std::vector <float> nearVals, farVals;
		Mat bundles;
		std::vector <float> lensDistortion;
		float centerX, centerY;
};

// render_<whatever>.cpp
class Render {
	public:
		virtual ~Render() {};
		virtual void loadMesh(const Mat points, const Mat indices) = 0;
		virtual Mat projected(const Mat camera, const Mat frame, const Mat projector) = 0;
		virtual Mat depth(const Mat camera) = 0;
};
Render *spawnRender(Heuristic hint);

// heuristic.cpp
typedef std::pair <int, std::vector <int> > numberedVector;
class Heuristic {
	public:
		Heuristic(Configuration *iconfig);
		void chooseCameras();
		bool notHappy(const Mat points);
		int beginMain(); // initialize and return frame number for first main camera
		int nextMain(); // return frame number for next main camera
		int beginSide(int mainNumber); // initialize and return frame number for first side camera
		int nextSide(int mainNumber); // return frame number for next side camera
		void filterPoints(Mat& points);
		void logAlpha(float alpha);
		static const int sentinel = -1;
	protected:
		Configuration *config;
		int iteration;
		int mainIdx, sideIdx;
		std::vector <numberedVector> chosenCameras;
		std::vector <float> alphaVals;
};
#endif
