#ifndef RECON_HPP
#define RECON_HPP

#include <opencv2/core/core.hpp>
typedef cv::Mat Mat;

class Configuration;
class Heuristic;

// alpha_shapes.cpp
Mat alphaShapeIndices(const Mat points);

// flow.cpp
void calculateFlow(const Mat prev, const Mat next, Mat flows);

// util.cpp
Mat triangulatePixels(const Mat flows, const Mat cameras, const Mat depth); //mělo by to jako poslední kanál zaznamenávat chybovou míru, aspoň nějak urvat
void saveImage(const Mat image, const char *fileName);
void saveMesh(const Mat points, const Mat indices, const char *fileName);
void addChannel(Mat dest, const Mat src);

// configuration.cpp
class Configuration {
	public:
		Configuration(int argc, char** argv);
		~Configuration();
		Mat reconstructedPoints();
		Mat frame(int number);
		Mat camera(int number);
};

// render_<whatever>.cpp
class Render {
	public:
		virtual ~Render() {};
		virtual Mat projected(const Mat camera, const Mat frame, const Mat projector, const Mat points, const Mat indices) = 0;
		virtual Mat depth(const Mat camera, const Mat points, const Mat indices) = 0;
};
Render *spawnRender(Heuristic hint);

// heuristic.cpp
class Heuristic {
	public:
		Heuristic(Configuration config);
		void chooseCameras();
		bool notHappy(Mat points);
		int beginMain(); // initialize and return frame number for first main camera
		int nextMain(); // return frame number for next main camera
		int beginSide(); // initialize and return frame number for first side camera
		int nextSide(); // return frame number for next side camera
		void filterPoints(Mat points);
		static const int sentinel = -1;
};
#endif
