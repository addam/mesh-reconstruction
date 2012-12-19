#include <opencv2/core/core.hpp>
typedef cv::Mat Mat;

class Configuration;
class Heuristic;

Mat alphaShapeIndices(const Mat points);
Mat render(const Mat camera, const Mat frame, const Mat projector, const Mat points, const Mat indices);
Mat renderDepth(const Mat camera, const Mat points, const Mat indices);
void calculateFlow(const Mat prev, const Mat next, Mat flows); //přidá další kanál do matice flows
Mat triangulatePixels(const Mat flows, const Mat cameras, const Mat depth); //mělo by to jako poslední kanál zaznamenávat chybovou míru, aspoň nějak urvat

void saveImage(const Mat image, const char *fileName);
void saveMesh(const Mat points, const Mat indices, const char *fileName);
void addChannel(Mat dest, const Mat src);

class Configuration {
	public:
		Configuration(int argc, char** argv);
		~Configuration();
		Mat reconstructedPoints();
		Mat frame(int number);
		Mat camera(int number);
};

class Render {
	public:
		Render(Heuristic hint);
		~Render();
		Mat projected(const Mat camera, const Mat frame, const Mat projector, const Mat points, const Mat indices);
		Mat depth(const Mat camera, const Mat points, const Mat indices);
};

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

