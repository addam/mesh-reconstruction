#include <opencv2/highgui/highgui.hpp>
#include <fstream>

#include "recon.hpp"

Mat triangulatePixels(const MatList flows, const MatList cameras, const Mat depth)
{
	//mělo by to jako poslední kanál zaznamenávat chybovou míru, aspoň nějak urvat
}

void addChannel(MatList dest, const Mat src)
{
	dest.push_back(src);
}

void saveImage(const Mat image, const char *fileName)
{
	cv::imwrite(fileName, image);
}

void saveMesh(Mat points, Mat indices, const char *fileName)
{
	std::ofstream os(fileName);
	for(int i=1; i <= points.rows; i ++) {
		const float* row = points.ptr<float>(i);
		os << "v " << row[0] << ' ' << row[1] << ' ' << row[2] << std::endl;
	}
	for (int i=0; i < indices.rows; i++){
		const int32_t* row = indices.ptr<int32_t>(i);
		os << "f " << row[0]+1 << ' ' << row[1]+1 << ' ' << row[2]+1 << std::endl;
	}
	os.close();
}
