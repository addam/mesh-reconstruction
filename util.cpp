#include <opencv2/highgui/highgui.hpp>
#include <fstream>

#include "recon.hpp"

Mat triangulatePixels(const MatList flows, const Mat mainCamera, const MatList cameras, const Mat depth)
{
	//mělo by to zaznamenávat přesnost výpočtu, aspoň nějak urvat
	return Mat(0, 3, CV_32FC3);
}

void addChannel(MatList dest, const Mat src)
{
	dest.push_back(src);
}

void saveImage(const Mat image, const char *fileName)
{
	saveImage(image, fileName, false);
}

void saveImage(const Mat image, const char *fileName, bool normalize)
{
	if (normalize) {
		double min, max;
		image.reshape(1);
		minMaxIdx(image, &min, &max);
		if (max == min)
			cv::imwrite(fileName, image);
		else {
			Mat normalized = (image - min) * 255 / (max - min);
			image.reshape(3);
			normalized.reshape(3);
			cv::imwrite(fileName, normalized);
		}
	} else {
		cv::imwrite(fileName, image);
	}
}

void saveMesh(Mat points, Mat indices, const char *fileName)
{
	std::ofstream os(fileName);
	for(int i=1; i <= points.rows; i ++) {
		const float* row = points.ptr<float>(i-1);
		os << "v " << row[0]/row[3] << ' ' << row[1]/row[3] << ' ' << row[2]/row[3] << std::endl;
	}
	for (int i=0; i < indices.rows; i++){
		const int32_t* row = indices.ptr<int32_t>(i);
		os << "f " << row[0]+1 << ' ' << row[1]+1 << ' ' << row[2]+1 << std::endl;
	}
	os.close();
}
