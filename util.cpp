#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <fstream>

#include "recon.hpp"

const float backgroundDepth = 1.0;

const Mat removeProjectionZ(const Mat projection)
{
	Mat result = Mat(projection.rowRange(0,2));
	result.push_back(projection.row(3));
	return result;
}

Mat triangulatePixels(const MatList flows, const Mat mainCamera, const MatList cameras, const Mat depth)
{
	MatList guesses;
	Mat mainCameraInv = mainCamera.inv();
	Mat mainCameraXYW = removeProjectionZ(mainCamera);
	int pointCount = 0;
	FILE *log = fopen("logpoints.obj", "w+");
	for (MatList::const_iterator flow=flows.begin(), camera=cameras.begin(); flow!=flows.end() && camera!=cameras.end(); flow++, camera++) {
		if (pointCount == 0) { // FIXME: vybírat jen viditelné pixely obou kamer, to může dávat různé počty pro každou
			for (int y=0; y < depth.rows; y++) {
				const float *depthRow = depth.ptr<float>(y); // you'll never get me down to Depth Row! --JP
				for (int x=0; x < depth.cols; x++) {
					if (depthRow[x] != backgroundDepth)
						pointCount ++;
				}
			}
		}
		float *pp = new float[2*pointCount], *np = new float[2*pointCount];
		Mat prevPoints(2, pointCount, CV_32FC1, (void*)pp), nextPoints(2, pointCount, CV_32FC1, (void*)np);
		int i=0;
		for (int y=0; y < depth.rows; y++) {
			const float *depthRow = depth.ptr<float>(y);
			const cv::Vec2f *flowRow = flow->ptr<cv::Vec2f>(y);
			for (int x=0; x < depth.cols; x++) {
				if (depthRow[x] != backgroundDepth) {
					// pp: bod viděný z hlavní kamery. Odhad se má měnit ve směru pohledu, takže viděný bod zůstane na svém místě.
					pp[i] = (2.0*x)/depth.cols - 1.0;
					pp[pointCount + i] = -(2.0*y)/depth.rows + 1.0;
					// np: bod posunutý podle optflow, promítnutý vedlejší kamerou
					// np = camera * mainCamera^(-1) * (x + flow.x, y + flow.y, depth, 1)
					float flowX = flowRow[x][0], flowY = flowRow[x][1], z = depthRow[x];
					cv::Vec4f q(2.0*(x + flowX)/depth.cols - 1.0, -2.0*(y + flowY)/depth.rows + 1.0, z, 1);
					q = cv::Vec4f(Mat((*camera) * mainCameraInv * Mat(q)));
					np[i] = q[0] / q[3];
					np[pointCount + i] = q[1] / q[3];
					i ++;
				}
			}
		}
		Mat guess;
		cv::triangulatePoints(mainCameraXYW, removeProjectionZ(*camera), prevPoints, nextPoints, guess);
		delete pp, np;
		//nechat body homogenní, ale zajistit, že w ~ 1/chyba výpočtu
		guesses.push_back(guess);
	}
	Mat avg(pointCount, 4, CV_32FC1, 0.0); //FIXME: až se budou počty pixelů lišit, tak bacha, který s kterým průměrovat
	for (MatList::const_iterator guess=guesses.begin(); guess!=guesses.end(); guess++) {
		for (int i=0; i<pointCount; i++) {
			avg.row(i) += guess->col(i).t() / guess->at<float>(3, i);
		}
	}
	//FIXME: tohle bude potřeba nahradit složitějším výpočtem, aby výsledná chyba (1/vektor.w) dávala smysl
	avg /= guesses.size();
	//mělo by to navracet přesnost výpočtu, aspoň nějak urvat
	return avg;
}

void mixBackground(Mat image, const Mat background, const Mat depth)
{
	Mat mask;
	cv::compare(depth, backgroundDepth, mask, cv::CMP_EQ);
	background.copyTo(image, mask);
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
	if (image.channels() == 2) {
		Mat bgr(image.rows, image.cols, CV_32FC3);
		int from_to[] = {-1,0, 0,1, 1,2};
		mixChannels(&image, 1, &bgr, 1, from_to, 3);
		saveImage(bgr, fileName, normalize);
		return;
	}
	if (normalize) {
		double min, max;
		image.reshape(1);
		minMaxIdx(image, &min, &max);
		printf("writing normalized image, min: %f, max: %f\n", min, max);
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
