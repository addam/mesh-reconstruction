#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <fstream>
#include <vector>

#include "recon.hpp"

//#define USE_COVAR_MATRICES

Mat dehomogenize(const Mat points) // expects points in rows, returns a new n x 3 matrix
{
	Mat result = Mat(points.rows, 3, CV_32FC1);
	const float *inp;
	float *out;
	for (int i=0; i<points.rows; i++) {
		inp = points.ptr<float>(i);
		out = result.ptr<float>(i);
		out[0] = inp[0] / inp[3];
		out[1] = inp[1] / inp[3];
		out[2] = inp[2] / inp[3];
	}
	return result;
}

Mat dehomogenize2D(const Mat points) // expects points in rows, returns a new n x 2 matrix
{
	Mat result = Mat(points.rows, 2, CV_32FC1);
	const float *inp;
	float *out;
	for (int i=0; i<points.rows; i++) {
		inp = points.ptr<float>(i);
		out = result.ptr<float>(i);
		out[0] = inp[0] / inp[2];
		out[1] = inp[1] / inp[2];
	}
	return result;
}

Mat extractCameraCenter(const Mat camera)
{
	Mat projection(3, 4, CV_32FC1);
	camera.rowRange(0,2).copyTo(projection.rowRange(0,2));
	camera.row(3).copyTo(projection.row(2));
	Mat K, R, T;
	cv::decomposeProjectionMatrix(projection, K, R, T); // FIXME: try cv::noArray()
	return T;
}

bool goodSample(const Mat image, const float x, const float y)
// throw away points whose neighboring pixels cannot be directly interpolated
{
	if (x < 0 || x > image.cols-1 || y < 0 || y > image.rows-1)
		return false;
	int ix = x, iy = y;
	return (image.at<float>(iy,ix) != backgroundDepth &&
	        image.at<float>(iy,ix+1) != backgroundDepth &&
	        image.at<float>(iy+1,ix) != backgroundDepth &&
	        image.at<float>(iy+1,ix+1) != backgroundDepth);
}

int totalIterations;
DensityPoint triangulatePixel(float x, float y, const Mat measuredPoints, const Mat invVariances, const Mat mainCameraInv, const MatList cameras, float depth) {
	Mat k(cv::Vec4f(x, y, depth, 1)); // estimated point as seen by main camera (only the 3rd coordinate may change during optimization)
	Mat p(2, cameras.size(), CV_32FC1), delta_p(2, cameras.size(), CV_32FC1); // estimated point, projected to each camera (in columns) and its derivative wrt. z
	Mat projectionDerivatives(2, cameras.size(), CV_32FC1); // derivative of homogeneous x, y projected to each camera (in columns) wrt. z in main camera's space
	Mat projectionW(cameras.size(), 4, CV_32FC1); // projection from main camera space to each camera's space (in rows); result of multiplication is just the w coordinate of each point
	float pdf = 1.0; // in the last run, save probability density of the resulting point
	#ifdef USE_COVAR_MATRICES
	std::vector<Mat> icovars;
	icovars.reserve(invVariances.rows);
	for (int i=0; i<invVariances.rows; i++) {
		icovars.push_back(invVariances.row(i).reshape(1,2));
	}
	#else
	const float *ivar = invVariances.ptr<float>(0);
	#endif
	{int i=0;	for (MatList::const_iterator camera=cameras.begin(); camera!=cameras.end(); camera++, i++) {
		Mat(camera->rowRange(0,2) * mainCameraInv.col(2)).copyTo(projectionDerivatives.col(i));
		camera->row(3).copyTo(projectionW.row(i));
	}}
	projectionW = projectionW * mainCameraInv; // now the projection is complete
	float last_delta_z = depth;
	
	for (int iterCount=0; ; iterCount++) {
		{int i=0;	for (MatList::const_iterator camera=cameras.begin(); camera!=cameras.end(); camera++, i++) {
			Mat estimatedPoint = *camera * mainCameraInv * k;
			estimatedPoint /= estimatedPoint.at<float>(3);
			estimatedPoint.rowRange(0,2).copyTo(p.col(i));
		}}
		Mat pointsW = projectionW * k;
		pointsW = pointsW.t();
		cv::divide(projectionDerivatives.row(0), pointsW, delta_p.row(0));
		cv::divide(projectionDerivatives.row(1), pointsW, delta_p.row(1));
		double firstDz = 0, secondDz = 0;
		Mat difference = p - measuredPoints;
		for (int i=0; i<delta_p.cols; i++) {
			#ifdef USE_COVAR_MATRICES
			Mat transformed = icovars[i] * delta_p.col(i);
			firstDz += difference.col(i).dot(transformed);
			secondDz += delta_p.col(i).dot(transformed);
			#else
			firstDz += delta_p.col(i).dot(difference.col(i)) * ivar[i];
			secondDz += delta_p.col(i).dot(delta_p.col(i)) * ivar[i];
			#endif
		}
		double delta_z = -firstDz/secondDz, eps = 1e-7;
		if (iterCount >= 5 || (delta_z < eps && delta_z > -eps)) {
			double exponent = 0, product_ivar = 1;
			#ifdef USE_COVAR_MATRICES
			#else
			for (int i=0; i<difference.cols; i++) {
				exponent -= difference.col(i).dot(difference.col(i));
				product_ivar *= ivar[i];
			}
			#endif
			pdf = 0.159 * product_ivar * exp(0.5*exponent);
			break;
		}
		// apply sanity constraints to delta_z (so that the point does not get too close to any of the cameras)
		float reasonableStep = 0.5, worstStep = reasonableStep;
		{int i=0;	for (MatList::const_iterator camera=cameras.begin(); camera!=cameras.end(); camera++, i++) {
			float thisStep = 1 + delta_z * camera->row(3).t().dot(mainCameraInv.col(2)) / pointsW.at<float>(i);
			if (thisStep < worstStep)
				worstStep = thisStep;
		}}
		if (worstStep < reasonableStep) {
			delta_z = (worstStep - 1) / (reasonableStep - 1);
		}
		// another heuristic so that it does not jump back and forth
		if ((delta_z > 0 && last_delta_z < 0 && 2*delta_z > -last_delta_z) ||
		    (delta_z < 0 && last_delta_z > 0 && -2*delta_z > last_delta_z))
			delta_z = -last_delta_z/2;
		/*if (totalIterations < 50) {
			double energy = 0;
			for (int i=0; i<difference.cols; i++)
				energy += difference.col(i).dot(difference.col(i)) * ivar[i];
			//printf("%g -> %g\n", delta_z, energy);
		}*/
		k.at<float>(2) += delta_z;
		last_delta_z = delta_z;
		//DEBUG totalIterations ++;
	}
	return DensityPoint(mainCameraInv*k, pdf);
}

Mat triangulatePixels(const MatList flows, const Mat mainCamera, const MatList cameras, const Mat depth)
//FIXME: needs to have access to more details about the camera: to distortion coefficients and principal point
{
	int width = depth.cols, height = depth.rows;
	Mat points(depth.rows*depth.cols, 4+3, CV_32FC1); // point \in P^3, normal (scaled by probability) \in R^3
	int pixelId=0;
	Mat mainCameraInv = mainCamera.inv();
	Mat gradient = imageGradient(depth);
	Mat pixelIndices = -Mat::ones(height, width, CV_32SC1);
	float maxDensity = 0; // the highest probability density encountered; all points will be normalized to this (TODO: wrong idea.)
	//DEBUG totalIterations = 0;
	for (int row=0; row < depth.rows; row++) {
		const float *depthRow = depth.ptr<float>(row); // you'll never get me down to Depth Row! --JP
		for (int col=0; col < depth.cols; col++) {
			if (depthRow[col] != backgroundDepth) {
				bool okay = true;
				float centerX = depth.cols/2.0, centerY = depth.rows/2.0; // FIXME: as noted above, this need not be true
				float scaleX = 2.0/depth.cols, scaleY = 2.0/depth.rows,
				      x = (col-centerX)*scaleX,
				      y = (centerY-row)*scaleY;
				Mat measuredPoints(2, cameras.size(), CV_32FC1); // points expected by the optical flow, for each camera (in columns)
				#ifdef USE_COVAR_MATRICES
				Mat invVariances(cameras.size(), 1, CV_32FC4); // inverted covariance matrices of each optical flow around this pixel, each a single row
				#else
				Mat invVariances(cameras.size(), 1, CV_32FC1); // estimated variance of each optical flow around this pixel
				#endif
				{int i=0;	for (MatList::const_iterator camera=cameras.begin(), flow=flows.begin(); camera!=cameras.end(); camera++, flow++, i++) {
					cv::Scalar_<float> fl = flow->at< cv::Scalar_<float> > (row, col);
					float flx = fl[0], fly = fl[1],
					      variance = fl[2];
					// try to sample from the projected position; if that is not meaningful, use original pixel's depth
					float z = goodSample(depth, col+flx, row+fly) ? sampleImage<float>(depth, col + flx, row + fly) : depthRow[col];
					Mat measuredPoint = *camera * mainCameraInv * Mat(cv::Vec4f(x + flx*scaleX, y + fly*scaleY, z, 1));
					#ifdef USE_COVAR_MATRICES
					Mat D = Mat::eye(3, 2, CV_32FC1);
					if (goodSample(depth, col+flx, row+fly))
						D.reshape(2).at<cv::Point>(2) = sampleImage<cv::Point>(gradient, col+flx, row+fly);
					else
						D.reshape(2).at<cv::Point>(2) = sampleImage<cv::Point>(gradient, col, row);
					Mat A = camera->rowRange(0,2).colRange(0,3) * mainCameraInv.rowRange(0,3).colRange(0,3) * D;
					A /= measuredPoint.at<float>(3);
					Mat icovarMatrix = (A * A.t()).inv() / variance;
					icovarMatrix.reshape(4, 1).copyTo(invVariances.row(i));
					#else
					invVariances.at<float>(i) = 1/variance;
					#endif
					measuredPoint /= measuredPoint.at<float>(3);
					if (measuredPoint.at<float>(2) < -1) {
						//printf(" One camera sees this point with depth %g, skipping\n", measuredPoint.at<float>(2));
						okay = false;
						break;
					}
					measuredPoint.rowRange(0,2).copyTo(measuredPoints.col(i));
				}}
				if (okay) {
					DensityPoint result = triangulatePixel(x, y, measuredPoints, invVariances, mainCameraInv, cameras, depthRow[col]);
					points.row(pixelId).colRange(0,4) = result.point.t();
					points.at<float>(pixelId, 4) = result.density; // just save the scale
					pixelIndices.at<int32_t>(row, col) = pixelId;
					if (result.density > maxDensity)
						maxDensity = result.density;
					pixelId ++;
				}
			}
		}
	}
	points.resize(pixelId);
	const int radius = 3;
	std::vector<Mat> cameraCenters(1, extractCameraCenter(mainCamera));
	for (MatList::const_iterator camera=cameras.begin(); camera!=cameras.end(); camera++) {
		cameraCenters.push_back(extractCameraCenter(*camera));
	}
	for (int i=0; i<cameraCenters.size(); i++) {
		cameraCenters[i] = cameraCenters[i].rowRange(0, 3).t() / cameraCenters[i].at<float>(3);
	}
	Mat neighborhood(0, 3, CV_32FC1);
	cv::PCA shape;
	neighborhood.reserve(4*(radius+1)*(radius+1));
	for (int row=0; row<height; row++) {
		for (int col=0; col<width; col++) {
			int pixelId = pixelIndices.at<int32_t>(row, col);
			if (pixelId < 0)
				continue;
			float pdf = points.at<float>(pixelId, 4);
			// wild guess: normalize pdf per side camera -> nth root
			if (cameras.size() > 1)
				pdf = pow(pdf, 1.0/cameras.size());
			for (int ny=row-radius; ny<=row+radius; ny++) {
				if (ny < 0 || ny >= height)
					continue;
				int32_t *idRow = pixelIndices.ptr<int32_t>(ny);
				for (int nx=col-radius; nx<=col+radius; nx++) {
					if (nx < 0 || nx >= width || idRow[nx] < 0)  // TODO: consider a circular neighborhood?
						continue;
					Mat point = points.row(idRow[nx]).colRange(0,3) / points.at<float>(idRow[nx], 3);
					//printf("point: %ix%i, neighborhood: %ix%i\n", point.rows, point.cols, neighborhood.rows, neighborhood.cols);
					neighborhood.push_back(point);
					//printf("altered neighborhood\n");
				}
			}
			Mat normal;
			if (neighborhood.rows >= 3) {
				// slow but easy
				shape(neighborhood, cv::noArray(), CV_PCA_DATA_AS_ROW);
				normal = shape.eigenvectors.row(2);
				float dot;
				for (int i=0; i<cameraCenters.size(); i++) {
					// weighting inversely to distance
					dot += 1/normal.dot(cameraCenters[i] - points.row(pixelId).colRange(0,3) / points.at<float>(pixelId, 3));
				}
				if (dot < 0)
					normal = -normal;
				neighborhood.resize(0);
			} else {
				normal = Mat::zeros(1, 3, CV_32FC1);
				for (int i=0; i<cameraCenters.size(); i++) {
					Mat vec = cameraCenters[i] - points.row(pixelId).colRange(0,3);
					normal += vec / vec.dot(vec);
				}
			}
			points.row(pixelId).colRange(4,7) = normal * pdf / cv::norm(normal);
		}
	}
	//points.colRange(4,7) /= maxDensity;
	//DEBUG printf(" Triangulated %i points using %i iterations in total, %g per point\n", points.rows, totalIterations, (float)totalIterations/points.rows);
	return points;
}

Mat compare(const Mat prev, const Mat next)
{
	std::vector<Mat> diffPyramid;
	int size = (prev.rows < prev.cols) ? prev.rows : prev.cols;
	Mat a, b;
	prev.convertTo(a, CV_32FC1);
	next.convertTo(b, CV_32FC1);
	while (1) {
		Mat diff;
		cv::absdiff(a, b, diff);
		diffPyramid.push_back(diff);
		if (size <= 2)
			break;
		cv::pyrDown(a, a);
		cv::pyrDown(b, b);
		size /= 2;
	}
	for (int i=diffPyramid.size()-2; i>=0; i--) {
		Mat upscaledDiff;
		cv::pyrUp(diffPyramid[i+1], upscaledDiff, diffPyramid[i].size());
		diffPyramid[i] += upscaledDiff;
	}
	return diffPyramid[0];
}

Mat mixBackground(const Mat image, const Mat background, const Mat depth)
// return image.first_channel if (image.second_channel > 0 and depth < backgroundDepth) else background
{
	assert(image.channels() == 3);
	assert(background.channels() == 1);
	Mat result(depth.rows, depth.cols, CV_8UC1);
	for (int i=0; i<image.rows; i++) {
		const uchar *srcrow = background.ptr<const uchar>(i),
		            *depthrow = depth.ptr<const uchar>(i);
		uchar *dstrow = result.ptr<uchar>(i);
		for (int j=0; j<image.cols; j++) {
			// black (0,0,0) in the image denotes invalid pixels; valid black would be (0,0,1)
			if (depthrow[j] == backgroundDepth || !dstrow[3*j+1]) {
				dstrow[j] = srcrow[3*j];
			}
		}
	}
	return result;
}

Mat flowRemap(const Mat flow, const Mat image)
{
	Mat flowMap(flow.rows, flow.cols, CV_32FC2);
	int fromTo[] = {0,0, 1,1};
	cv::mixChannels(&flow, 1, &flowMap, 1, fromTo, 2);
	for (int x=0; x < flow.cols; x++)
		flowMap.col(x) += cv::Scalar(x, 0);
	for (int y=0; y < flow.rows; y++)
		flowMap.row(y) += cv::Scalar(0, y);
	Mat remapped;
	
	cv::remap(image, remapped, flowMap, Mat(), CV_INTER_CUBIC);
	return remapped;
}

float sampleImage(const Mat image, float radiusSquared, const float x, const float y, char channel)
//x, y is pointing directly into pixel grid, pixel coordinates are in their corners
//warning: return -1 if coordinates are out of image
{
	if (image.isContinuous() && image.depth() == CV_8U) {
		char ch = image.channels();
		float sum = 0., weightSum = 0.;
		//sample brightness from given neighborhood
		float radius = sqrt(radiusSquared);
		for (int ny = (int)MAX(0, y - radius); ny < MIN(y + radius + 1, image.rows); ny++) {
			const uchar *row = image.ptr<uchar>(ny);
			for (int nx = (int)MAX(0, x - radius); nx < MIN(x + radius + 1, image.cols); nx++) {
				float dx = nx - x, dy = ny - y;
				if (dx*dx + dy*dy <= radiusSquared) {
					sum += row[nx*ch + channel];
					weightSum += ch;
				}
			}
		}
		if (weightSum > 0)
			return sum / weightSum;
		else
			return -1;
	} else {
		return -1;
	}
}

template <class T>
T sampleImage(const Mat image, const float x, const float y)
//x, y is pointing directly into pixel grid, pixel coordinates are in their corners
{
	if (x < 0 || x > image.cols-1 || y < 0 || y > image.rows-1) {
		throw -1;
		//return T(NAN);
	}
	// prepare weights
	float lw = fmod(x, 1), rw = 1-lw,
	      tw = fmod(y, 1), bw = 1-tw;
	if (rw == 0) {
		if (bw == 0) {
			return image.at<T>(y,x);
		} else {
			return image.at<T>(y,x)*tw + image.at<T>(y+1,x)*bw;
		}
	} else {
		if (bw == 0) {
			return image.at<T>(y,x)*lw + image.at<T>(y,x+1)*rw;
		} else {
			return (image.at<T>(y,x)*lw + image.at<T>(y,x+1)*rw)*tw + (image.at<T>(y+1,x)*lw + image.at<T>(y+1,x+1)*rw)*bw;
		}
	}
}

Mat imageGradient(const Mat image)
{
	if (image.channels() > 1) {
		Mat image_gray;
		cv::cvtColor(image, image_gray, CV_RGB2GRAY);
		return imageGradient(image_gray);
	}
	Mat grad[2];
	Sobel(image, grad[0], CV_32F, 1, 0);
	Sobel(image, grad[1], CV_32F, 0, 1);
	Mat result(image.rows, image.cols, CV_32FC2);
	int from_to[] = {0,0, 1,1};
	mixChannels(grad, 2, &result, 1, from_to, 2);
	return result;
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
	if (image.channels() > 1 && image.channels() != 3) {
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
		// DEBUG printf("writing normalized image, min: %f, max: %f\n", min, max);
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

void saveMesh(const Mesh mesh, const char *fileName)
{
	std::ofstream os(fileName);
	for(int i=0; i < mesh.vertices.rows; i ++) {
		const float* row = mesh.vertices.ptr<float>(i);
		os << "v " << row[0]/row[3] << ' ' << row[1]/row[3] << ' ' << row[2]/row[3] << std::endl;
	}
	for (int i=0; i < mesh.faces.rows; i++){
		const int32_t* row = mesh.faces.ptr<int32_t>(i);
		os << "f " << row[0]+1 << ' ' << row[1]+1 << ' ' << row[2]+1 << std::endl;
	}
	os.close();
}
