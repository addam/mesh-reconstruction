// util.cpp: various functions for geometry (including the triangulation) and image processing

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <fstream>
#include <vector>
#include <cstring>

#include "recon.hpp"

#define USE_COVAR_MATRICES

// convert 3D homogeneous points to their Cartesian representation
// expects points in rows, returns a new n x 3 matrix
Mat dehomogenize(const Mat points) 
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

// Get the center of the given camera matrix
// expects a 4x4 matrix, outputs a 4x1 column vector
Mat extractCameraCenter(const Mat camera)
{
	Mat projection(3, 4, CV_32FC1);
	camera.rowRange(0,2).copyTo(projection.rowRange(0,2));
	camera.row(3).copyTo(projection.row(2));
	Mat K, R, T;
	cv::decomposeProjectionMatrix(projection, K, R, T);
	return T;
}

// check if all four nearest pixels are defined in the image (a depth map), so that bilinear interpolation can be directly performed
bool goodSample(const Mat image, const float x, const float y)
{
	int ix = x, iy = y;
	if (ix <= 0 || ix >= image.cols-1 || iy <= 0 || iy >= image.rows-1)
		return false;
	return (image.at<float>(iy,ix) != backgroundDepth &&
	        image.at<float>(iy,ix+1) != backgroundDepth &&
	        image.at<float>(iy+1,ix) != backgroundDepth &&
	        image.at<float>(iy+1,ix+1) != backgroundDepth);
}

// Triangulate a 3D homogeneous point at given pixel position
// x, y: camera-space positions in [-1; 1]
// measuredPoints: 2D points in columns, i-th column corresponding to cameras[i]
// invVariances: (if USE_COVAR_MATRICES) data joined from all the covariance matrices, each row corresponding to a side camera with the same index
//               (if not USE_COVAR_MATRICES) a vector of \sigma^{-2}, each corresponding to a side camera with the same index
// cameras: side cameras, list of matrices
// depth: initial depth estimate
DensityPoint triangulatePixel(float x, float y, const Mat measuredPoints, const Mat invVariances, const Mat mainCameraInv, const MatList cameras, float depth) {
	// estimated point as seen by main camera (only the 3rd coordinate may change during optimization)
	Mat k(cv::Vec4f(x, y, depth, 1)); 
	// estimated point, projected to each camera (in columns) and its derivative wrt. z
	Mat p(2, cameras.size(), CV_32FC1), delta_p(2, cameras.size(), CV_32FC1); 
	// derivative of homogeneous x, y projected to each camera (in columns) wrt. z in main camera's space
	Mat projectionDerivatives(2, cameras.size(), CV_32FC1); 
	// projection from main camera space to each camera's space (in rows); result of multiplication is just the w coordinate of each point
	Mat projectionW(cameras.size(), 4, CV_32FC1); 
	// in the last run, save probability density of the resulting point to this variable
	float pdf = 1.0; 
	
	#ifdef USE_COVAR_MATRICES
	std::vector<Mat> icovars;
	icovars.reserve(invVariances.rows);
	for (int i=0; i<invVariances.rows; i++) {
		icovars.push_back(invVariances.row(i).reshape(1,2));
	}
	#else
	const float *ivar = invVariances.ptr<float>(0);
	#endif
	
	// initialize the projectionDerivatives and projectionW matrices
	{int i=0;	for (MatList::const_iterator camera=cameras.begin(); camera!=cameras.end(); camera++, i++) {
		Mat(camera->rowRange(0,2) * mainCameraInv.col(2)).copyTo(projectionDerivatives.col(i));
		camera->row(3).copyTo(projectionW.row(i));
	}}
	projectionW = projectionW * mainCameraInv;
	// * now the projection is complete *
	
	// size of the last step; may be used for some heuristics during the minimization
	float last_delta_z = depth;
	
	// minimize the energy function
	for (int iterCount=0; ; iterCount++) {
		// calculate projected positions from side cameras of the current point estimate
		{int i=0;	for (MatList::const_iterator camera=cameras.begin(); camera!=cameras.end(); camera++, i++) {
			Mat estimatedPoint = *camera * mainCameraInv * k;
			estimatedPoint /= estimatedPoint.at<float>(3);
			estimatedPoint.rowRange(0,2).copyTo(p.col(i));
		}}
		
		// calculate the third column of the Jacobian matrix J_{C^i}, value for each side camera C^i stored in column i
		Mat pointsW = projectionW * k;
		pointsW = pointsW.t();
		cv::divide(projectionDerivatives.row(0), pointsW, delta_p.row(0));
		cv::divide(projectionDerivatives.row(1), pointsW, delta_p.row(1));
		
		// calculate the first and the second derivative at the current point
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
		
		// calculate the update step and end if it would be small enough
		double delta_z = -firstDz/secondDz, eps = 1e-7;
		if (iterCount >= 50 || (delta_z < eps && delta_z > -eps)) {
			// calculate the combined probability of the result
			double exponent = 0, product_ivar = 1;
			#ifdef USE_COVAR_MATRICES
			for (int i=0; i<difference.cols; i++) {
				Mat transformed = icovars[i] * difference.col(i);
				exponent -= difference.col(i).dot(transformed);
				product_ivar *= cv::determinant(icovars[i]);
			}
			#else
			for (int i=0; i<difference.cols; i++) {
				exponent -= difference.col(i).dot(difference.col(i));
				product_ivar *= ivar[i];
			}
			#endif
			pdf = 0.159 * product_ivar * exp(0.5*exponent);
			break;
		}
		
		// DISABLED: minimization heuristics and camera constraints
		// apply sanity constraints to delta_z (so that the point does not get too close to any of the cameras)
		//float reasonableStep = 0.5, worstStep = reasonableStep;
		//{int i=0;	for (MatList::const_iterator camera=cameras.begin(); camera!=cameras.end(); camera++, i++) {
		//	float thisStep = 1 + delta_z * camera->row(3).t().dot(mainCameraInv.col(2)) / pointsW.at<float>(i);
		//	if (thisStep < worstStep)
		//		worstStep = thisStep;
		//}}
		//if (worstStep < reasonableStep) {
		//	delta_z = (worstStep - 1) / (reasonableStep - 1);
		//}
		// another heuristic so that it does not jump back and forth
		//if ((delta_z > 0 && last_delta_z < 0 && 2*delta_z > -last_delta_z) ||
		//    (delta_z < 0 && last_delta_z > 0 && -2*delta_z > last_delta_z))
		//	delta_z = -last_delta_z/2;
		k.at<float>(2) += delta_z;
		last_delta_z = delta_z;
	}
	return DensityPoint(mainCameraInv*k, pdf);
}

// Triangulate all available pixels of the main camera's frame
Mat triangulatePixels(const MatList flows, const Mat mainCamera, const MatList cameras, const Mat depth)
{
	int width = depth.cols, height = depth.rows;
	
	// point \in P^3, normal (scaled by probability) \in R^3
	Mat points(depth.rows*depth.cols, 4+3, CV_32FC1);
	int pixelId=0;
	Mat mainCameraInv = mainCamera.inv();
	#ifdef USE_COVAR_MATRICES
	Mat gradient = imageGradient(depth);
	#endif
	Mat pixelIndices = -Mat::ones(height, width, CV_32SC1);

	for (int row=0; row < depth.rows; row++) {
		const float *depthRow = depth.ptr<float>(row); // you'll never get me down to Depth Row! --Judas Priest
		for (int col=0; col < depth.cols; col++) {
			if (depthRow[col] != backgroundDepth) {
				bool okay = true;
				float centerX = depth.cols/2.0, centerY = depth.rows/2.0;
				float scaleX = 2.0/depth.cols, scaleY = 2.0/depth.rows,
				      x = (col-centerX)*scaleX,
				      y = (centerY-row)*scaleY;
	      // points expected by the optical flow, for each side camera (in columns)
				Mat measuredPoints(2, cameras.size(), CV_32FC1);
				
				#ifdef USE_COVAR_MATRICES
				// inverted covariance matrices of each optical flow around this pixel, each a single row
				Mat invVariances(cameras.size(), 1, CV_32FC4);
				#else
				// estimated variance of each optical flow around this pixel
				Mat invVariances(cameras.size(), 1, CV_32FC1);
				#endif
				
				// process each side camera separately and calculate its measured point s^i using the optical flow
				{int i=0;	for (MatList::const_iterator camera=cameras.begin(), flow=flows.begin(); camera!=cameras.end(); camera++, flow++, i++) {
					// get optical flow and its estimated variance at the given pixel
					cv::Scalar_<float> fl = flow->at< cv::Scalar_<float> > (row, col);
					float flx = fl[0], fly = fl[1],
					      variance = fl[2];
					
					// try to sample from the projected position; if that is not meaningful, use original pixel's depth
					float z = goodSample(depth, col+flx, row+fly) ? sampleImage<float>(depth, col + flx, row + fly) : depthRow[col];
					Mat measuredPoint = *camera * mainCameraInv * Mat(cv::Vec4f(x + flx*scaleX, y + fly*scaleY, z, 1));
					
					#ifdef USE_COVAR_MATRICES
					// get affine matrix of the raycast mapping (image coordinates to camera space)
					Mat D = Mat::eye(3, 2, CV_32FC1);
					if (goodSample(depth, col+flx, row+fly))
						D.reshape(2).at<cv::Point>(2) = sampleImage<cv::Point>(gradient, col+flx, row+fly);
					else
						D.reshape(2).at<cv::Point>(2) = sampleImage<cv::Point>(gradient, col, row);
					// combine it with affine mappings of the camera back-projection and projection
					Mat A = camera->rowRange(0,2).colRange(0,3) * mainCameraInv.rowRange(0,3).colRange(0,3) * D;
					A /= measuredPoint.at<float>(3);
					// calculate the inverse of the covariance matrix
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
					// save the density of the point to be processed later in this function
					points.at<float>(pixelId, 4) = result.density;
					pixelIndices.at<int32_t>(row, col) = pixelId;
					pixelId ++;
				}
			}
		}
	}
	// release unnecessary matrix rows
	points.resize(pixelId);
	
	// == BEGIN Estimate normals from neighborhood in the main frame ==

	// half size of the square neighborhood to be considered
	const int radius = 10;
	// centers of all side cameras, used to obtain correct normal orientation
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
	
	// estimate the normal for each triangulated point
	for (int row=0; row<height; row++) {
		for (int col=0; col<width; col++) {
			// get index of the point triangulated from this position (or skip if no such point)
			int pixelId = pixelIndices.at<int32_t>(row, col);
			if (pixelId < 0)
				continue;
			
			// the density value as calculated previously
			float pdf = points.at<float>(pixelId, 4);
			// wild guess: normalize pdf per side camera -> nth root
			if (cameras.size() > 1)
				pdf = pow(pdf, 1.0/cameras.size());
			
			// add all neighbor points to the neighborhood matrix
			for (int ny=row-radius; ny<=row+radius; ny++) {
				if (ny < 0 || ny >= height)
					continue;
				int32_t *idRow = pixelIndices.ptr<int32_t>(ny);
				for (int nx=col-radius; nx<=col+radius; nx++) {
					// check that a point corresponding to this neighbor exists
					if (nx < 0 || nx >= width || idRow[nx] < 0)
						continue;
					Mat point = points.row(idRow[nx]).colRange(0,3) / points.at<float>(idRow[nx], 3);
					neighborhood.push_back(point);
				}
			}
			
			// calculate the normal based on the neighborhood
			Mat normal;
			if (neighborhood.rows >= 3) {
				// apply PCA to the neighbors: slow but easy
				shape(neighborhood, cv::noArray(), CV_PCA_DATA_AS_ROW);
				// normal is the smallest eigenvector, up to flipping (and scale, in this implementation)
				normal = shape.eigenvectors.row(2);
				float dot;
				for (int i=0; i<cameraCenters.size(); i++) {
					// weighting of cameras inversely to distance
					dot += 1/normal.dot(cameraCenters[i] - points.row(pixelId).colRange(0,3) / points.at<float>(pixelId, 3));
				}
				
				// if the majority of the cameras views the normal from the back, flip it
				if (dot < 0)
					normal = -normal;
				
				// clear the neighborhood
				neighborhood.resize(0);
			} else {
				// if not enough neighbors available, try to guess a normal from the camera centers
				normal = Mat::zeros(1, 3, CV_32FC1);
				for (int i=0; i<cameraCenters.size(); i++) {
					Mat vec = cameraCenters[i] - points.row(pixelId).colRange(0,3);
					normal += vec / vec.dot(vec);
				}
			}
			
			// normalize the normal and scale it according to the triangulation probability
			points.row(pixelId).colRange(4,7) = normal * pdf / cv::norm(normal);
		}
	}

	return points;
}

// estimate the variance given a reference image and an image remapped by the optical flow
Mat compare(const Mat prev, const Mat next)
{
	std::vector<Mat> diffPyramid;
	int size = (prev.rows < prev.cols) ? prev.rows : prev.cols;
	Mat a, b;
	prev.convertTo(a, CV_32FC1);
	next.convertTo(b, CV_32FC1);
	
	// go up the pyramid and calculate the L1 difference between the downscaled versions of the images
	while (1) {
		Mat diff;
		cv::absdiff(a, b, diff);
		diffPyramid.push_back(diff);
		if (size <= 2)
			// repeat until the topmost level of the pyramid
			break;
		cv::pyrDown(a, a);
		cv::pyrDown(b, b);
		size /= 2;
	}
	
	// go down and sum up all the differences to each pixel
	for (int i=diffPyramid.size()-2; i>=0; i--) {
		Mat upscaledDiff;
		cv::pyrUp(diffPyramid[i+1], upscaledDiff, diffPyramid[i].size());
		diffPyramid[i] += upscaledDiff;
	}
	
	return diffPyramid[0];
}

// mask out areas where the result of raycasting was undefined
// return image.first_channel if (image.second_channel > 0 and depth < backgroundDepth) else background
// set depth = backgroundDepth in the masked out areas
Mat mixBackground(const Mat image, const Mat background, Mat &depth)
{
	assert(image.channels() == 3);
	assert(background.channels() == 1);
	Mat result(depth.rows, depth.cols, CV_8UC1);
	for (int i=0; i<image.rows; i++) {
		const uchar *srcrow = image.ptr<const uchar>(i),
		            *bgrow = background.ptr<const uchar>(i);
		float *depthrow = depth.ptr<float>(i);
		uchar *dstrow = result.ptr<uchar>(i);
		for (int j=0; j<image.cols; j++) {
			// black (0,0,0) in the image denotes invalid pixels; valid black would be (0,0,1)
			if (depthrow[j] == backgroundDepth || !srcrow[3*j+1]) {
				dstrow[j] = bgrow[j];
				depthrow[j] = backgroundDepth;
			} else {
				dstrow[j] = srcrow[3*j];
			}
		}
	}
	return result;
}

// remap the given image using the given optical flow
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

// Sample color from the given channel of the image, box averaging over a circular neighborhood
// x, y: coordinates pointing directly into pixel grid, pixel coordinates are in their corners
// WARNING: returns -1 if coordinates are out of image domain
float sampleImage(const Mat image, float radiusSquared, const float x, const float y, char channel)
{
	assert (image.isContinuous() && image.depth() == CV_8U);
	char ch = image.channels();
	float sum = 0.;
	int weightSum = 0;
	//sample brightness from given neighborhood
	float radius = sqrt(radiusSquared);
	for (int ny = (int)MAX(0, y - radius); ny < MIN(y + radius + 1, image.rows); ny++) {
		const uchar *row = image.ptr<uchar>(ny);
		for (int nx = (int)MAX(0, x - radius); nx < MIN(x + radius + 1, image.cols); nx++) {
			float dx = nx - x, dy = ny - y;
			uchar val = row[nx*ch + channel];
			if (dx*dx + dy*dy <= radiusSquared && val > 0 && val < 255) {
				sum += val;
				weightSum += 1;
			}
		}
	}
	if (weightSum > 0)
		return sum / weightSum;
	else {
		//printf("Failed on %f, %f, %i\n", x, y, channel);
		return -1;
	}
}

// Sample any data type as required from the image by bilinear interpolation (used for sampling the gradient)
// x, y: coordinates pointing directly into pixel grid, pixel coordinates are in their corners
// WARNING: throws an error if coordinates are out of image domain
template <class T>
T sampleImage(const Mat image, const float x, const float y)
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

// Calculate the gradient of the given image
// returns a two-channel matrix (gx, gy)
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

// Wrapper for the OpenCV function
void saveImage(const Mat image, const char *fileName)
{
	saveImage(image, fileName, false);
}

// Wrapper for the OpenCV function
// optionally applies normalization: scales and translates the values so that they fit 0..255 range, all channels by the same mapping
void saveImage(const Mat image, const char *fileName, bool normalize)
{
	// if the suppliad image has an unsuitable number of channels, extend or remove them
	if (image.channels() > 1 && image.channels() != 3) {
		Mat bgr(image.rows, image.cols, CV_32FC3);
		int from_to[] = {-1,0, 0,1, 1,2};
		mixChannels(&image, 1, &bgr, 1, from_to, 3);
		saveImage(bgr, fileName, normalize);
		return;
	}
	
	if (normalize) {
		// calculate the minimum and maximum of all pixels of all channels
		double min, max;
		image.reshape(1);
		minMaxIdx(image, &min, &max);
		if (max == min) {
			// if the image is constant, write it unnormalized
			cv::imwrite(fileName, image);
		} else {
			// write a normalized image
			Mat normalized = (image - min) * 255 / (max - min);
			image.reshape(3);
			normalized.reshape(3);
			cv::imwrite(fileName, normalized);
		}
	} else {
		// write the image directly if normalization not requested
		cv::imwrite(fileName, image);
	}
}

// read a simple OBJ file
// supports only vertices and triangle faces, other data may cause it crash
Mesh readMesh(const char *fileName)
{
	// go through the file for the first time and count the vertices and faces
	std::ifstream is(fileName);
	std::string line;
	int vertexCount=0, faceCount=0;
	while (is.good()) {
		std::getline(is, line);
		if (line[0] == 'v')
			vertexCount += 1;
		else if (line[0] == 'f')
			faceCount += 1;
	}
	
	// read the actual mesh data
	Mesh mesh(Mat(vertexCount, 4, CV_32FC1), Mat(faceCount, 3, CV_32SC1));
	Mat vertices = mesh.vertices, faces = mesh.faces;
	// clear the EOF flag and rewind the file
	is.clear();
	is.seekg(std::ios_base::beg);
	
	int vi=0, fi=0;
	while (is.good()) {
		std::getline(is, line);
		if (line[0] == 'v') {
			float *vertex = vertices.ptr<float>(vi);
			// I am sorry for the scanf, but C++ seems to have a bit too talkative tools for this
			sscanf(line.c_str(), "v %f %f %f", &vertex[0], &vertex[1], &vertex[2]);
			vertex[3] = 1.0;
			vi ++;
		} else if (line[0] == 'f') {
			// TODO: instead of ignoring polygons, split them as a fan
			int iface[3]; // 32-bit compatibility... I hope
			sscanf(line.c_str(), "f %i %i %i", &iface[0], &iface[1], &iface[2]);
			int32_t *face = faces.ptr<int32_t>(fi);
			face[0] = iface[0]-1; face[1] = iface[1]-1; face[2] = iface[2]-1;
			fi ++;
		}
	}
	
	is.close();
	assert(vi == vertexCount); assert(fi == faceCount);
	return mesh;
}

// save the given mesh as a simple OBJ format
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
