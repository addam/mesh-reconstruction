#include <pcl/point_types.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/features/normal_3d.h>
#include <pcl/surface/gp3.h>
#include <pcl/surface/poisson.h>
#include <pcl/surface/marching_cubes_rbf.h>
#include <pcl/io/obj_io.h>
#include <pcl/common/io.h>

#ifdef TEST_BUILD
	#include <iostream>
	#include <fstream>
	#include <utility>
	#include <opencv2/core/core.hpp>
	typedef cv::Mat Mat;
	typedef struct Mesh{Mat vertices, faces; Mesh(Mat v, Mat m):vertices(v), faces(m) {};} Mesh;
#else
	#include "recon.hpp"
#endif
typedef pcl::PointCloud<pcl::PointNormal> NormalCloud;

NormalCloud::Ptr convert(const Mat points, const Mat normals)
{
	NormalCloud::Ptr cloud(new NormalCloud);
	assert(points.cols == normals.cols);
	cloud->reserve(points.cols);
	for (int i=0; i<points.cols; i++) {
		pcl::PointNormal p;
		for (char j=0; j<3; j++) {
			p.data[j] = points.at<float>(j,i) / points.at<float>(3,i);
			p.normal[j] = normals.at<float>(j,i);
		}
		cloud->push_back(p);
	}
	return cloud;
}

void convert(Mesh dst, std::vector<pcl::Vertices> faces)
{
	for (int i=0; i<faces.size(); i++) {
		assert(faces[i].vertices.size() == 3);
		for (char j=0; j<3; j++) {
			dst.faces.at<int32_t>(i, j) = faces[i].vertices[j];
		}
	}
}

int convert(Mesh dst, const pcl::PolygonMesh &mesh)
{
  if (mesh.cloud.data.empty())
    return -1;
  
  int nr_points  = mesh.cloud.width * mesh.cloud.height;
  unsigned point_size = static_cast<unsigned> (mesh.cloud.data.size() / nr_points);
  unsigned nr_faces = static_cast<unsigned> (mesh.polygons.size());
  
  size_t field_map[3];
	for (size_t d = 0; d < mesh.cloud.fields.size(); d++) {
		if (mesh.cloud.fields[d].datatype != sensor_msgs::PointField::FLOAT32)
			continue;
		if (mesh.cloud.fields[d].name == "x") {
			field_map[0] = mesh.cloud.fields[d].offset;
		} else if (mesh.cloud.fields[d].name == "y") {
			field_map[1] = mesh.cloud.fields[d].offset;
		} else if (mesh.cloud.fields[d].name == "z") {
			field_map[2] = mesh.cloud.fields[d].offset;
		}
	}
	
	for (char j=0; j<3; j++) {
		size_t d = field_map[j];
		float *coords = dst.vertices.ptr<float>(j);
	  for (int i = 0; i < nr_points; ++i) {
			float* src = (float*) (&mesh.cloud.data[i*point_size + d]);
			coords[i] = *src;
    }
	}

  /*if(normal_index != -1)
  {    
    fs << "# Normals in (x,y,z) form; normals might not be unit." <<  std::endl;
    // Write vertex normals
    for (int i = 0; i < nr_points; ++i) {
      int nxyz = 0;
      for (size_t d = 0; d < mesh.cloud.fields.size (); ++d) {
        int c = 0;
        // adding vertex
        if ((mesh.cloud.fields[d].datatype == sensor_msgs::PointField::FLOAT32) && (
              mesh.cloud.fields[d].name == "normal_x" ||
              mesh.cloud.fields[d].name == "normal_y" ||
              mesh.cloud.fields[d].name == "normal_z")) {
          if (mesh.cloud.fields[d].name == "normal_x")
            fs << "vn ";
          
          float value;
          memcpy (&value, &mesh.cloud.data[i * point_size + mesh.cloud.fields[d].offset + c * sizeof (float)], sizeof (float));
          fs << value;
          if (++nxyz == 3)
            break;
          fs << " ";
        }
      }
      if (nxyz != 3)
      {
        PCL_ERROR ("[pcl::io::saveOBJFile] Input point cloud has no normals!\n");
        return (-2);
      }
      fs << std::endl;
    }
  }*/

	for (int i=0; i<nr_faces; i++) {
		assert(mesh.polygons[i].vertices.size() == 3);
		int32_t* verts = dst.faces.ptr<int32_t>(i);
		for (char j=0; j<3; j++) {
			verts[j] = mesh.polygons[i].vertices[j];
		}
	}
	
	return 0;
}

Mesh poissonSurface(const Mat points, const Mat normals)
{
	NormalCloud::Ptr cloud(convert(points, normals));
	
	pcl::Poisson<pcl::PointNormal> poisson;
	poisson.setDegree(3);
	poisson.setOutputPolygons(false);
	poisson.setInputCloud (cloud);
	
	pcl::PolygonMesh mesh;
 	poisson.reconstruct(mesh);

	Mesh result(Mat(4, mesh.cloud.width * mesh.cloud.height, CV_32FC1), Mat(mesh.polygons.size(), 3, CV_32SC1));
	//pcl::io::saveOBJFile("temp_poisson.obj", mesh);
	convert(result, mesh);
	return result;
}

Mesh rbfSurface(const Mat points, const Mat normals)
{
	NormalCloud::Ptr cloud(convert(points, normals));
	
	pcl::MarchingCubesRBF<pcl::PointNormal> mc;
	mc.setInputCloud (cloud);
	
	pcl::PolygonMesh mesh;
 	mc.reconstruct(mesh);
	
	Mesh result(Mat(4, mesh.cloud.width * mesh.cloud.height, CV_32FC1), Mat(mesh.polygons.size(), 3, CV_32SC1));
	convert(result, mesh);
	return result;
}

Mesh greedyProjection(const Mat points, const Mat normals)
{
	NormalCloud::Ptr cloud (convert(points, normals));
	
	pcl::search::KdTree<pcl::PointNormal>::Ptr tree2 (new pcl::search::KdTree<pcl::PointNormal>);
	tree2->setInputCloud (cloud);
	
	pcl::GreedyProjectionTriangulation<pcl::PointNormal> gp3;
	pcl::PolygonMesh triangles;
	
	// Set the maximum distance between connected points (maximum edge length)
	gp3.setSearchRadius (0.025);
	// Set typical values for the parameters
	gp3.setMu (2.5);
	gp3.setMaximumNearestNeighbors (100);
	gp3.setMaximumSurfaceAngle(M_PI/4); // 45 degrees
	gp3.setMinimumAngle(M_PI/18); // 10 degrees
	gp3.setMaximumAngle(2*M_PI/3); // 120 degrees
	gp3.setNormalConsistency(false);
	
	// Get result
	gp3.setInputCloud (cloud);
	gp3.setSearchMethod (tree2);
	gp3.reconstruct (triangles);
	
	// Additional vertex information
	//std::vector<int> parts = gp3.getPartIDs();
	//std::vector<int> states = gp3.getPointStates();
	
	Mesh result(points, Mat(triangles.polygons.size(), 3, CV_32SC1));
	convert(result, triangles.polygons);
	
	return result;
}

#ifdef TEST_BUILD
Mat estimatedNormals(Mat points)
{
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZ>);

	cloud->reserve(points.rows);
	for (int i=0; i<points.rows; i++) {
		pcl::PointXYZ p;
		for (char j=0; j<3; j++) {
			p.data[j] = points.at<float>(i,j);
		}
		cloud->push_back(p);
	}
	
  pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> n;
  pcl::PointCloud<pcl::Normal>::Ptr normals (new pcl::PointCloud<pcl::Normal>);
  pcl::search::KdTree<pcl::PointXYZ>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZ>);
  tree->setInputCloud (cloud);
  n.setInputCloud (cloud);
  n.setSearchMethod (tree);
  n.setKSearch (20);
  n.compute (*normals);
  
  Mat result(3, normals->width, CV_32FC1);
	for (int i=0; i<normals->width; i++) {
		for (char j=0; j<3; j++) {
			result.at<float>(j, i) = (*normals)(i,0).normal[j]; // slow as hell
		}
	}
  return result;
}
int main()
{
	//read input
	std::ifstream is("shit/bunny_1000");
	//std::ifstream is("shit/stanford_dragon_big");
	std::ofstream os("shit/bunny_greedyprojection.obj");
  os.precision (5);
	int n;
	is >> n;
	std::cout << "Reading " << n << " points..." << std::endl;
	Mat points(0, 1, CV_32FC3);
	cv::Point3f point;
	for(int i=0; i < n; i ++) {
		is >> point.x >> point.y >> point.z;
		points.push_back(point);
	}
	is.close();
	points = points.reshape(1);
	std::cout << points.rows << " points, " << points.cols << " dimensions" << std::endl;
	std::cout << "Calculating normals..." << std::endl;
	Mat normals = estimatedNormals(points);
	std::cout << "Calculating surface..." << std::endl;
	points = points.t();
	Mat lastrow (Mat::ones(1, points.cols, CV_32FC1));
	points.push_back(lastrow); //homogenize
	Mesh result = poissonSurface(points, normals);//greedyProjection(points, normals);
	std::cout << result.vertices.cols << " vertices, " << result.faces.rows << " faces" << std::endl;
	for(int i=0; i < result.vertices.cols; i ++) {
		os << "v " << result.vertices.at<float>(0,i) << ' ' << result.vertices.at<float>(1,i) << ' ' << result.vertices.at<float>(2,i) << std::endl;
	}
	for (int i=0; i < result.faces.rows; i++) {
		const int32_t* row = result.faces.ptr<int32_t>(i);
		os << "f " << row[0]+1 << ' ' << row[1]+1 << ' ' << row[2]+1 << std::endl;
	}
	os.close();
	return EXIT_SUCCESS;
}
#endif
