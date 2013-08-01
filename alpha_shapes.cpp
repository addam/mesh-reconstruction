// alpha_shapes.cpp: wrapper for CGAL alpha shape calculation

#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Delaunay_triangulation_3.h>
#include <CGAL/Alpha_shape_3.h>

#include <vector>
#include <map>

#ifdef TEST_BUILD
	#include <iostream>
	#include <fstream>
	#include <opencv2/core/core.hpp>
	typedef cv::Mat Mat;
#else
	#include "recon.hpp"
#endif

typedef CGAL::Exact_predicates_inexact_constructions_kernel Gt;

typedef CGAL::Alpha_shape_vertex_base_3<Gt>          Vb;
typedef CGAL::Alpha_shape_cell_base_3<Gt>            Fb;
typedef CGAL::Triangulation_data_structure_3<Vb,Fb>  Tds;
typedef CGAL::Delaunay_triangulation_3<Gt,Tds>       Triangulation_3;
typedef CGAL::Alpha_shape_3<Triangulation_3>      Alpha_shape_3;

typedef Alpha_shape_3::Facet Facet;
typedef Alpha_shape_3::Cell_handle Cell_handle;
typedef Alpha_shape_3::Cell Cell;
typedef Gt::Point_3 Point;
typedef Alpha_shape_3::Alpha_iterator Alpha_iterator;

// return vertex indices forming all the faces of the alpha shape
// alpha is an output parameter, its value is calculated to make the alpha shape a single component
Mat alphaShapeFaces(Mat points, float *alpha)
{
	if (points.rows == 0 || points.cols == 0)
		return Mat(0, 3, CV_32SC1);
	
	// convert points to Cartesian if necessary, and to a format suitable for CGAL
	std::vector<Point> lp;
	std::map<Point, int> vertex_indices;
	lp.reserve(points.rows);
	if (points.cols == 3) {
		for (int i = 0; i < points.rows; i++) {
			const float* cvPoint = points.ptr<float>(i);
			Point p(cvPoint[0], cvPoint[1], cvPoint[2]);
			vertex_indices[p] = i;
			lp.push_back(p);
		}
	}	else if (points.cols == 4) {
		for (int i = 0; i < points.rows; i++) {
			const float* cvPoint = points.ptr<float>(i);
			Point p(cvPoint[0]/cvPoint[3], cvPoint[1]/cvPoint[3], cvPoint[2]/cvPoint[3]);
			vertex_indices[p] = i;
			lp.push_back(p);
		}
	} else {
		assert(false);
	}

	// Calculate the alpha shape from the given points
  Alpha_shape_3 as(lp.begin(),lp.end());

	// Choose an optimal value of alpha
  Alpha_iterator opt = as.find_optimal_alpha(1);
  #ifdef TEST_BUILD
  if (*alpha > 0)
	  as.set_alpha(*alpha);
	else
		as.set_alpha(*opt);
	#else
  as.set_alpha(*opt);
  assert(as.number_of_solid_components() == 1);
	#endif
  if (alpha != NULL)
	  *alpha = *opt;

	// Get all faces of the alpha shape into an OpenCV matrix
  std::vector<Facet> facets;
  as.get_alpha_shape_facets(back_inserter(facets), Alpha_shape_3::REGULAR);
  #ifdef TEST_BUILD
  as.get_alpha_shape_facets(back_inserter(facets), Alpha_shape_3::SINGULAR);
  #endif
  Mat result(facets.size(), 3, CV_32SC1);
  for (int i=0; i < facets.size(); i++) {
		Cell cell = *(facets[i].first);
		int vertex_excluded = facets[i].second;
		int32_t* outfacet = result.ptr<int32_t>(i);
		// a little magic so that face normals are oriented outside
		char sign = (vertex_excluded % 2 == (0 == cell.get_alpha() || cell.get_alpha() > *opt)) ? 1 : 2; // 2 = -1 (mod 3)
		for (char j = vertex_excluded + 1; j < vertex_excluded + 4; j++){
			outfacet[(sign*j)%3] = vertex_indices[cell.vertex(j%4)->point()];
		}
	}
	
	return result;
}

Mat alphaShapeFaces(Mat points)
{
	return alphaShapeFaces(points, NULL);
}

#ifdef TEST_BUILD
int main(int argc, char**argv)
{
	//read input
	std::ifstream is("test/bunny_5000");
	std::ofstream os("test/bunny_alpha.obj");
	float alpha = 0;
	if (argc>1)
		alpha = atof(argv[1]);
	int n;
	is >> n;
	std::cout << "Reading " << n << " points..." << std::endl;
	Mat points(0, 1, CV_32FC3);
	cv::Point3f point;
	for(int i=0; i < n; i ++) {
		// split the bunny
		/*
		is >> point.x >> point.y >> point.z;
		if (point.x < -0.02)
			point.x -= 0.12;
		*/
			
		points.push_back(point);
		const float* row = points.ptr<float>();
		os << "v " << point.x << ' ' << point.y << ' ' << point.z << std::endl;
	}
	is.close();
	points = points.reshape(1);
	std::cout << points.rows << " points, " << points.cols << " dimensions" << std::endl;
	std::cout << "Calculating alpha shape..." << std::endl;
	Mat alphaShape = alphaShapeFaces(points, &alpha);
	for (int i=0; i < alphaShape.rows; i++){
		const int32_t* row = alphaShape.ptr<int32_t>(i);
		os << "f " << row[0]+1 << ' ' << row[1]+1 << ' ' << row[2]+1 << std::endl;
	}
	os.close();
}
#endif
