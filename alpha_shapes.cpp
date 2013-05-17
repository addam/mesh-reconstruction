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

Mat alphaShapeIndices(Mat points, float *alpha)
// alpha is only a side output, its original value is not used
{
	if (points.rows == 0 || points.cols == 0)
		return Mat(0, 3, CV_32SC1);
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

  Alpha_shape_3 as(lp.begin(),lp.end());

  Alpha_iterator opt = as.find_optimal_alpha(1);
  as.set_alpha(*opt);
  assert(as.number_of_solid_components() == 1);
  if (alpha != NULL)
	  *alpha = *opt;

  std::vector<Facet> facets;
  as.get_alpha_shape_facets(back_inserter(facets), Alpha_shape_3::REGULAR);
  //as.get_alpha_shape_facets(back_inserter(facets), Alpha_shape_3::SINGULAR);
  Mat result(facets.size(), 3, CV_32SC1);
  for (int i=0; i < facets.size(); i++) {
		Cell cell = *(facets[i].first);
		int vertex_excluded = facets[i].second;
		// TODO: is it possible to have the normal right away correct?
		int32_t* outfacet = result.ptr<int32_t>(i);
		for (char j = vertex_excluded + 1; j < vertex_excluded + 4; j++){
			outfacet[j%3] = vertex_indices[cell.vertex(j%4)->point()];
		}
	}
	return result;
}

Mat alphaShapeIndices(Mat points)
{
	return alphaShapeIndices(points, NULL);
}

#ifdef TEST_BUILD
int main()
{
	//read input
	//std::ifstream is("shit/bunny_1000");
	std::ifstream is("shit/stanford_dragon_big");
	std::ofstream os("shit/dragon_alpha.obj");
	int n;
	is >> n;
	std::cout << "Reading " << n << " points..." << std::endl;
	Mat points(0, 1, CV_32FC3);
	cv::Point3f point;
	for(int i=0; i < n; i ++) {
		is >> point.x >> point.y >> point.z;
		points.push_back(point);
		const float* row = points.ptr<float>(i);
		os << "v " << row[0] << ' ' << row[1] << ' ' << row[2] << std::endl;
	}
	is.close();
	points = points.reshape(1);
	std::cout << points.rows << " points, " << points.cols << " dimensions" << std::endl;
	std::cout << "Calculating alpha shape..." << std::endl;
	Mat alphaShape = alphaShapeIndices(points);
	for (int i=0; i < alphaShape.rows; i++){
		const int32_t* row = alphaShape.ptr<int32_t>(i);
		os << "f " << row[0]+1 << ' ' << row[1]+1 << ' ' << row[2]+1 << std::endl;
	}
	os.close();
}
#endif
