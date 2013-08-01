// cgal_poisson.cpp wrapper for the Poisson reconstruction via CGAL

#define CGAL_EIGEN3_ENABLED

#include <CGAL/trace.h>
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Polyhedron_3.h>
#include <CGAL/IO/Polyhedron_iostream.h>
#include <CGAL/Surface_mesh_default_triangulation_3.h>
#include <CGAL/make_surface_mesh.h>
#include <CGAL/Implicit_surface_3.h>
#include <CGAL/IO/output_surface_facets_to_polyhedron.h>
#include <CGAL/Poisson_reconstruction_function.h>
#include <CGAL/Point_with_normal_3.h>
#include <CGAL/property_map.h>
#include <CGAL/IO/read_xyz_points.h>
#include <CGAL/compute_average_spacing.h>

#include <vector>
#include <map>

#ifdef TEST_BUILD
	#include <fstream>
	#include <iostream>
	#include <opencv2/core/core.hpp>
	typedef cv::Mat Mat;
	typedef struct Mesh{Mat vertices, faces; Mesh(Mat v, Mat f):vertices(v), faces(f) {};} Mesh;
#else
	#include "recon.hpp"
#endif

// Types
typedef CGAL::Exact_predicates_inexact_constructions_kernel Kernel;
typedef Kernel::FT FT;
typedef Kernel::Point_3 Point;
typedef Kernel::Vector_3 Vector;
typedef CGAL::Point_with_normal_3<Kernel> Point_with_normal;
typedef Kernel::Sphere_3 Sphere;
typedef std::vector<Point_with_normal> PointList;
typedef CGAL::Poisson_reconstruction_function<Kernel> Poisson_reconstruction_function;
typedef CGAL::Surface_mesh_default_triangulation_3 STr;
typedef STr::Cell Cell;
typedef CGAL::Surface_mesh_complex_2_in_triangulation_3<STr> C2t3;
typedef CGAL::Implicit_surface_3<Kernel, Poisson_reconstruction_function> Surface_3;

// adapted from http://www.cgal.org/Manual/beta/examples/Surface_reconstruction_points_3/poisson_reconstruction_example.cpp
Mesh poissonSurface(const Mat ipoints, const Mat normals)
{
		// Poisson options
		FT sm_angle = 20.0; // Min triangle angle in degrees.
		FT sm_radius = 300; // Max triangle size w.r.t. point set average spacing.
		FT sm_distance = 0.375; // Surface Approximation error w.r.t. point set average spacing.

		PointList points;
		points.reserve(ipoints.rows);
		float min = 0, max = 0;
		for (int i=0; i<ipoints.rows; i++) {
			float const *p = ipoints.ptr<float const>(i), *n = normals.ptr<float const>(i);
			points.push_back(Point_with_normal(Point(p[0]/p[3], p[1]/p[3], p[2]/p[3]), Vector(n[0], n[1], n[2])));
			if (p[0]/p[3] > max) max = p[0]/p[3];
			if (p[0]/p[3] < min) min = p[0]/p[3];
		}

		// Creates implicit function from the read points using the default solver.

		// Note: this method requires an iterator over points
		// + property maps to access each point's position and normal.
		// The position property map can be omitted here as we use iterators over Point_3 elements.
		Poisson_reconstruction_function function(points.begin(), points.end(), CGAL::make_normal_of_point_with_normal_pmap(points.begin()) );

		// Computes the Poisson indicator function f() at each vertex of the triangulation.
		bool success = function.compute_implicit_function();
		assert(success);
#ifdef TEST_BUILD
		printf("implicit function ready. Meshing...\n");
#endif
		// Computes average spacing
		FT average_spacing = CGAL::compute_average_spacing(points.begin(), points.end(), 6 /* knn = 1 ring */);

		// Gets one point inside the implicit surface
		// and computes implicit function bounding sphere radius.
		Point inner_point = function.get_inner_point();
		Sphere bsphere = function.bounding_sphere();
		FT radius = std::sqrt(bsphere.squared_radius());

		// Defines the implicit surface: requires defining a
		// conservative bounding sphere centered at inner point.
		FT sm_sphere_radius = 5.0 * radius;
		FT sm_dichotomy_error = sm_distance*average_spacing/1000.0; // Dichotomy error must be << sm_distance
		Surface_3 surface(function, Sphere(inner_point,sm_sphere_radius*sm_sphere_radius), sm_dichotomy_error/sm_sphere_radius);

		// Defines surface mesh generation criteria
		// parameters: min triangle angle (degrees), max triangle size, approximation error
		CGAL::Surface_mesh_default_criteria_3<STr> criteria(sm_angle, sm_radius*average_spacing, sm_distance*average_spacing);

		// Generates surface mesh with manifold option
		STr tr; // 3D Delaunay triangulation for surface mesh generation
		C2t3 c2t3(tr); // 2D complex in 3D Delaunay triangulation
		// parameters: reconstructed mesh, implicit surface, meshing criteria, 'do not require manifold mesh'
		CGAL::make_surface_mesh(c2t3, surface, criteria, CGAL::Non_manifold_tag());

		assert(tr.number_of_vertices() > 0);
		
		// Search structure: index of the vertex at an exactly given position
		std::map<Point, int> vertexIndices;
		
		// Extract all vertices of the resulting mesh
		Mat vertices(tr.number_of_vertices(), 4, CV_32FC1), faces(c2t3.number_of_facets(), 3, CV_32SC1);
#ifdef TEST_BUILD
		printf("%i vertices, %i facets. Converting...\n", vertices.rows, faces.rows);
#endif
		{int i=0;
			for (C2t3::Vertex_iterator it=c2t3.vertices_begin(); it!=c2t3.vertices_end(); it++, i++) {
				float *vertex = vertices.ptr<float>(i);
				Point p = it->point();
				vertex[0] = p.x(); vertex[1] = p.y(); vertex[2] = p.z();
				vertex[3] = 1;
				vertexIndices[p] = i;
			}
			vertices.resize(i);
		}
		
		// Extract all faces of the resulting mesh and calculate the index of each of their vertices
		{int i=0; for (C2t3::Facet_iterator it=c2t3.facets_begin(); it!=c2t3.facets_end(); it++, i++) {
			Cell cell = *it->first;
			int vertex_excluded = it->second;
			int32_t* outfacet = faces.ptr<int32_t>(i);
			// a little magic so that face normals are oriented outside
			char sign = (function(cell.vertex(vertex_excluded)->point()) > 0) ? 1 : 2; // 2 = -1 (mod 3)
			for (char j = vertex_excluded + 1; j < vertex_excluded + 4; j++){
				outfacet[(sign*j)%3] = vertexIndices[cell.vertex(j%4)->point()];
			}
		}}

		return Mesh(vertices, faces);
}

#ifdef TEST_BUILD
int main()
{
	//read input
	std::ifstream is("test/suzanne");
	std::ofstream os("test/suzanne_poisson.obj");
	int n;
	is >> n;
	std::cout << "Reading " << n << " points with normals..." << std::endl;
	Mat points(n, 4, CV_32FC1), normals(n, 3, CV_32FC1);
	for(int i=0; i < n; i ++) {
		float *point = points.ptr<float>(i), *normal = normals.ptr<float>(i);
		is >> point[0] >> point[1] >> point[2] >> normal[0] >> normal[1] >> normal[2];
		point[3] = 1;
	}
	is.close();
	std::cout << points.rows << " points, " << points.cols << " dimensions" << std::endl;
	std::cout << "Running Poisson reconstruction..." << std::endl;
	Mesh result = poissonSurface(points, normals);
	for (int i=0; i < result.vertices.rows; i++){
		float* row = result.vertices.ptr<float>(i);
		os << "v " << row[0] << ' ' << row[1] << ' ' << row[2] << std::endl;
	}
	for (int i=0; i < result.faces.rows; i++){
		const int32_t* row = result.faces.ptr<int32_t>(i);
		os << "f " << row[0]+1 << ' ' << row[1]+1 << ' ' << row[2]+1 << std::endl;
	}
	os.close();
}
#endif
