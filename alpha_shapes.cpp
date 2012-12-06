#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Delaunay_triangulation_3.h>
#include <CGAL/Alpha_shape_3.h>
#include <CGAL/Polyhedron_3.h>

#include <fstream>
#include <list>
#include <map>

typedef CGAL::Exact_predicates_inexact_constructions_kernel Gt;

typedef CGAL::Alpha_shape_vertex_base_3<Gt>          Vb;
typedef CGAL::Alpha_shape_cell_base_3<Gt>            Fb;
typedef CGAL::Triangulation_data_structure_3<Vb,Fb>  Tds;
typedef CGAL::Delaunay_triangulation_3<Gt,Tds>       Triangulation_3;
typedef CGAL::Alpha_shape_3<Triangulation_3>      Alpha_shape_3;

typedef CGAL::Polyhedron_3<Gt> Polyhedron_3;
typedef Alpha_shape_3::Facet  Facet;
typedef Alpha_shape_3::Cell_handle Cell_handle;
typedef Alpha_shape_3::Cell Cell;
typedef Gt::Point_3                                  Point;
typedef Alpha_shape_3::Alpha_iterator             Alpha_iterator;

int main()
{
  std::list<Point> lp;
  std::map<Point, int> vertex_indices;

  //read input
  std::ifstream is("./bunny_1000");
  std::ofstream os("./bunny_alpha");
  int n;
  is >> n;
  std::cout << "Reading " << n << " points " << std::endl;
  Point p;
  for(int i=1; i <= n; i ++) {
    is >> p;
    os << "v " << p << std::endl;
    vertex_indices[p] = i;
    lp.push_back(p);
  }

  // compute alpha shape
  Alpha_shape_3 as(lp.begin(),lp.end());
  std::cout << "Alpha shape computed in REGULARIZED mode by default" << std::endl;

  // find optimal alpha value
  Alpha_iterator opt = as.find_optimal_alpha(1);
  std::cout << "Optimal alpha value to get one connected component is " << *opt << std::endl;
  as.set_alpha(*opt);
  assert(as.number_of_solid_components() == 1);
  
  std::list<Facet> facets;
  as.get_alpha_shape_facets(back_inserter(facets), Alpha_shape_3::REGULAR);
  as.get_alpha_shape_facets(back_inserter(facets), Alpha_shape_3::SINGULAR);
  for (std::list<Facet>::iterator it = facets.begin(); it != facets.end(); it ++){
		Facet facet = *it;
		Cell_handle cellh = facet.first;
		Cell cell = *cellh;
		int vertex_excluded = facet.second;
		os << "f ";
		for (char i = 0; i < 4; i++){
			if (i != vertex_excluded)
				os << vertex_indices[cell.vertex(i)->point()] << " ";
		}
		os << std::endl;
	}
	os.close();
  return 0;
}
