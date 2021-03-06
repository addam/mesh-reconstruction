# only option as of now
SYSTEM_OPENGL = glx
# either 'pcl' or 'cgal'
POISSON_LIBRARY = cgal
CXX = g++
CXXFLAGS = -O2
EIGEN_INCLUDE_DIR = /usr/include/eigen3
PCL_INCLUDE_DIR = /usr/local/include/pcl-1.6

opencv_LIBS = -lopencv_core -lopencv_calib3d -lopencv_video -lopencv_highgui -lopencv_imgproc -lopencv_flann -lopencv_videoio -lopencv_imgcodecs -lopencv_optflow
cgal_LIBS = -lCGAL -lboost_thread -lgmp -lmpfr
pcl_LIBS = -lpcl_common -lpcl_kdtree -lpcl_search -lpcl_surface -lpcl_features
RENDER_glx_LIBS = -lGL -lGLEW -lopencv_highgui -lX11

LIBS = ${cgal_LIBS} ${RENDER_${SYSTEM_OPENGL}_LIBS} ${opencv_LIBS} ${${POISSON_LIBRARY}_LIBS}
FILES = recon.cpp flow.cpp alpha_shapes.cpp heuristic.cpp configuration.cpp util.cpp render_${SYSTEM_OPENGL}.cpp pcl.cpp
OBJS = recon.o flow.o alpha_shapes.o heuristic.o configuration.o

all: recon

recon: Makefile recon.o alpha_shapes.o render_${SYSTEM_OPENGL}.o heuristic.o configuration.o util.o flow.o ${POISSON_LIBRARY}_poisson.o
	${CXX} ${CXXFLAGS} recon.hpp recon.o alpha_shapes.o render_${SYSTEM_OPENGL}.o heuristic.o configuration.o util.o flow.o ${POISSON_LIBRARY}_poisson.o ${LIBS} -o recon

recon.o: recon.cpp
heuristic.o: heuristic.cpp
flow.o: flow.cpp
configuration.o: configuration.cpp
util.o: util.cpp
render_glx.o: render_glx.cpp shaders.hpp

pcl_poisson.o: pcl.cpp
	${CXX} ${CXXFLAGS} -c pcl.cpp -I${PCL_INCLUDE_DIR} -I${EIGEN_INCLUDE_DIR} -Wno-deprecated-declarations -o pcl_poisson.o

cgal_poisson.o: cgal_poisson.cpp
	${CXX} ${CXXFLAGS} -c cgal_poisson.cpp -frounding-math -O2 -I${EIGEN_INCLUDE_DIR} -o cgal_poisson.o

alpha_shapes.o: alpha_shapes.cpp
	${CXX} ${CXXFLAGS} -c alpha_shapes.cpp -frounding-math -O2 -o alpha_shapes.o

shaders.hpp: pack_shaders.awk shader.vert shader.frag
	awk -f pack_shaders.awk shader.vert shader.frag > shaders.hpp

test: recon
	rm frame*.png || true
	./recon tracks/koberec-.yaml -v

test_alpha_shapes: alpha_shapes.cpp
	${CXX} ${CXXFLAGS} alpha_shapes.cpp -frounding-math -O2 ${cgal_LIBS} -lopencv_core -DTEST_BUILD -o test_alpha_shapes
	/usr/bin/time -f '%e seconds, %M kBytes' ./test_alpha_shapes

test_cgal_poisson: cgal_poisson.cpp
	${CXX} ${CXXFLAGS} cgal_poisson.cpp -frounding-math -O2 ${cgal_LIBS} -lopencv_core -I${EIGEN_INCLUDE_DIR} -DTEST_BUILD -o test_cgal_poisson
	/usr/bin/time -f '%e seconds, %M kBytes' ./test_cgal_poisson

test_pcl: pcl.cpp
	${CXX} ${CXXFLAGS} pcl.cpp -O2 -I${PCL_INCLUDE_DIR} -I${EIGEN_INCLUDE_DIR} -Wno-deprecated-declarations ${pcl_LIBS} -lpcl_io -lpcl_features -lopencv_core -DTEST_BUILD -o test_pcl
	/usr/bin/time -f '%e seconds, %M kBytes' ./test_pcl

test_flow: flow.cpp
	${CXX} ${CXXFLAGS} flow.cpp -DTEST_BUILD -g ${opencv_LIBS} -o test_flow

test_glx: render_glx.cpp shaders.hpp
	${CXX} ${CXXFLAGS} render_glx.cpp ${RENDER_glx_LIBS} -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_imgcodecs -DTEST_BUILD -o glx
	./glx

clean:
	rm recon *.o shaders.hpp

clean_images:
	rm depth-frame*.png || true
	rm flow-frame*from*.png || true
	rm frame*.png || true
	rm project-frame*from*.png || true
