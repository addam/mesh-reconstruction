SYSTEM_OPENGL = glx

OPENCV_LIBS = -lopencv_core -lopencv_calib3d -lopencv_video -lopencv_highgui -lopencv_imgproc
ALPHA_SHAPES_LIBS = -lCGAL -lboost_thread -lgmp
RENDER_glx_LIBS = -lGL -lGLEW -lopencv_highgui

LIBS = ${ALPHA_SHAPES_LIBS} ${RENDER_${SYSTEM_OPENGL}_LIBS} ${OPENCV_LIBS}
FILES = recon.cpp flow.cpp alpha_shapes.cpp heuristic.cpp configuration.cpp util.cpp render_${SYSTEM_OPENGL}.o
OBJS = recon.o flow.o alpha_shapes.o heuristic.o configuration.o

all: recon

recon: recon.o alpha_shapes.o render_${SYSTEM_OPENGL}.o heuristic.o configuration.o util.o flow.o
	g++ recon.hpp recon.o alpha_shapes.o render_${SYSTEM_OPENGL}.o heuristic.o configuration.o util.o flow.o ${LIBS} -o recon

recon.o: recon.cpp
heuristic.o: heuristic.cpp
flow.o: flow.cpp
configuration.o: configuration.cpp
util.o: util.cpp
render_glx.o: render_glx.cpp shaders.hpp

alpha_shapes.o: alpha_shapes.cpp
	g++ -c alpha_shapes.cpp -frounding-math -O2 -o alpha_shapes.o

shaders.hpp: pack_shaders.awk shader.vert shader.frag
	awk -f pack_shaders.awk shader.vert shader.frag > shaders.hpp

test_reader: test_reader.cpp
	g++ test_reader.cpp -g ${LIBS} -o test_reader

test: test_reader
	./test_reader

test_alpha_shapes: alpha_shapes.cpp
	g++ alpha_shapes.cpp -frounding-math -O2 ${ALPHA_SHAPES_LIBS} -lopencv_core -DTEST_BUILD -o test_alpha_shapes
	./test_alpha_shapes

test_glx: render_glx.cpp
	g++ render_glx.cpp ${RENDER_glx_LIBS} -lopencv_core -lopencv_highgui -DTEST_BUILD -o glx
	./glx

sandbox: sandbox.cpp
	g++ sandbox.cpp -g -lopencv_core -lopencv_gpu -lopencv_contrib -lopencv_legacy -lopencv_objdetect -lopencv_calib3d -lopencv_features2d -lopencv_video -lopencv_highgui -lopencv_ml -lopencv_imgproc -lopencv_flann -o sandbox

clean:
	rm recon *.o shaders.hpp
