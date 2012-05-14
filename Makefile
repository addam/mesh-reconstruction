LIBS = -lopencv_gpu -lopencv_contrib -lopencv_legacy -lopencv_objdetect -lopencv_calib3d -lopencv_features2d -lopencv_video -lopencv_highgui -lopencv_ml -lopencv_imgproc -lopencv_flann -lopencv_core
all: test_reader

test_reader: test_reader.cpp
	g++ test_reader.cpp -o test_reader ${LIBS}
