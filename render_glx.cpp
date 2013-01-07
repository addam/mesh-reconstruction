#include <X11/Xlib.h>
#include <X11/Xutil.h>
#include <GL/glew.h>
#include <GL/glx.h>

//sets the variables vertexShaderSources, fragmentShaderSources
#include "shaders.hpp"

#ifdef TEST_BUILD
	#include <stdio.h>
	#include <stdlib.h>
	#include <string>
	#include <iostream>
	#include <fstream>
	
	#include <opencv2/core/core.hpp>
	#include <opencv2/highgui/highgui.hpp>
	typedef cv::Mat Mat;
	class Render {};
	class Heuristic {};
#else
	#include "recon.hpp"
#endif

#define GLX_CONTEXT_MAJOR_VERSION_ARB		0x2091
#define GLX_CONTEXT_MINOR_VERSION_ARB		0x2092
typedef GLXContext (*GLXCREATECONTEXTATTRIBSARBPROC)(Display*, GLXFBConfig, GLXContext, Bool, const int*);

class RenderGLX;

class RenderGLX: public Render {
	public:
		RenderGLX(int width, int height, char *displayName);
		~RenderGLX();
		virtual void loadMesh(const Mat points, const Mat indices);
		virtual Mat projected(const Mat camera, const Mat frame, const Mat projector);
		virtual Mat depth(const Mat camera);		
	protected:
		GLuint programID, mainMatrixID, sideMatrixID, textureSamplerID, vertexbuffer, vertexArrayID, imgw, imgh;
		Display *display;
		GLXContext context;
		GLXPbuffer glxbuffer;
		int vertex_count;
};

Render *spawnRender(Heuristic hint)
{
	RenderGLX *render = new RenderGLX(640, 480, getenv("DISPLAY"));
	return render;
}

GLuint createTexture(const Mat image){
	Mat flipped(image.rows, image.cols, image.channels());
	cv::flip(image, flipped, 0);
	GLuint texture;
	glGenTextures(1, &texture);
	glBindTexture(GL_TEXTURE_2D, texture);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, flipped.cols, flipped.rows, 0, GL_RGB, GL_UNSIGNED_BYTE, flipped.data);
	
	// nearest neighbor filtering
	//glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	//glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST); 

	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR); 
	glGenerateMipmap(GL_TEXTURE_2D);

	return texture;
}

GLuint LoadShaders(){
	// Create the shaders
	GLuint VertexShaderID = glCreateShader(GL_VERTEX_SHADER);
	GLuint FragmentShaderID = glCreateShader(GL_FRAGMENT_SHADER);

	GLint Result = GL_FALSE;
	int InfoLogLength;

	// Compile Vertex Shader
	glShaderSource(VertexShaderID, 1, vertexShaderSources, NULL);
	glCompileShader(VertexShaderID);

	#ifdef TEST_BUILD
	// Check Vertex Shader
	glGetShaderiv(VertexShaderID, GL_COMPILE_STATUS, &Result);
	glGetShaderiv(VertexShaderID, GL_INFO_LOG_LENGTH, &InfoLogLength);
	if (InfoLogLength > 1){
		std::string VertexShaderErrorMessage(InfoLogLength+1, 0);
		glGetShaderInfoLog(VertexShaderID, InfoLogLength, NULL, &VertexShaderErrorMessage[0]);
		std::cerr << VertexShaderErrorMessage;
	}
	#endif

	// Compile Fragment Shader
	glShaderSource(FragmentShaderID, 1, fragmentShaderSources, NULL);
	glCompileShader(FragmentShaderID);

	#ifdef TEST_BUILD
	// Check Fragment Shader
	glGetShaderiv(FragmentShaderID, GL_COMPILE_STATUS, &Result);
	glGetShaderiv(FragmentShaderID, GL_INFO_LOG_LENGTH, &InfoLogLength);
	if (InfoLogLength > 1){
		std::string FragmentShaderErrorMessage(InfoLogLength+1, 0);
		glGetShaderInfoLog(FragmentShaderID, InfoLogLength, NULL, &FragmentShaderErrorMessage[0]);
		std::cerr << FragmentShaderErrorMessage;
	}
	#endif

	// Link the program
	GLuint ProgramID = glCreateProgram();
	glAttachShader(ProgramID, VertexShaderID);
	glAttachShader(ProgramID, FragmentShaderID);
	glLinkProgram(ProgramID);

	#ifdef TEST_BUILD
	// Check the program
	glGetProgramiv(ProgramID, GL_LINK_STATUS, &Result);
	glGetProgramiv(ProgramID, GL_INFO_LOG_LENGTH, &InfoLogLength);
	if (InfoLogLength > 1){
		std::string ProgramErrorMessage(InfoLogLength+1, 0);
		glGetProgramInfoLog(ProgramID, InfoLogLength, NULL, &ProgramErrorMessage[0]);
		std::cerr << ProgramErrorMessage;
	}
	#endif

	glDeleteShader(VertexShaderID);
	glDeleteShader(FragmentShaderID);

	return ProgramID;
}

RenderGLX::RenderGLX(int width, int height, char *displayName)
{
	vertexbuffer = -1;
	vertex_count = 0;
	
	imgw = width;
	imgh = height;
	_Xdebug = 1;
	display = XOpenDisplay(displayName);
	XSynchronize(display, 0);

	//NOTE: for some reason, my system has only a RGBA Visual and only a RGB FrameBuffer
	int visualattributes[] = {GLX_RGBA, GLX_RED_SIZE, 8, GLX_GREEN_SIZE, 8, GLX_BLUE_SIZE, 8, GLX_DEPTH_SIZE, 1, None};
	XVisualInfo *visual = glXChooseVisual(display, XDefaultScreen(display), visualattributes);
	
	int dummy;
	int fbattributes[] = {GLX_RED_SIZE, 8, GLX_GREEN_SIZE, 8, GLX_BLUE_SIZE, 8, GLX_DEPTH_SIZE, 1, None};
	GLXFBConfig *fbconfig = glXChooseFBConfig(display, visual->screen, fbattributes, &dummy);

	int pbattributes[] = {GLX_PBUFFER_WIDTH, imgw, GLX_PBUFFER_HEIGHT, imgh, None};
	glxbuffer = glXCreatePbuffer(display, fbconfig[0], pbattributes);

	GLXContext oldContext = glXCreateContext(display, visual, None, GL_TRUE);
	glXMakeCurrent(display, glxbuffer, oldContext);

 	GLXCREATECONTEXTATTRIBSARBPROC glXCreateContextAttribsARB = (GLXCREATECONTEXTATTRIBSARBPROC) glXGetProcAddress((const GLubyte*)"glXCreateContextAttribsARB");
	int ctxattribs[] = {GLX_CONTEXT_MAJOR_VERSION_ARB, 3, GLX_CONTEXT_MINOR_VERSION_ARB, 0, GLX_CONTEXT_PROFILE_MASK_ARB, GLX_CONTEXT_CORE_PROFILE_BIT_ARB, 0};
	context = glXCreateContextAttribsARB(display, *fbconfig, 0, GL_TRUE, ctxattribs);
	glXMakeCurrent(display, glxbuffer, context);
	glXDestroyContext(display, oldContext);
	
	glewInit();

	glClearColor(0.5f, 0.5f, 0.5f, 0.0f);
	glEnable(GL_DEPTH_TEST);
	glDepthFunc(GL_LESS); 

	programID = LoadShaders();
	mainMatrixID = glGetUniformLocation(programID, "mainMVP");
	sideMatrixID = glGetUniformLocation(programID, "sideMVP");
	textureSamplerID = glGetUniformLocation(programID, "textureSampler");

	glGenVertexArrays(1, &vertexArrayID);
}

RenderGLX::~RenderGLX()
{
	//deinitScene
	glDeleteBuffers(1, &vertexbuffer);
	glDeleteProgram(programID);
	glDeleteVertexArrays(1, &vertexArrayID);

	//deinitSystem
	glXDestroyContext(display, context);
	glXDestroyPbuffer(display, glxbuffer);
	XCloseDisplay(display);
}

void RenderGLX::loadMesh(const Mat points, const Mat indices) {
	assert (indices.isContinuous() && points.isContinuous());

	int face_count = indices.rows;
	vertex_count = face_count*3;
	GLfloat *vertex_buffer_data = new GLfloat[3*vertex_count];
	for (int i=0; i < face_count; i++) {
		for (int j=0; j < 3; j++) {
			const float *point = points.ptr<float>(indices.at<int32_t>(i, j));
			vertex_buffer_data[3*(3*i+j)+0] = point[0]/point[3];
			vertex_buffer_data[3*(3*i+j)+1] = point[1]/point[3];
			vertex_buffer_data[3*(3*i+j)+2] = point[2]/point[3];
		}
	}
	
	glBindVertexArray(vertexArrayID);
	glGenBuffers(1, &vertexbuffer);
	glBindBuffer(GL_ARRAY_BUFFER, vertexbuffer);

	glBufferData(GL_ARRAY_BUFFER, 3*vertex_count*sizeof(GLfloat), vertex_buffer_data, GL_STATIC_DRAW);
	delete vertex_buffer_data;
}

Mat RenderGLX::projected(const Mat camera, const Mat frame, const Mat projector)
{
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glUseProgram(programID);

	glUniformMatrix4fv(mainMatrixID, 1, GL_TRUE, (float*)camera.data);
	glUniformMatrix4fv(sideMatrixID, 1, GL_TRUE, (float*)projector.data);

	glActiveTexture(GL_TEXTURE0);
	GLuint texture = createTexture(frame);
	glBindTexture(GL_TEXTURE_2D, texture);
	// Set our "myTextureSampler" sampler to user Texture Unit 0
	glUniform1i(textureSamplerID, 0);

	glEnableVertexAttribArray(0);
	glBindBuffer(GL_ARRAY_BUFFER, vertexbuffer);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, (void*)0);

	glViewport(0, 0, imgw, imgh);
	glDrawArrays(GL_TRIANGLES, 0, vertex_count);

	glDisableVertexAttribArray(0);
	glDisableVertexAttribArray(1);

	Mat result(imgh, imgw, CV_8UC3);
	glReadBuffer(GL_COLOR_ATTACHMENT0);
	glReadPixels(0, 0, imgw, imgh, GL_RGB, GL_UNSIGNED_BYTE, result.data);
	glDeleteTextures(1, &texture);
	cv::flip(result, result, 0);
	return result;
}	

Mat RenderGLX::depth(const Mat camera) {
	glClear(GL_DEPTH_BUFFER_BIT);

	glUseProgram(programID); //FIXME: disable fragment shader?

	glUniformMatrix4fv(mainMatrixID, 1, GL_TRUE, (float*)camera.data);

	glEnableVertexAttribArray(0);
	glBindBuffer(GL_ARRAY_BUFFER, vertexbuffer);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, (void*)0);

	glViewport(0, 0, imgw, imgh);
	glDrawArrays(GL_TRIANGLES, 0, vertex_count);

	glDisableVertexAttribArray(0);
	glDisableVertexAttribArray(1);

	Mat result(imgh, imgw, CV_32FC1);
	glReadBuffer(GL_DEPTH_ATTACHMENT);
	glReadPixels(0, 0, imgw, imgh, GL_DEPTH_COMPONENT, GL_FLOAT, result.data);
	cv::flip(result, result, 0);
	return result;
}

#ifdef TEST_BUILD
int main(int argc, char ** argv)
{
	Mat MVP(cv::Matx44f(
	-2.16, 0.0, 0.0, 0.0,
	0.0, -2.89, 0.0, 0.0,
	0.0, 0.0, 4.02, 101.,
	0.0, 0.0, -1.0, 0.0));
	 /*1.086396,  0.000000, -1.448528, 0.000000,
	-0.993682,  2.070171, -0.745262, 0.000000,
	-0.687368, -0.515526, -0.515526, 5.642426,
	-0.685994, -0.514496, -0.514496, 5.830953));*/
	Mat sideMVP(cv::Matx44f(
	 0.940846, 0.543198, -1.448528, 0.000000,
	 -1.895640, 1.295979, -0.745262, 0.000000,
	 -0.337515, -0.790143, -0.515526, 5.642426,
	 -0.336840, -0.788564, -0.514496, 5.830953));
	RenderGLX r = RenderGLX(640, 480, (char*)":0");

	Mat points = (cv::Mat_<float>(17, 4) << -0.2664756178855896, -1.791168451309204, -21.19037437438965, 1.0, 5.731060981750488, 3.035867691040039, -23.173986434936523, 1.0, -4.842716693878174, 1.0105935335159302, -21.295154571533203, 1.0, -4.621604919433594, -8.304047584533691, -24.590070724487305, 1.0, 1.674145221710205, -2.720536947250366, -22.068742752075195, 1.0, -0.3098718523979187, 3.13543438911438, -24.555044174194336, 1.0, -4.936746597290039, 7.746241092681885, -27.882299423217773, 1.0, -8.664413452148438, -3.6351583003997803, -23.135316848754883, 1.0, 10.364237785339355, -6.200657367706299, -26.767850875854492, 1.0, 2.185030221939087, -7.903203010559082, -25.241413116455078, 1.0, 3.0006182193756104, 4.61848258972168, -22.982234954833984, 1.0, 9.871418952941895, 0.6571488380432129, -30.438135147094727, 1.0, 5.872250080108643, 5.336030006408691, -33.424095153808594, 1.0, -7.577719211578369, -5.29838228225708, -20.12104034423828, 1.0, 1.0, 0.0, 0.0, 1, 0, 1, 0, 1, 0, 0, 0, 1);
	Mat indices = (cv::Mat_<int32_t>(25,3) << 13, 7, 3, 0, 10, 1, 0, 2, 13, 13, 7, 2, 1, 12, 11, 2, 7, 5, 5, 7, 3, 0, 2, 10, 5, 6, 12, 1, 4, 8, 1, 11, 8, 0, 5, 4, 0, 1, 4, 4, 8, 9, 0, 3, 13, 11, 9, 8, 11, 9, 4, 2, 5, 10, 10, 6, 5, 10, 6, 12, 1, 10, 12, 0, 5, 3, 11, 5, 12, 4, 5, 11, 14, 15, 16);
	r.loadMesh(points, indices);
	Mat tex = cv::imread("opengl/uvtemplate.bmp");
	Mat frame = r.projected(MVP, tex, sideMVP);
			for (int i=0; i<points.rows; i++) {
				Mat point = points.row(i).t();
				point = MVP * point;
				float pointW = point.at<float>(3, 0), pointX = point.at<float>(0, 0)/pointW, pointY = point.at<float>(1, 0)/pointW, pointZ = point.at<float>(2, 0)/pointW;
				cv::Scalar color = (pointZ <= 1 && pointZ >= -1) ? cv::Scalar(128*(1-pointZ), 128*(pointZ+1), 0) : cv::Scalar(0, 0, 255);
				cv::circle(frame, cv::Point(frame.cols*(0.5 + pointX*0.5), frame.rows * (0.5 - pointY*0.5)), 3, color, -1, 8);
			}
	cv::imwrite("projected.png", frame);
	Mat depth = r.depth(MVP);
	double min, max;
	minMaxIdx(depth, &min, &max);
	if (min != max)
		depth = 255 * (depth - min) / (max - min);
	cv::imwrite("depth.png", depth);
	std::cout << "Depth min: " << min << ", max: " << max << std::endl;
	return 0;
}
#endif
