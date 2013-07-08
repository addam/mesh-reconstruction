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
	typedef struct Mesh{
		Mat vertices, faces;
		Mesh(Mat v, Mat f):vertices(v), faces(f) {};} Mesh;
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
		virtual void loadMesh(const Mesh mesh);
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
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, flipped.cols, flipped.rows, 0, GL_BGR, GL_UNSIGNED_BYTE, flipped.data);
	
	// nearest neighbor filtering
	//glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	//glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST); 

	GLfloat anisotropy;
	glGetFloatv(GL_MAX_TEXTURE_MAX_ANISOTROPY_EXT, &anisotropy);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAX_ANISOTROPY_EXT, anisotropy); 
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	//glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
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
	if (vertexbuffer != -1)
		glDeleteBuffers(1, &vertexbuffer);
	glDeleteProgram(programID);
	glDeleteVertexArrays(1, &vertexArrayID);

	//deinitSystem
	glXDestroyContext(display, context);
	glXDestroyPbuffer(display, glxbuffer);
	XCloseDisplay(display);
}

void RenderGLX::loadMesh(const Mesh mesh) {
	assert (mesh.vertices.isContinuous() && mesh.faces.isContinuous());

	int face_count = mesh.faces.rows;
	vertex_count = face_count*3;
	GLfloat *vertex_buffer_data = new GLfloat[3*vertex_count];
	for (int i=0; i < face_count; i++) {
		const int32_t *face = mesh.faces.ptr<int32_t>(i);
		for (int j=0; j < 3; j++) {
			const float *point = mesh.vertices.ptr<float>(face[j]);
			vertex_buffer_data[3*(3*i+j)+0] = point[0]/point[3];
			vertex_buffer_data[3*(3*i+j)+1] = point[1]/point[3];
			vertex_buffer_data[3*(3*i+j)+2] = point[2]/point[3];
		}
	}
	
	glBindVertexArray(vertexArrayID);
	if (vertexbuffer != -1)
		glDeleteBuffers(1, &vertexbuffer);
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

//BEGIN SHADOW MAP
/*	 
	// Poor filtering. Needed !
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	
	GLuint shadowbuffer;
	glGenBuffers(1, &shadowbuffer);
	glBufferData(GL_PIXEL_PACK_BUFFER, imgw*imgh, 0, GL_STREAM_COPY);
	glBindBuffer(GL_PIXEL_PACK_BUFFER, shadowbuffer);

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glViewport(0,0,imgw, imgh);
	glDrawArrays(GL_TRIANGLES, 0, vertex_count);
	
	glReadBuffer(GL_DEPTH_ATTACHMENT);
	glReadPixels(0, 0, imgw, imgh, GL_DEPTH_COMPONENT, GL_FLOAT, 0);
	glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);
	
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, shadowbuffer);
	// The texture we're going to render to
	GLuint renderedTexture;
	glGenTextures(1, &renderedTexture);
	// "Bind" the newly created texture : all future texture functions will modify this texture
	glBindTexture(GL_TEXTURE_2D, renderedTexture);
	// Give an empty image to OpenGL ( the last "0" ) -- no, it should be read from the buffer currently bound
	glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT, imgw, imgh, 0, GL_RED, GL_FLOAT, 0);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
	glBindTexture(GL_TEXTURE_2D, 0);
*/
//END SHADOW MAP

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
	glReadPixels(0, 0, imgw, imgh, GL_BGR, GL_UNSIGNED_BYTE, result.data);
	glDeleteTextures(1, &texture);
//	glDeleteBuffers(1, &shadowbuffer); //DELETE SHADOW MAP
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
	result = 2*result - 1;
	return result;
}

#ifdef TEST_BUILD
int main(int argc, char ** argv)
{
	RenderGLX r = RenderGLX(640, 480, (char*)":0");

Mat points = (cv::Mat_<float>(25.0, 4) << 0.5127, -3.9222, -29.4300, 1.0000, 0.6195, -0.2643, -27.4378, 1.0000, 4.5767, 0.2684, -28.6282, 1.0000, 4.4699, -3.3895, -30.6204, 1.0000, 1.8125, -5.8448, -25.9695, 1.0000, 1.9193, -2.1869, -23.9774, 1.0000, 5.8765, -1.6541, -25.1678, 1.0000, -3.7263, 1.9956, -20.7352, 1.0000, -5.1135, -5.5956, -28.2388, 1.0000, -5.0067, -1.9377, -26.2467, 1.0000, -1.0495, -1.4050, -27.4371, 1.0000, -1.1563, -5.0629, -29.4292, 1.0000, -3.8137, -7.5182, -24.7784, 1.0000, 0.2503, -3.3276, -23.9766, 1.0000, 0.1435, -6.9855, -25.9688, 1.0000, -4.5209, -0.3826, -22.9609, 1.0000, -4.4455, 2.1991, -21.5549, 1.0000, -1.6526, 2.5750, -22.3950, 1.0000, -1.7281, -0.0066, -23.8010, 1.0000, -3.6036, -1.7395, -20.5186, 1.0000, -3.5282, 0.8422, -19.1126, 1.0000, -0.7353, 1.2181, -19.9528, 1.0000, -0.8107, -1.3635, -21.3588, 1.0000, -3.3029, 1.3693, -19.6080, 1.0000, -2.0139, 1.5429, -19.9957, 1.0000);
Mat indices = (cv::Mat_<int32_t>(27.0, 3) << 4, 5, 1, 5, 6, 1, 0, 1, 2, 13, 14, 11, 14, 12, 8, 8, 9, 10, 19, 20, 16, 20, 21, 16, 21, 22, 17, 22, 19, 18, 15, 16, 17, 22, 21, 20, 0, 4, 1, 21, 17, 16, 13, 10, 9, 3, 0, 2, 8, 12, 9, 22, 18, 17, 10, 13, 11, 11, 14, 8, 11, 8, 10, 15, 19, 16, 23, 24, 7, 6, 2, 1, 18, 15, 17, 19, 22, 20, 19, 15, 18);
Mat MVP(cv::Matx44f(-1.195982575416565, 1.350219488143921, 1.237614393234253, 30.956573486328125, -0.1888779103755951, -2.055802583694458, 2.06032657623291, 47.59274673461914, -0.8364689946174622, -0.35027408599853516, -0.4261872172355652, 7.458727836608887, -0.834797739982605, -0.3495742380619049, -0.42533570528030396, 7.643625259399414));
Mat sideMVP(cv::Matx44f(-1.831691861152649, -1.1502554416656494, -0.3270684480667114, -11.764444351196289, 1.391772985458374, -2.4397428035736084, 0.7858548760414124, 19.515047073364258, 0.267280250787735, -0.1545732319355011, -0.9532451629638672, -12.715036392211914, 0.2667462229728699, -0.1542643904685974, -0.9513405561447144, -12.489831924438477));
MVP = sideMVP;

	r.loadMesh(Mesh(points, indices));
	Mat tex = cv::imread("opengl/grid.png");
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
