#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <iostream>

#include <X11/Xlib.h>
#include <X11/Xutil.h>
#include <GL/glew.h>
#include <GL/glx.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

//sets the variables vertexShaderSources, fragmentShaderSources
#include "shaders.hpp"

#include "recon.hpp"

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
		GLuint programID, MatrixID, InvMatrixID, Texture, TextureID, vertexbuffer, VertexArrayID, imgw, imgh;
		Display *display;
		GLXContext context;
		GLXPbuffer glxbuffer;
};

Render *spawnRender(Heuristic hint)
{
	RenderGLX *render = new RenderGLX(640, 480, getenv("DISPLAY"));
	return render;
}

GLuint createTexture(const Mat image){
	GLuint textureID;
	glGenTextures(1, &textureID);
	glBindTexture(GL_TEXTURE_2D, textureID);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, image.cols, image.rows, 0, GL_BGR, GL_UNSIGNED_BYTE, image.data);
	
	// nearest neighbor filtering
	//glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	//glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST); 

	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR); 
	glGenerateMipmap(GL_TEXTURE_2D);

	return textureID;
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

	// Check Vertex Shader
	glGetShaderiv(VertexShaderID, GL_COMPILE_STATUS, &Result);
	glGetShaderiv(VertexShaderID, GL_INFO_LOG_LENGTH, &InfoLogLength);
	if (InfoLogLength > 1){
		std::string VertexShaderErrorMessage(InfoLogLength+1, 0);
		glGetShaderInfoLog(VertexShaderID, InfoLogLength, NULL, &VertexShaderErrorMessage[0]);
		std::cerr << VertexShaderErrorMessage;
	}

	// Compile Fragment Shader
	glShaderSource(FragmentShaderID, 1, fragmentShaderSources, NULL);
	glCompileShader(FragmentShaderID);

	// Check Fragment Shader
	glGetShaderiv(FragmentShaderID, GL_COMPILE_STATUS, &Result);
	glGetShaderiv(FragmentShaderID, GL_INFO_LOG_LENGTH, &InfoLogLength);
	if (InfoLogLength > 1){
		std::string FragmentShaderErrorMessage(InfoLogLength+1, 0);
		glGetShaderInfoLog(FragmentShaderID, InfoLogLength, NULL, &FragmentShaderErrorMessage[0]);
		std::cerr << FragmentShaderErrorMessage;
	}

	// Link the program
	GLuint ProgramID = glCreateProgram();
	glAttachShader(ProgramID, VertexShaderID);
	glAttachShader(ProgramID, FragmentShaderID);
	glLinkProgram(ProgramID);

	// Check the program
	glGetProgramiv(ProgramID, GL_LINK_STATUS, &Result);
	glGetProgramiv(ProgramID, GL_INFO_LOG_LENGTH, &InfoLogLength);
	if (InfoLogLength > 1){
		std::string ProgramErrorMessage(InfoLogLength+1, 0);
		glGetProgramInfoLog(ProgramID, InfoLogLength, NULL, &ProgramErrorMessage[0]);
		std::cerr << ProgramErrorMessage;
	}

	glDeleteShader(VertexShaderID);
	glDeleteShader(FragmentShaderID);

	return ProgramID;
}

RenderGLX::RenderGLX(int width, int height, char *displayName)
{
	vertexbuffer = -1;
	
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

	glClearColor(0.0f, 0.0f, 0.3f, 0.0f);
	glEnable(GL_DEPTH_TEST);
	glDepthFunc(GL_LESS); 

	programID = LoadShaders();
	MatrixID = glGetUniformLocation(programID, "MVP");
	InvMatrixID = glGetUniformLocation(programID, "sideMVP");
	TextureID = glGetUniformLocation(programID, "myTextureSampler");

	Texture = createTexture(cv::imread("uvtemplate.bmp"));
	glGenVertexArrays(1, &VertexArrayID);
}

RenderGLX::~RenderGLX()
{
	//deinitScene
	glDeleteBuffers(1, &vertexbuffer);
	glDeleteProgram(programID);
	glDeleteTextures(1, &TextureID);
	glDeleteVertexArrays(1, &VertexArrayID);

	//deinitSystem
	glXDestroyContext(display, context);
	glXDestroyPbuffer(display, glxbuffer);
	XCloseDisplay(display);
}

void RenderGLX::loadMesh(const Mat points, const Mat indices) {
	assert (indices.isContinuous() && points.isContinuous());

	const int *idx = indices.ptr<int>(0);
	int vertex_count = indices.rows;
	GLfloat *vertex_buffer_data = new GLfloat[3*vertex_count];
	for (int i=0; i < vertex_count;) {
		const float *row = points.ptr<float>(idx[i]);
		vertex_buffer_data[i++] = row[0];
		vertex_buffer_data[i++] = row[1];
		vertex_buffer_data[i++] = row[2];
	}
	
	glBindVertexArray(VertexArrayID);
	glGenBuffers(1, &vertexbuffer);
	glBindBuffer(GL_ARRAY_BUFFER, vertexbuffer);
	glBufferData(GL_ARRAY_BUFFER, sizeof(vertex_buffer_data), vertex_buffer_data, GL_STATIC_DRAW);	
}

Mat RenderGLX::projected(const Mat camera, const Mat frame, const Mat projector)
{
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glUseProgram(programID);
	glUniformMatrix4fv(MatrixID, 1, GL_FALSE, (float*)camera.data);
	glUniformMatrix4fv(InvMatrixID, 1, GL_FALSE, (float*)projector.data);

	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, Texture);
	// Set our "myTextureSampler" sampler to user Texture Unit 0
	glUniform1i(TextureID, 0);

	glEnableVertexAttribArray(0);
	glBindBuffer(GL_ARRAY_BUFFER, vertexbuffer);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, (void*)0);

	glViewport(0, 0, imgw, imgh);
	glDrawArrays(GL_TRIANGLES, 0, 12*3); // 12*3 indices starting at 0 -> 12 triangles

	glDisableVertexAttribArray(0);
	glDisableVertexAttribArray(1);

	Mat result(imgw, imgh, CV_8UC3);
	glReadBuffer(GL_COLOR_ATTACHMENT0);
	glReadPixels(0, 0, imgw, imgh, GL_RGB, GL_UNSIGNED_BYTE, result.data);
}	

Mat RenderGLX::depth(const Mat camera) {
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glEnableVertexAttribArray(0);
	glBindBuffer(GL_ARRAY_BUFFER, vertexbuffer);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, (void*)0);

	glViewport(0, 0, imgw, imgh);
	glDrawArrays(GL_TRIANGLES, 0, 12*3); // 12*3 indices starting at 0 -> 12 triangles

	glDisableVertexAttribArray(0);
	glDisableVertexAttribArray(1);

	Mat result(imgw, imgh, CV_8UC1);
	glReadBuffer(GL_DEPTH_ATTACHMENT);
	glReadPixels(0, 0, imgw, imgh, GL_DEPTH_COMPONENT, GL_UNSIGNED_BYTE, result.data);
	return result;
}

#ifdef TEST_BUILD
void save(const char* filename)
{
	unsigned char *data = new unsigned char[imgw*imgh*3];
	glReadBuffer(GL_COLOR_ATTACHMENT0);
	glReadPixels(0, 0, imgw, imgh, GL_RGB, GL_UNSIGNED_BYTE, &data[0]);
	
	FILE* outf = fopen(filename, "wb+");
	for (int i=0; i<imgw*imgh; i++) {
		fputc((unsigned char)(data[3*i]), outf);
		fputc((unsigned char)(data[3*i+1]), outf);
		fputc((unsigned char)(data[3*i+2]), outf);
	}
	fclose(outf);
	delete data;
}

int main(int argc, char ** argv)
{
	Mat MVP = Mat(cv::Matx44f(
	 1.086396, -0.993682, -0.687368, -0.685994,
	 0.000000,  2.070171, -0.515526, -0.514496,
	-1.448528, -0.745262, -0.515526, -0.514496,
	 0.000000,  0.000000,  5.642426,  5.830953));
	Mat sideMVP = Mat(cv::Matx44f(
	 0.940846, -1.895640, -0.337515, -0.336840,
	 0.543198, 1.295979, -0.790143, -0.788564,
	 -1.448528, -0.745262, -0.515526, -0.514496,
	 0.000000, 0.000000, 5.642426, 5.830953));
	initSystem();
	initScene();
	render(MVP, sideMVP);
	save("frame.data");
	deinitScene();
	deinitSystem();
	return 0;
}
#endif
