#ifndef RENDER_GLX_CPP
#define RENDER_GLX_CPP

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

GLuint createTexture(const Mat image){
	GLuint textureID;
	glGenTextures(1, &textureID);
	glBindTexture(GL_TEXTURE_2D, textureID);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, image.cols, image.rows, 0, GL_BGR, GL_UNSIGNED_BYTE, image.data);
	
	// Poor filtering, or ...
	//glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	//glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST); 

	// ... nice trilinear filtering.
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

// pryč s tim -> class
GLuint programID, MatrixID, InvMatrixID, Texture, TextureID, vertexbuffer, VertexArrayID, uvbuffer, framebuffer, renderbuffer, imgw=800, imgh=600;

Render::Render(Heuristic hint)
{
}

Render::~Render()
{
}

Mat Render::projected(const Mat camera, const Mat frame, const Mat projector, const Mat points, const Mat indices)
{
	Mat result;
	return result;
}
void render(const Mat camera, const Mat projector)
{
	// Clear the screen
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	// Use our shader
	glUseProgram(programID);

	// Send our transformation to the currently bound shader, 
	// in the "MVP" uniform
	glUniformMatrix4fv(MatrixID, 1, GL_FALSE, (float*)camera.data);

	glUniformMatrix4fv(InvMatrixID, 1, GL_FALSE, (float*)projector.data);

	// Bind our texture in Texture Unit 0
	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, Texture);
	// Set our "myTextureSampler" sampler to user Texture Unit 0
	glUniform1i(TextureID, 0);

	// 1rst attribute buffer : vertices
	glEnableVertexAttribArray(0);
	glBindBuffer(GL_ARRAY_BUFFER, vertexbuffer);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, (void*)0);

	glViewport(0, 0, imgw, imgh);
	glDrawArrays(GL_TRIANGLES, 0, 12*3); // 12*3 indices starting at 0 -> 12 triangles

	glDisableVertexAttribArray(0);
	glDisableVertexAttribArray(1);
}	

Mat Render::depth(const Mat camera, const Mat points, const Mat indices) {
	
}

Display *display;
GLXContext context;
GLXPbuffer glxbuffer;

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

void initSystem()
{
	_Xdebug = 1;
	display = XOpenDisplay(getenv("DISPLAY"));
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
}


void deinitSystem()
{
	glXDestroyContext(display, context);
	glXDestroyPbuffer(display, glxbuffer);
	XCloseDisplay(display);
}

void initScene()
{
	glClearColor(0.0f, 0.0f, 0.3f, 0.0f);
	glEnable(GL_DEPTH_TEST);
	glDepthFunc(GL_LESS); 

	programID = LoadShaders();
	MatrixID = glGetUniformLocation(programID, "MVP");
	InvMatrixID = glGetUniformLocation(programID, "sideMVP");

	Texture = createTexture(cv::imread("uvtemplate.bmp"));
	TextureID = glGetUniformLocation(programID, "myTextureSampler");
	
	static const GLfloat g_vertex_buffer_data[] = { 
		-1.0f,-1.0f,-1.0f,
		-1.0f,-1.0f, 1.0f,
		-1.0f, 1.0f, 1.0f,
		 1.0f, 1.0f,-1.0f,
		-1.0f,-1.0f,-1.0f,
		-1.0f, 1.0f,-1.0f,
		 1.0f,-1.0f, 1.0f,
		-1.0f,-1.0f,-1.0f,
		 1.0f,-1.0f,-1.0f,
		 1.0f, 1.0f,-1.0f,
		 1.0f,-1.0f,-1.0f,
		-1.0f,-1.0f,-1.0f,
		-1.0f,-1.0f,-1.0f,
		-1.0f, 1.0f, 1.0f,
		-1.0f, 1.0f,-1.0f,
		 1.0f,-1.0f, 1.0f,
		-1.0f,-1.0f, 1.0f,
		-1.0f,-1.0f,-1.0f,
		-1.0f, 1.0f, 1.0f,
		-1.0f,-1.0f, 1.0f,
		 1.0f,-1.0f, 1.0f,
		 1.0f, 1.0f, 1.0f,
		 1.0f,-1.0f,-1.0f,
		 1.0f, 1.0f,-1.0f,
		 1.0f,-1.0f,-1.0f,
		 1.0f, 1.0f, 1.0f,
		 1.0f,-1.0f, 1.0f,
		 1.0f, 1.0f, 1.0f,
		 1.0f, 1.0f,-1.0f,
		-1.0f, 1.0f,-1.0f,
		 1.0f, 1.0f, 1.0f,
		-1.0f, 1.0f,-1.0f,
		-1.0f, 1.0f, 1.0f,
		 1.0f, 1.0f, 1.0f,
		-1.0f, 1.0f, 1.0f,
		 1.0f,-1.0f, 1.0f
	};

	glGenVertexArrays(1, &VertexArrayID);
	glBindVertexArray(VertexArrayID);
	glGenBuffers(1, &vertexbuffer);
	glBindBuffer(GL_ARRAY_BUFFER, vertexbuffer);
	glBufferData(GL_ARRAY_BUFFER, sizeof(g_vertex_buffer_data), g_vertex_buffer_data, GL_STATIC_DRAW);
}

void deinitScene()
{
	glDeleteBuffers(1, &vertexbuffer);
	glDeleteBuffers(1, &uvbuffer);
	glDeleteProgram(programID);
	glDeleteTextures(1, &TextureID);
	glDeleteVertexArrays(1, &VertexArrayID);
}

#ifdef TEST_BUILD
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
#endif
