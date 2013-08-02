// render_glx.cpp: wrapper for off-screen rendering using glX

#include <X11/Xlib.h>
#include <X11/Xutil.h>
#include <GL/glew.h>
#include <GL/glx.h>

//sets the variables vertexShaderSources, fragmentShaderSources
#include "shaders.hpp"

#ifdef TEST_BUILD
	#include <cstdio>
	#include <cstdlib>
	#include <string>
	#include <iostream>
	#include <fstream>
	
	#include <opencv2/core/core.hpp>
	#include <opencv2/highgui/highgui.hpp>
	#include <opencv2/imgproc/imgproc.hpp>
	typedef cv::Mat Mat;
	class Render {};
	class Heuristic {public: cv::Size renderSize(){return cv::Size(0,0);};};
	typedef struct Mesh{
		Mat vertices, faces;
		Mesh(Mat v, Mat f):vertices(v), faces(f) {};} Mesh;
#else
	#include "recon.hpp"
#endif

// pointer to a function that can create an OpenGL 3.0 context
typedef GLXContext (*GLXCREATECONTEXTATTRIBSARBPROC)(Display*, GLXFBConfig, GLXContext, Bool, const int*);

// Specialization of the generic Render class
class RenderGLX;

class RenderGLX: public Render {
	public:
		RenderGLX(int width, int height, char *displayName);
		~RenderGLX();
		virtual void loadMesh(const Mesh mesh);
		virtual Mat projected(const Mat camera, const Mat frame, const Mat projector);
		virtual Mat depth(const Mat camera);
	protected:
		static int instanceCount;
		GLuint programID, mainMatrixID, sideMatrixID, textureSamplerID, shadowSamplerID, vertexbuffer, vertexArrayID, imgw, imgh;
		Display *display;
		GLXContext context;
		GLXPbuffer glxbuffer;
		int vertex_count;
};
// To solve some uncomfortable interference, some resources are freed just at the end of the program
int RenderGLX::instanceCount = 0;

// A generic function to create a Render instance; if this cpp file is used, it will be a RenderGLX instance
Render *spawnRender(Heuristic hint)
{
	cv::Size size = hint.renderSize();
	RenderGLX *render = new RenderGLX(size.width, size.height, getenv("DISPLAY"));
	return render;
}

// Load a given grayscale image into OpenGL
GLuint createTexture(const Mat image){
	assert(image.channels() == 1);
	// OpenCV stores images top-down flipped
	Mat flipped(image.rows, image.cols, 1);
	cv::flip(image, flipped, 0);
	
	// Allocate texture data
	GLuint texture;
	glGenTextures(1, &texture);
	glBindTexture(GL_TEXTURE_2D, texture);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RED, flipped.cols, flipped.rows, 0, GL_RED, GL_UNSIGNED_BYTE, flipped.data);
	
	// Set texture quality parameters
	GLfloat anisotropy;
	glGetFloatv(GL_MAX_TEXTURE_MAX_ANISOTROPY_EXT, &anisotropy); // get maximum value for anisotropic filtering
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAX_ANISOTROPY_EXT, anisotropy); // set this value
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT); // coordinates out of image domain are filled with 'wrap'
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT); // (but this does not matter as we mask them out anyway)
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR); // magnifying filter: bilinear interpolation
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR); // minifying filter: (anisotropic) mipmap interpolated across levels
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

// Initialize all system resources necessary for rendering (program crashes if this is unsuccessful)
RenderGLX::RenderGLX(int width, int height, char *displayName)
{
	instanceCount += 1;
	
	vertexbuffer = -1;
	vertex_count = 0;
	
	imgw = width;
	imgh = height;
	_Xdebug = 1;
	display = XOpenDisplay(displayName);
	XSynchronize(display, 0);

	// Get a glX 'visual' structure on the default screen
	// NOTE: for some reason, my system has only a RGBA Visual and only a RGB FrameBuffer. Some systems may require to remove the GLX_RGBA flag
	int visualattributes[] = {GLX_RGBA, GLX_RED_SIZE, 8, GLX_GREEN_SIZE, 8, GLX_BLUE_SIZE, 8, GLX_DEPTH_SIZE, 1, None};
	XVisualInfo *visual = glXChooseVisual(display, XDefaultScreen(display), visualattributes);
	
	// get a glX framebuffer configuration
	int dummy;
	// NOTE: some systems may require to add the GLX_RGBA flag
	int fbattributes[] = {GLX_RED_SIZE, 8, GLX_GREEN_SIZE, 8, GLX_BLUE_SIZE, 8, GLX_DEPTH_SIZE, 1, None};
	GLXFBConfig *fbconfig = glXChooseFBConfig(display, visual->screen, fbattributes, &dummy);

	// Create a PBuffer object
	int pbattributes[] = {GLX_PBUFFER_WIDTH, imgw, GLX_PBUFFER_HEIGHT, imgh, None};
	glxbuffer = glXCreatePbuffer(display, fbconfig[0], pbattributes);

	// Firstly, we have to create a basic OpenGL context using the only function we can use
	GLXContext oldContext = glXCreateContext(display, visual, None, GL_TRUE);
	glXMakeCurrent(display, glxbuffer, oldContext);

	// Secondly, we use the basic context to get a more advanced (3.0) one
 	GLXCREATECONTEXTATTRIBSARBPROC glXCreateContextAttribsARB = (GLXCREATECONTEXTATTRIBSARBPROC) glXGetProcAddress((const GLubyte*)"glXCreateContextAttribsARB");
	int ctxattribs[] = {GLX_CONTEXT_MAJOR_VERSION_ARB, 3, GLX_CONTEXT_MINOR_VERSION_ARB, 0, GLX_CONTEXT_PROFILE_MASK_ARB, GLX_CONTEXT_CORE_PROFILE_BIT_ARB, 0};
	context = glXCreateContextAttribsARB(display, *fbconfig, 0, GL_TRUE, ctxattribs);
	glXMakeCurrent(display, glxbuffer, context);
	glXDestroyContext(display, oldContext);
	
	// Supplies all available OpenGL extensions to be directly
	glewInit();

	// Basic scene settings
	glClearColor(0,0,0,0);
	glEnable(GL_DEPTH_TEST);
	glDepthFunc(GL_LESS); 

	// Load the vertex and fragment shader, and remember path to their parameters
	programID = LoadShaders();
	mainMatrixID = glGetUniformLocation(programID, "mainMVP");
	sideMatrixID = glGetUniformLocation(programID, "sideMVP");
	shadowSamplerID = glGetUniformLocation(programID, "shadowSampler");
	textureSamplerID = glGetUniformLocation(programID, "textureSampler");

	// prepare an empty Vertex Buffer Object
	glGenVertexArrays(1, &vertexArrayID);
}

RenderGLX::~RenderGLX()
{
	instanceCount -= 1;
	
	// these buffers seem to interfere, may be caused by some optimization
	if (vertexbuffer != -1 && instanceCount == 0)
		glDeleteBuffers(1, &vertexbuffer);

	// Deallocate resources
	glDeleteProgram(programID);
	glDeleteVertexArrays(1, &vertexArrayID);
	glXDestroyContext(display, context);
	glXDestroyPbuffer(display, glxbuffer);

	// Display is closed only at the end of the program
	if (instanceCount == 0)
		XCloseDisplay(display);
}

// loads given Mesh structure into the OpenGL Vertex Buffer Object for rendering
void RenderGLX::loadMesh(const Mesh mesh) {
	assert (mesh.vertices.isContinuous() && mesh.faces.isContinuous());

	// the structure is just a list of triplets of vertices, each denoting a single face
	// vertices are repeated as necessary
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
	
	// Set this vertex buffer to be used during rendering
	glBindVertexArray(vertexArrayID);
	if (vertexbuffer != -1)
		glDeleteBuffers(1, &vertexbuffer);
	glGenBuffers(1, &vertexbuffer);
	glBindBuffer(GL_ARRAY_BUFFER, vertexbuffer);
	
	// Load the vertex positions into the GPU memory
	glBufferData(GL_ARRAY_BUFFER, 3*vertex_count*sizeof(GLfloat), vertex_buffer_data, GL_STATIC_DRAW);
	delete vertex_buffer_data;
}

// Renders the (previously loaded) scene from the given (main) camera, with frame being projected from projector (aka. side camera)
Mat RenderGLX::projected(const Mat camera, const Mat frame, const Mat projector)
{
	glUseProgram(programID);

	glUniformMatrix4fv(mainMatrixID, 1, GL_TRUE, (float*)projector.data);
	glUniformMatrix4fv(sideMatrixID, 1, GL_TRUE, (float*)projector.data);

	glEnableVertexAttribArray(0);
	glBindBuffer(GL_ARRAY_BUFFER, vertexbuffer);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, (void*)0);

	// == BEGIN generate a shadow map ==
	Mat shadow(imgh, imgw, CV_32FC1);
	GLuint shadowMapTexture;
	{
		glClear(GL_DEPTH_BUFFER_BIT);
		
		// Render the scene without setting any texture or projector
		glViewport(0, 0, imgw, imgh);
		glDrawArrays(GL_TRIANGLES, 0, vertex_count);
		
		glDisableVertexAttribArray(0);
		glDisableVertexAttribArray(1);
	
		// Read the depth data into the matrix (flipping not necessary as we read them back later)
		glReadPixels(0, 0, imgw, imgh, GL_DEPTH_COMPONENT, GL_FLOAT, shadow.data);
		{
			// apply a size 1 dilation filter to reduce shadow acne
			float *prevRowHF = new float[shadow.cols]; // values on the previous row after horizontal filtering
			float *curRow = shadow.ptr<float>(0), *prevRow; // current and previous row of the image, respectively
			float prevVal = curRow[0]; // value of the pixel to the left, before filtering
			for (int j=1; j<shadow.cols-1; j++) {
				prevRowHF[j] = curRow[j];
				if (curRow[j-1] < prevRowHF[j]) prevRowHF[j] = curRow[j-1];
				if (curRow[j+1] < prevRowHF[j]) prevRowHF[j] = curRow[j+1];
				curRow[j] = prevRowHF[j];
			}
			for (int i=1; i<shadow.rows; i++) {
				prevRow = curRow;
				curRow = shadow.ptr<float>(i);
				float prevVal = curRow[0];
				for (int j=1; j<shadow.cols-1; j++) {
					float val = curRow[j];
					if (prevVal > curRow[j]) curRow[j] = prevVal;
					if (curRow[j+1] > curRow[j]) curRow[j] = curRow[j+1];
					float valHF = curRow[j];
					if (prevRowHF[j] > curRow[j]) curRow[j] = prevRowHF[j];
					if (curRow[j] > prevRow[j]) prevRow[j] = curRow[j];
					prevRowHF[j] = valHF;
					prevVal = val;
				}
			}
			delete prevRowHF;
		}
	
		// set shadow map parameters
		glGenTextures(1, &shadowMapTexture);
		glBindTexture(GL_TEXTURE_2D, shadowMapTexture);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST); // magnifying filter: nearest neighbor
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST); // minifying filter: neares neighbor
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
		
		// load the shadow map back into GPU
		glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT, imgw, imgh, 0, GL_DEPTH_COMPONENT, GL_FLOAT, shadow.data);

		glBindTexture(GL_TEXTURE_2D, 0);
	}
	// == END generate a shadow map ==

	// again, set all necessary scene parameters
	glEnableVertexAttribArray(0);
	glBindBuffer(GL_ARRAY_BUFFER, vertexbuffer);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, (void*)0);

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	glUniformMatrix4fv(mainMatrixID, 1, GL_TRUE, (float*)camera.data);
	GLuint texture = createTexture(frame);
	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, texture);
	glActiveTexture(GL_TEXTURE1);
	glBindTexture(GL_TEXTURE_2D, shadowMapTexture);

	glUniform1i(textureSamplerID, 0);
	glUniform1i(shadowSamplerID, 1);

	// Render the scene
	glViewport(0, 0, imgw, imgh);
	glDrawArrays(GL_TRIANGLES, 0, vertex_count);

	glDisableVertexAttribArray(0);
	glDisableVertexAttribArray(1);

	// Read off rendered image to the matrix
	Mat result(imgh, imgw, CV_8UC3);
	glReadBuffer(GL_FRONT);
	// although it is a RGB, only the red channel is the actual data; G, B are the mask
	glReadPixels(0, 0, imgw, imgh, GL_RGB, GL_UNSIGNED_BYTE, result.data);
	
	glDeleteTextures(1, &texture);
	glDeleteTextures(1, &shadowMapTexture);
	
	// Flip the result top-down before returning
	cv::flip(result, result, 0);
	return result;
}	

Mat RenderGLX::depth(const Mat camera) {
	glClear(GL_DEPTH_BUFFER_BIT);

	// render without using any texture nor projector
	glUseProgram(programID);

	glUniformMatrix4fv(mainMatrixID, 1, GL_TRUE, (float*)camera.data);

	glEnableVertexAttribArray(0);
	glBindBuffer(GL_ARRAY_BUFFER, vertexbuffer);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, (void*)0);

	// render the scene
	glViewport(0, 0, imgw, imgh);
	glDrawArrays(GL_TRIANGLES, 0, vertex_count);

	glDisableVertexAttribArray(0);
	glDisableVertexAttribArray(1);

	// read off depth data
	Mat result(imgh, imgw, CV_32FC1);
	glReadBuffer(GL_DEPTH_ATTACHMENT);
	glReadPixels(0, 0, imgw, imgh, GL_DEPTH_COMPONENT, GL_FLOAT, result.data);
	cv::flip(result, result, 0);

	// remap the values linearly to get actual depth map values
	result = 2*result - 1;
	return result;
}

#ifdef TEST_BUILD
// Main function for testing of the shadows etc.
// should output files 'depth.png' (black and white) and 'projected.png' (mostly yellow and black)
// If this function does not run, this code is incompatible with your system
int main(int argc, char ** argv)
{
	RenderGLX r = RenderGLX(640, 480, (char*)":0");

Mat points = (cv::Mat_<float>(25.0, 4) << 0.5127, -3.9222, -29.4300, 1.0000, 0.6195, -0.2643, -27.4378, 1.0000, 4.5767, 0.2684, -28.6282, 1.0000, 4.4699, -3.3895, -30.6204, 1.0000, 1.8125, -5.8448, -25.9695, 1.0000, 1.9193, -2.1869, -23.9774, 1.0000, 5.8765, -1.6541, -25.1678, 1.0000, -3.7263, 1.9956, -20.7352, 1.0000, -5.1135, -5.5956, -28.2388, 1.0000, -5.0067, -1.9377, -26.2467, 1.0000, -1.0495, -1.4050, -27.4371, 1.0000, -1.1563, -5.0629, -29.4292, 1.0000, -3.8137, -7.5182, -24.7784, 1.0000, 0.2503, -3.3276, -23.9766, 1.0000, 0.1435, -6.9855, -25.9688, 1.0000, -4.5209, -0.3826, -22.9609, 1.0000, -4.4455, 2.1991, -21.5549, 1.0000, -1.6526, 2.5750, -22.3950, 1.0000, -1.7281, -0.0066, -23.8010, 1.0000, -3.6036, -1.7395, -20.5186, 1.0000, -3.5282, 0.8422, -19.1126, 1.0000, -0.7353, 1.2181, -19.9528, 1.0000, -0.8107, -1.3635, -21.3588, 1.0000, -3.3029, 1.3693, -19.6080, 1.0000, -2.0139, 1.5429, -19.9957, 1.0000);
Mat indices = (cv::Mat_<int32_t>(27.0, 3) << 4, 5, 1, 5, 6, 1, 0, 1, 2, 13, 14, 11, 14, 12, 8, 8, 9, 10, 19, 20, 16, 20, 21, 16, 21, 22, 17, 22, 19, 18, 15, 16, 17, 22, 21, 20, 0, 4, 1, 21, 17, 16, 13, 10, 9, 3, 0, 2, 8, 12, 9, 22, 18, 17, 10, 13, 11, 11, 14, 8, 11, 8, 10, 15, 19, 16, 23, 24, 7, 6, 2, 1, 18, 15, 17, 19, 22, 20, 19, 15, 18);
Mat MVP(cv::Matx44f(-1.195982575416565, 1.350219488143921, 1.237614393234253, 30.956573486328125, -0.1888779103755951, -2.055802583694458, 2.06032657623291, 47.59274673461914, -1.0203083753585815, -0.42725738883018494, -0.519854724407196, 2.6755423545837402, -0.834797739982605, -0.3495742380619049, -0.42533570528030396, 7.643625259399414));
Mat sideMVP(cv::Matx44f(-1.831691861152649, -1.1502554416656494, -0.3270684480667114, -11.764444351196289, 1.391772985458374, -2.4397428035736084, 0.7858548760414124, 19.515047073364258, 0.3260231614112854, -0.188545361161232, -1.1627495288848877, -21.932016372680664, 0.2667462229728699, -0.1542643904685974, -0.9513405561447144, -12.489831924438477));

	r.loadMesh(Mesh(points, indices));
	Mat tex = cv::imread("test/grid.png");
	cv::cvtColor(tex, tex, CV_RGB2GRAY);
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
