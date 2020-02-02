#version 320 es

layout(location = 0) in vec3 vertexPosition_modelspace;

out vec3 pos; // world space position; will get linearly interpolated for all fragments

uniform mat4 mainMVP;

void main(){
	// set vertex position from the Vertex Buffer Object
	gl_Position = mainMVP * vec4(vertexPosition_modelspace, 1);
	pos = vertexPosition_modelspace;
}

