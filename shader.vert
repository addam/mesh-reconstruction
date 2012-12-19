#version 330 core

// Input vertex data, different for all executions of this shader.
layout(location = 0) in vec3 vertexPosition_modelspace;
layout(location = 1) in vec2 vertexUV;

// Output data ; will be interpolated for each fragment.
out vec3 pos;

// Values that stay constant for the whole mesh.
uniform mat4 MVP;

void main(){
	gl_Position =  MVP * vec4(vertexPosition_modelspace,1);
	pos = vertexPosition_modelspace;
}

