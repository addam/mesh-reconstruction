#version 330 core

layout(location = 0) in vec3 vertexPosition_modelspace;

out vec3 pos;

uniform mat4 mainMVP;

void main(){
	gl_Position = mainMVP * vec4(vertexPosition_modelspace, 1);
	pos = vertexPosition_modelspace;
}

