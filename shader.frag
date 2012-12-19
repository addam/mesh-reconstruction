#version 330 core

in vec3 pos;
out vec3 color;

uniform sampler2D myTextureSampler;
uniform mat4 sideMVP;

void main(){
	vec4 modelCoord = vec4(pos, 1);
	vec4 screenCoord = sideMVP * modelCoord;
	color = texture(myTextureSampler, screenCoord.xy / screenCoord.w).rgb;
}
