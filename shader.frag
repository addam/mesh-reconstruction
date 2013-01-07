#version 330 core

in vec3 pos;
out vec3 color;

uniform sampler2D textureSampler;
uniform mat4 sideMVP;

void main(){
	vec4 modelCoord = vec4(pos, 1);
	vec4 screenCoord = sideMVP * modelCoord;
	color = texture(textureSampler, (screenCoord.xy / (2*screenCoord.w)) - 0.5).rgb;
}
