#version 330 core

in vec3 pos;
out vec3 color;

uniform sampler2D textureSampler;
uniform sampler2D shadowSampler;
uniform mat4 sideMVP;

void main(){
	vec4 modelCoord = vec4(pos, 1);
	vec4 screenCoord = sideMVP * modelCoord;
	float shadowDepth = texture(shadowSampler, (screenCoord.xy / (2*screenCoord.w)) - 0.5).r * 2 - 1;
	bool visible = shadowDepth + 0.01 > screenCoord.z/screenCoord.w;
	bool inframe = (screenCoord.x/screenCoord.w > -1 && screenCoord.x/screenCoord.w < 1 && screenCoord.y/screenCoord.w > -1 && screenCoord.y/screenCoord.w < 1);
	bool normals = gl_FrontFacing;
	color.r = texture(textureSampler, (screenCoord.xy / (2*screenCoord.w)) - 0.5).r;
	color = (visible && inframe && normals) ? vec3(color.r, 1, 1) : vec3(0, 0, 0);
}
