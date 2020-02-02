#version 320 es
precision mediump float;

in vec3 pos; // world space position of the fragment being processed (obtained from vertex buffer)
out vec3 color; // output RGB color of the fragment

uniform sampler2D textureSampler;
uniform sampler2D shadowSampler;
uniform mat4 sideMVP;

void main(){
	// world space homogeneous position
	vec4 modelCoord = vec4(pos, 1);
	// project the fragment's position from the side camera
	vec4 screenCoord = sideMVP * modelCoord;
	// check for visibility from the side camera
	float shadowDepth = texture(shadowSampler, (screenCoord.xy / (2.0*screenCoord.w)) - 0.5).r * 2.0 - 1.0;
	bool visible = shadowDepth + 0.01 > screenCoord.z/screenCoord.w;
	bool inframe = (screenCoord.x/screenCoord.w > -1.0 && screenCoord.x/screenCoord.w < 1.0 && screenCoord.y/screenCoord.w > -1.0 && screenCoord.y/screenCoord.w < 1.0);
	// disabled: check correctly faced normal
	//bool normals = gl_FrontFacing;
	color.r = texture(textureSampler, (screenCoord.xy / (2.0*screenCoord.w)) - 0.5).r;
	// mask out if not visible
	color = (visible && inframe) ? vec3(color.r, 1.0, 1.0) : vec3(0.0, 0.0, 0.0);
}
