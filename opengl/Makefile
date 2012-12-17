glx: glx.cpp shaders.h
	g++ glx.cpp -lGL -lGLEW -o glx

shaders.h: pack_shaders.awk TransformVertexShader.vert TextureFragmentShader.frag
	awk -f pack_shaders.awk TransformVertexShader.vert TextureFragmentShader.frag > shaders.h
