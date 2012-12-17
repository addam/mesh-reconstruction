#!/usr/bin/awk -f
BEGIN {
	print "// Generated from shader sources automatically by pack_shaders.awk";
}
BEGINFILE  {
	ORS = "\\n";
	if (FILENAME ~ /\.vert$/) {
		print "const char* vertexShaderSources[] = {\"";
	} else if (FILENAME ~ /\.frag$/) {
		print "const char* fragmentShaderSources[] = {\"";
	}
}
{
	print;
}
ENDFILE {
	ORS = "\n"
	print "\"};";
}
