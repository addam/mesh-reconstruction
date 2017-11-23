# Mesh Reconstruction
This program performs dense scene reconstruction given that sparse reconstruction data is already available.

The output format is a 3-dimensional triangle mesh.
The input is a RGB camera video sequence along with a special text file describing the spatial configuration of the scene.
It is expected that the user will create sparse reconstruction data in Blender and then use the supplied program `io_export_tracks.py` to save them in the appropriate format.

Most notable external dependencies are the CGAL library, OpenCV2 and OpenGL bindings provided by GLEW.
The code may be useful as an example of OpenGL3 off-screen rendering; the application needs a running X server for communication with the graphics card, but does not even create a window.

This code is my bachelor thesis; it is already finished, though not really usable. The text is available on my website: http://adam.dominec.eu/bachelor.pdf
