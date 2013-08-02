import bpy, mathutils
from bpy_extras.io_utils import ExportHelper
from bpy.props import StringProperty, BoolProperty
from bpy.types import Operator
from itertools import chain
from os.path import dirname

bl_info = {
    "name": "Export Tracks",
    "author": "Addam Dominec",
    "version": (0, 2),
    "blender": (2, 6, 8),
		"api": 58757,
    "location": "File > Export",
    "description": "Exports camera calibration and tracked bundles from video clip",
    "warning": "",
    "wiki_url": "",
    "tracker_url": "",
    "category": "Import-Export"}


def PerspectiveMatrix(fovx, aspect, near=0.1, far=1000.0):
	"""Get internal camera matrix"""
	return mathutils.Matrix([
		[2/fovx, 0, 0, 0],
		[0, 2*aspect/fovx, 0, 0],
		[0, 0, (far+near)/(far-near), (2*far*near)/(near-far)],
		[0, 0, 1, 0]])


def write_tracks(context, filepath, include_hidden):
	"""Main function, writes all data to the given filepath"""
	f = open(filepath, 'w', encoding='utf-8')
	f.write("%YAML:1.0\n")
	
	# general info about clip
	clip = context.scene.active_clip
	tr = clip.tracking
	fov = tr.camera.sensor_width/tr.camera.focal_length
	f.write("clip:\n"
					" path: {path}\n"
					" width: {width}\n"
					" height: {height}\n"
					" fov: {fov}\n"
					" distortion: {distortion}\n"
					" center-x: {center_x}\n"
					" center-y: {center_y}\n".format(
					path=bpy.path.relpath(clip.filepath, start=dirname(filepath))[2:],
					width=clip.size[0],
					height=clip.size[1],
					fov=fov,
					distortion=[tr.camera.k1, tr.camera.k2, tr.camera.k3],
					center_x=tr.camera.principal[0],
					center_y=tr.camera.principal[1]))
	
	# info about each frame's camera
	f.write("camera:\n")
	# Blender uses a different convention than the config file to be written
	flip = mathutils.Matrix(((1,0,0,0), (0,1,0,0), (0,0,-1,0), (0,0,0,1)));
	for camera in tr.reconstruction.cameras:
		cammat = camera.matrix * flip
		cam_inv = cammat.inverted()
		distances = [(cam_inv * track.bundle.to_4d()).zw for track in tr.tracks if include_hidden or not track.hide]
		# guess near and far value based on distances to the tracked points
		near, far = 0.8*min(z/w for z,w in distances if z/w > 0), 2*max(z/w for z,w in distances)
		persp = PerspectiveMatrix(fovx=fov, aspect=clip.size[0]/clip.size[1], near=near, far=far)
		f.write(" - frame: {frame}\n"
						"   near: {near}\n"
						"   far: {far}\n"
						"   projection: !!opencv-matrix\n"
		        "    rows: 4\n"
	          "    cols: 4\n"
	          "    dt: f\n"
	          "    data: [ {projection}]\n"
	          "   position: !!opencv-matrix\n"
	          "    rows: 4\n"
	          "    cols: 1\n"
	          "    dt: f\n"
	          "    data: [ {position}]\n".format(
	          frame=camera.frame, near=near, far=far,
	          projection=", ".join(str(val) for val in chain(*(persp * cam_inv))),
		        position = ", ".join(str(val) for val in cammat.translation.to_4d())
	         ))
	
	# info about each track
	f.write("tracks:\n")
	for track in tr.tracks:
		if include_hidden or not track.hide:
			f.write(" - bundle: !!opencv-matrix\n"
			        "    rows: 4\n"
			        "    cols: 1\n"
			        "    dt: f\n"
			        "    data: [ {data}]\n"
			        "   frames-enabled: [{frames}]\n".format(
			        data=", ".join(str(s) for s in track.bundle.to_4d()),
			        frames=", ".join(str(marker.frame) for marker in track.markers if not marker.mute)))
	
	f.close()

	return {'FINISHED'}


class ExportTracks(Operator, ExportHelper):
	'''Export camera calibration and tracked bundles from a movie clip'''
	bl_idname = "export_anim.tracks"
	bl_label = "Export Tracks"

	filename_ext = ".yaml"

	filter_glob = StringProperty(
		default="*.yaml",
		options={'HIDDEN'},
		)

	include_hidden = BoolProperty(
		name="Include Hidden",
		description="Export both visible and hidden tracks",
		default=True,
		)

	def execute(self, context):
		return write_tracks(context, self.filepath, self.include_hidden)


def menu_func(self, context):
	self.layout.operator(ExportTracks.bl_idname, text="Tracks (.yaml)")


def register():
	bpy.utils.register_class(ExportTracks)
	bpy.types.INFO_MT_file_export.append(menu_func)


def unregister():
	bpy.utils.unregister_class(ExportTracks)
	bpy.types.INFO_MT_file_export.remove(menu_func)


if __name__ == "__main__":
	# if run from the text editor, register
	register()
