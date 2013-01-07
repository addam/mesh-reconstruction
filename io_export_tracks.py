import bpy, mathutils
from bpy_extras.io_utils import ExportHelper
from bpy.props import StringProperty, BoolProperty
from bpy.types import Operator
from itertools import chain

bl_info = {
    "name": "Export Tracks",
    "author": "Addam Dominec",
    "version": (0, 1),
    "blender": (2, 6, 5),
    "location": "File > Export",
    "description": "Exports camera calibration and tracked bundles from video clip",
    "warning": "",
    "wiki_url": "",
    "tracker_url": "",
    "category": "Import-Export"}


def PerspectiveMatrix(fovx, aspect, near=0.1, far=1000.0):
	return mathutils.Matrix([
		[-2/fovx, 0, 0, 0],
		[0, -2*aspect/fovx, 0, 0],
		[0, 0, (far+near)/(far-near), (2*far*near)/(far-near)],
		[0, 0, -1, 0]])

def write_tracks(context, filepath, include_hidden):
	f = open(filepath, 'w', encoding='utf-8')
	f.write("%YAML:1.0\n")
	
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
					path=bpy.path.abspath(clip.filepath),
					width=clip.size[0],
					height=clip.size[1],
					fov=fov,
					distortion=[tr.camera.k1, tr.camera.k2, tr.camera.k3],
					center_x=tr.camera.principal[0],
					center_y=tr.camera.principal[1]))
	f.write("camera:\n")
	pout = open("points.txt", "w+")
	for camera in tr.reconstruction.cameras:
		translation = camera.matrix.translation
		direction = camera.matrix * mathutils.Vector((0,0,-1,1))
		direction = direction.xyz/direction.w
		direction -= translation
		direction.normalize()
		distances = [(track.bundle - translation).dot(direction) for track in tr.tracks if include_hidden or not track.hide]
		near, far = min(d for d in distances if d > 0), max(distances)
		persp = PerspectiveMatrix(fovx=fov, aspect=clip.size[0]/clip.size[1], near=near, far=far)
		cam = persp*camera.matrix.inverted()
		pout.write("Frame {}, near {}, far {}:\n".format(camera.frame, near, far))
		for track in tr.tracks:
			bundle4 = track.bundle.to_4d()
			bundle4 = cam * bundle4
			pout.write(str(bundle4.xyz / bundle4.w))
			pout.write("\n")
		f.write(" - frame: {frame}\n"
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
	          frame=camera.frame,
	          projection=", ".join(str(val) for val in chain(*cam)),
	          position = ", ".join(str(val) for val in camera.matrix.translation.to_4d())
	         ))
			#camera.average_error, frame, matrix
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
			#tr.bundle, average_error, hide
	
	f.close()

	return {'FINISHED'}


class ExportTracks(Operator, ExportHelper):
	'''Export camera calibration and tracked bundles from a movie clip'''
	bl_idname = "export_anim.tracks"  # important since its how bpy.ops.import_test.some_data is constructed
	bl_label = "Export Tracks"

	# ExportHelper mixin class uses this
	filename_ext = ".yaml"

	filter_glob = StringProperty(
		default="*.yaml",
		options={'HIDDEN'},
		)

	# List of operator properties, the attributes will be assigned
	# to the class instance from the operator settings before calling.
	include_hidden = BoolProperty(
		name="Include Hidden",
		description="Export both visible and hidden tracks",
		default=True,
		)

	def execute(self, context):
		return write_tracks(context, self.filepath, self.include_hidden)


# Only needed if you want to add into a dynamic menu
def menu_func(self, context):
	self.layout.operator(ExportTracks.bl_idname, text="Tracks (.yaml)")


def register():
	bpy.utils.register_class(ExportTracks)
	bpy.types.INFO_MT_file_export.append(menu_func)


def unregister():
	bpy.utils.unregister_class(ExportTracks)
	bpy.types.INFO_MT_file_export.remove(menu_func)


if __name__ == "__main__":
	register()
