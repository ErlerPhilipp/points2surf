import bpy
from bpy import data as D
from bpy import context as C
from mathutils import *
from math import *

import blensor

evd_files = {evd_files}
obj_locations = {obj_locations}
obj_rotations = {obj_rotations}
scan_sigmas = {scan_sigmas}

# delete default mesh
bpy.ops.object.select_all(action="DESELECT")
bpy.data.objects["Cube"].select = True
bpy.ops.object.delete()

# load our mesh
file_loc = '{file_loc}'
imported_object = bpy.ops.import_mesh.ply(filepath=file_loc)
obj_object = bpy.context.selected_objects[0]
obj_object.rotation_mode = 'QUATERNION'

"""If the scanner is the default camera it can be accessed 
for example by bpy.data.objects["Camera"]"""
scanner = bpy.data.objects["Camera"]
scanner.rotation_mode = 'QUATERNION'
scanner.local_coordinates = False
scanner.location = Vector([0.0, 0.0, 0.0])

# Kinect settings
# https://github.com/mgschwan/blensor/blob/master/release/scripts/addons/blensor/kinect.py
# scanner.kinect_max_dist=6.0
# scanner.kinect_min_dist=0.7
# scanner.kinect_noise_mu=0.0  # default 0.0
# scanner.kinect_noise_sigma=0.0  # default 0.0
# scanner.kinect_xres=640
# scanner.kinect_yres=480
# scanner.kinect_flength=0.73
# scanner.kinect_enable_window=False  # experimental
# scanner.kinect_ref_dist=0.0
# scanner.kinect_ref_limit=0.01
# scanner.kinect_ref_slope=0.16
# scanner.kinect_noise_scale=0.25  # default 0.25
# scanner.kinect_noise_smooth=1.5  # default 1.5
# scanner.kinect_inlier_distance=0.05

for i in range(len(evd_files)):
    def do_scan(scanner, pcd_file_out):
        """Scan the scene with the Velodyne scanner and save it
        to the file "/tmp/scan.pcd"
        Note: The data will actually be saved to /tmp/scan00000.pcd
        and /tmp/scan_noisy00000.pcd
        """
        # blensor.blendodyne.scan_advanced(
        #     scanner,
        #     rotation_speed=10.0,
        #     simulation_fps=24,
        #     angle_resolution=0.1728,
        #     max_distance=120,
        #     evd_file=pcd_file_out,
        #     noise_mu=0.0,
        #     noise_sigma=0.03,
        #     start_angle=0.0,
        #     end_angle=360.0,
        #     evd_last_scan=True,
        #     add_blender_mesh=False,
        #     add_noisy_blender_mesh=False)

        # blensor.kinect.scan_advanced(
        #     scanner,
        #     evd_file=pcd_file_out,
        #     evd_last_scan=True
        #     )

        # TOF settings
        # https://github.com/mgschwan/blensor/blob/master/release/scripts/addons/blensor/tof.py
        # Blensor 1.0.18 RC 10 Windows has a bug in evd.py: https://github.com/mgschwan/blensor/issues/30
        blensor.tof.scan_advanced(
            scanner,
            evd_file=pcd_file_out,
            evd_last_scan=True,
            max_distance=10.0,
            add_blender_mesh=False,
            add_noisy_blender_mesh=False,
            tof_res_x=176,
            tof_res_y=144,
            lens_angle_w=43.6,
            lens_angle_h=34.6,
            flength=10.0,
            noise_mu=0.0,
            # noise_sigma=scanner_noise_sigma,  # default 0.0004
            noise_sigma=scan_sigmas[i],  # default 0.0004
            backfolding=False,
        )


    evd_file = evd_files[i]
    obj_object.location = Vector(obj_locations[i])
    obj_object.rotation_quaternion = Quaternion(obj_rotations[i])
    do_scan(scanner, evd_file)

bpy.ops.wm.quit_blender()