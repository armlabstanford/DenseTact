import argparse, sys, os
import json
import bpy
from mathutils import Vector
import numpy as np
import math
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import gc
import random
import pprint
#import bmesh
import sys

# parameters and lists
i = 0
RESOLUTION = 1024
RESULTS_PATH = r'depth_0.25'
FORMAT = 'PNG'
cut_off_index = 10
render_engine = 'BLENDER_EEVEE'
#render_device = 'GPU'
camera_type = 'PERSPECTIVE'
#panorama_type = 'FISHEYE_EQUIDISTANT'
clip_start = 0.0001
clip_end = 0.5
camera_initial_loc = Vector((2,0,0))
offset_distance = -0.55
angle_x = 60


path_1 = r"E:\Stanford\Stanford COURSES\First Year\Quarter 2\Research Assistant\depth_camera_orientation.npy"
path_2 = r"E:\Stanford\Stanford COURSES\First Year\Quarter 2\Research Assistant\depth_camera_positions.npy"


depth_camera_orientation = np.load(path_1)
depth_camera_position = np.load(path_2)

#print(depth_camera_orientation)

# to add the camera
bpy.ops.object.camera_add(enter_editmode = False, location = camera_initial_loc)
camera = bpy.context.active_object

# to set the rendering settings
bpy.context.scene.render.engine = render_engine

bpy.context.scene.use_nodes = True
#bpy.context.scene.view_layers["ViewLayer"].use_pass_z = True

bpy.context.object.data.clip_start = clip_start
bpy.context.object.data.clip_end = clip_end


# to set the focal length and field of view
bpy.data.objects['Camera'].data.angle_x = np.deg2rad(angle_x)

#sys.exit()


# specify and return changed camera look direction
def look_at(obj_camera, point):
    looking_direction = obj_camera.location - Vector(point)
    rot_quat = looking_direction.to_track_quat('Z', 'Y')
    #camera_orientation.append(rot_quat.to_euler())
    obj_camera.rotation_euler = rot_quat.to_euler()
    #return rot_quat.to_euler()



# def camera_rotation(camera, rotation):
#     '''
#     to rotate the camera based on previous camera location
#     '''
#     camera.rotation_euler = rotation



'''
CODE TO GET THE COLOR AND THE DEPTH IMAGES
'''
fp = bpy.path.abspath(f"//{RESULTS_PATH}")


# this function stores the matrix_world values of any given object
def listify_matrix(matrix):
    matrix_list = []
    for row in matrix:
        matrix_list.append(list(row))
    return matrix_list


def get_depth_image(b_invert = False, save_fn = None):
    raw = np.asarray(bpy.data.images["Viewer Node"].pixels)
    scene = bpy.data.scenes['Scene']
    raw = np.reshape(raw, (scene.render.resolution_x, scene.render.resolution_y, 4))
    raw = raw[:, :, 0]
    raw = np.flipud(raw)
    # to get the min & max distance before the object is clipped by the camera in meters
    depth0 = bpy.data.objects['Camera'].data.clip_start
    depth1 = bpy.data.objects['Camera'].data.clip_end 
    # to assign raw image to the "depth" variable
    depth = raw
    # to normalize in range 0 to 255
    img8 = (raw - depth0)/(depth1-depth0)*255
    img8 = img8.astype(np.uint8)
    # if we want to get the disparity image then use this condition
    if b_invert:
        depth = 1.0 / depth
    if save_fn is not None:
        if save_fn[-3:] == "npy":
            np.save(save_fn, depth)
            pth = save_fn[:-4]
            cv2.imwrite(pth+'.png', img8)
        else:
            cv2.imwrite(save_fn, depth)


# to make a directory to save the rendered color and depth images
if not os.path.exists(fp):
    os.makedirs(fp)
    
if not os.path.exists(os.path.join(fp,RESULTS_PATH)):
    os.makedirs(os.path.join(fp,RESULTS_PATH))


# Data to store in JSON file
out_data = {
    'camera_angle_x': bpy.data.objects['Camera'].data.angle_x,
}

# Data to store in JSON file
depth_out_data = {
    'camera_angle_x': bpy.data.objects['Camera'].data.angle_x,
}


# Background
bpy.context.scene.render.dither_intensity = 0.0
bpy.context.scene.render.film_transparent = True
    
# to set the scene resoltuion
scene = bpy.context.scene
scene.render.resolution_x = RESOLUTION
scene.render.resolution_y = RESOLUTION
scene.render.resolution_percentage = 100

scene.render.image_settings.file_format = FORMAT  # set output format to .png

out_data['frames'] = []
depth_out_data['frames'] = []

scene.render.image_settings.file_format = FORMAT  # set output format to .png
#scene.camera = context.object

print("Starting the script......")

look_at_pt = (0., 0., 0.)
bpy.context.scene.camera = camera
for i, pos, orient in zip(range(0, len(depth_camera_orientation)), depth_camera_position[0], depth_camera_orientation):
    camera.location = pos
    look_at(camera, look_at_pt)
    #camera_rotation(camera, orient)
    location = camera.location.copy()
    distance = location.length
    new_distance = distance + offset_distance
    direction = location.normalized()
    new_location = direction * new_distance
    camera.location = new_location

    scene.render.filepath = fp + '/r_' + str(i)
    bpy.ops.render.render(write_still=True)
    print("New image taken....")
    # to get the depth image
    get_depth_image(b_invert=False, save_fn=os.path.join(fp,RESULTS_PATH, 'r_' + str(i)+'.npy'))
    # to dump the matrix_world of the camera for getting the color and depth images
    frame_data = {
        'camera': "camera_1",
        'file_path': f"./{RESULTS_PATH}/r_{i}",
        'rotation': 0.1,
        'transform_matrix': listify_matrix(camera.matrix_world)
    }
    depth_frame_data = {
        'camera': 'camera_1',
        'file_path': f"./{RESULTS_PATH}/r_{i}",
        'rotation': 0.1,
        'transform_matrix': listify_matrix(camera.matrix_world)
    }
    out_data['frames'].append(frame_data)
    depth_out_data['frames'].append(depth_frame_data)


with open(fp + '/' + f'transforms_vert_{RESULTS_PATH}.json', 'w') as out_file:
    json.dump(out_data, out_file, indent=4)

with open(os.path.join(fp,RESULTS_PATH, f'transforms_vert_{RESULTS_PATH}.json'), 'w') as out_file:
    json.dump(depth_out_data, out_file, indent=4)





