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
import bmesh
import sys

#camera = 'Camera'  #'Camera.001' # 'Camera'
#VIEWS = 400
##NUM = 800
#RESOLUTION_X = 600#1024 #800
#RESOLUTION_Y = 480#1024 #600
fib_radius = 0.85
RESULTS_PATH = 'remapping'
CAMERA_TYPE = ['color', 'depth', 'touch']
TRAIN_TYPE = ['train']#, 'test']#, 'val']
FORMAT = 'PNG'
specified_offset_units = 0.0

color_points = []
depth_points = []
touch_points = []
#camera_orientation = []
camera_orientation_color = []
camera_orientation_depth = []
camera_orientation_touch = []

#radius = .115 # remember the box is centered on origin and is with halflength of 1m
#zcutoff = None
fp = bpy.path.abspath(f"//{RESULTS_PATH}")
bpy.context.scene.render.use_lock_interface = True

              
# generate camera origin viewpoints:
def gen_viewpoints(camera):
    # print("radius used:")
    # print(camera["radius"])
    if camera["pose_method"] == "fibonacci_sphere":
        points = fibonacci_sphere(r=camera["radius"], zcutoff=camera["zcutoff"], samples=camera["num_views"])
        return points


# generate camera positions around a sphere
def fibonacci_sphere(r=1., zcutoff=None, samples=1000):

    points = []
    phi = math.pi * (3. - math.sqrt(5.))  # golden angle in radians

    for i in range(samples):
        y = 1. - (i / float(samples - 1)) * 2.  # y goes from 1 to -1
        radius = math.sqrt(1. - y * y)  # radius at y

        theta = phi * i  # golden angle increment

        x = math.cos(theta) * radius
        z = math.sin(theta) * radius
        
        if zcutoff is not None:
            if r*z >= zcutoff:
                points.append((r*x, r*y, r*z))
        else:
            points.append((r*x, r*y, r*z))
    return points



# specify and return changed camera look direction
def look_at(obj_camera, point):
    looking_direction = obj_camera.location - Vector(point)
    rot_quat = looking_direction.to_track_quat('Z', 'Y')
    #camera_orientation.append(rot_quat.to_euler())
    #obj_camera.rotation_euler = rot_quat.to_euler()
    return rot_quat.to_euler()



# obtain corresponding depth image for the RGBD camera
def get_depth_image(cam_name=None, b_invert=False, save_fn=None):
    '''
    values 0 -> 255, 0 is furthest, 255 is closest
    assumes range map node maps from 0 to 1 values

    set b_invert to True if you want disparity image
    '''
    raw = np.asarray(bpy.data.images["Viewer Node"].pixels)
    scene = bpy.data.scenes['Scene']
    raw = np.reshape(raw, (scene.render.resolution_y, scene.render.resolution_x, 4))
    raw = raw[:, :, 0]
    raw = np.flipud(raw)

    depth0 = bpy.data.objects[cam_name].data.clip_start
    depth1 = bpy.data.objects[cam_name].data.clip_end
    
    raw = np.clip(raw,0,depth1)

#    print("check flip")
#    print("raw")
#    print(raw)
#    print(np.max(raw))
#    print(np.min(raw))
#    print(depth1)
#    print(depth0)
    depth = raw
    img8 = (raw - depth0)/(depth1-depth0)*255
    #img8 = (raw - 0.35*depth1) / (fib_radius - 0.35*depth1) * 255
    img8 = img8.astype(np.uint8)
    #print('shape of the image: ', img8.shape)
#    plt.imshow(img8)
#    plt.show()
#    print(img8)
#    print(np.max(img8))
#    print(np.min(img8))
#    #stop
#    print(np.max(img8))
#    print(np.min(img8))

    if b_invert:
        depth = 1.0 / depth

    if not save_fn is None:
        if save_fn[-3:] == "npy":
#            np.save(save_fn, depth)
            pth = save_fn[:-4]
            cv2.imwrite(pth+'.png', img8)
        else:
            cv2.imwrite(save_fn, depth)
            
            
    return depth

# convert camera transform into a list for the json file
def listify_matrix(matrix):
    matrix_list = []
    for row in matrix:
        matrix_list.append(list(row))
    return matrix_list



if __name__ == "__main__":

    if not os.path.exists(fp):
        os.makedirs(fp)
    
    # generate file structure to save data
    for ctype in CAMERA_TYPE:
        if not os.path.exists(os.path.join(fp, ctype)):
             os.makedirs(os.path.join(fp,ctype))
             for type in TRAIN_TYPE:
                os.makedirs(os.path.join(fp,ctype, type))

    # generate N cameras with desired intrinsics
    # Possible Camera Pose Generation Methods: fibonacci_sphere, cube_look, cone_look, local_look
    cameras = [#{"name": "camera_1", "fl":50, "near":1e-4,
#                "far":0.025, "res_x": 600, "res_y": 480,
#                "radius": .115, "zcutoff": None, "num_views": 400,
#                "type":['color', 'depth'], "pose_method": "local_look",
#                "circle_r": .03, "phi": np.pi/14, "theta": np.pi/16,
#                "psi": 0,
#               },
#               {"name": "camera_1", "angle_x":160, "near":1e-4,
#                "far":0.015, "res_x": 600, "res_y": 600,
#                "radius": .115, "zcutoff": None, "num_views": 300,
#                "type":['color', 'depth'], "pose_method": "local_look",
#                "circle_r": .03, "phi": np.pi/14, "theta": np.pi/16,
#                "psi": 0,
#               },
            #    {"name": "camera_1", "angle_x":180, "fl_x":25, "near":1e-4,
            #     "far":0.025, "res_x": 1024, "res_y": 1024,
            #     "radius": .118, "zcutoff": None, "num_views": 200,
            #     "type":['touch'], "pose_method": "local_look",
            #     "circle_r": .03, "phi": np.pi/14, "theta": np.pi/16,
            #     "psi": 0,
            #    },

               {"name": "camera_2", "angle_x":60, "near":.1,
                "far":1, "res_x": 1024, "res_y": 1024,
                "radius": .8, "zcutoff": None, "num_views": 10,
                "type":['depth'], "pose_method": "fibonacci_sphere",
               },
#
#               {"name": "camera_3", "angle_x":70, "near":1e-4,
#                "far":0.5, "res_x": 1024, "res_y": 1024,
#                "radius": .25, "zcutoff": None, "num_views": 300,
#                "type":['depth'], "pose_method": "fibonacci_sphere",
#               },
               
#               {"name": "camera_4", "angle_x":70, "near":1e-4,
#                "far":1, "res_x": 1024, "res_y": 1024,
#                "radius": .5, "zcutoff": None, "num_views": 300,
#                "type":['depth'], "pose_method": "fibonacci_sphere",
#               },
               
#               {"name": "camera_5", "angle_x":70, "near":1e-4,
#                "far":2, "res_x": 1024, "res_y": 1024,
#                "radius": 1, "zcutoff": None, "num_views": 50,
#                "type":['color', 'depth'], "pose_method": "fibonacci_sphere",
#               },
#
#               {"name": "camera_6", "angle_x":70, "near":1e-4,
#                "far":2, "res_x": 1024, "res_y": 1024,
#                "radius": 1.5, "zcutoff": None, "num_views": 50,
#                "type":['color', 'depth'], "pose_method": "fibonacci_sphere",
#               },

               {"name": "camera_7", "angle_x":30, "near":1e-4,
                "far":2, "res_x": 1024, "res_y": 1024,
                "radius": .4, "zcutoff": None, "num_views": 10,
                "type":['color'], "pose_method": "fibonacci_sphere",
               },
              ]


# Specify Camera look point
look_at_pt = (0., 0., 0.)

# create camera objects
for camera in cameras:
    camera_data = bpy.data.cameras.new(name=camera["name"])
    camera_object = bpy.data.objects.new(camera["name"], camera_data)
    bpy.context.scene.collection.objects.link(camera_object)
    bpy.data.objects[camera["name"]].select_set(True)
    
    if 'touch' in camera['type']:
        bpy.data.objects[camera["name"]].data.type = 'PANO'
        bpy.data.objects[camera["name"]].data.cycles.fisheye_lens = camera["fl_x"]
        bpy.data.objects[camera["name"]].data.cycles.fisheye_fov = np.deg2rad(camera["angle_x"])
    else:
        bpy.data.objects[camera["name"]].data.angle_x = np.deg2rad(camera["angle_x"])

    print("HEY")
    print("camera type: ", camera_object.data.type)

    
    bpy.data.objects[camera["name"]].data.clip_start = camera["near"]
    bpy.data.objects[camera["name"]].data.clip_end = camera["far"]
    bpy.data.objects[camera["name"]].select_set(False)

# Background
bpy.context.scene.render.dither_intensity = 0.0
bpy.context.scene.render.film_transparent = True

# Create collection for objects not to render with background
objs = [ob for ob in bpy.context.scene.objects if ob.type in ('EMPTY') and 'Empty' in ob.name]
bpy.ops.object.delete({"selected_objects": objs})

# get blender scene
scene = bpy.context.scene
out_data = {}

if len(cameras) == 1:
    camera = cameras[0]
    for ctype in CAMERA_TYPE:
        out_data[ctype] = {}
        for type in TRAIN_TYPE:
            if ctype in camera["type"]:
                out_data[ctype][type] = {}
                out_data[ctype][type]["frames"] = []
                if bpy.data.objects[camera["name"]].data.type == "PANO":
                    out_data[ctype][type]["camera_angle_x"] = np.deg2rad(camera["angle_x"]).item()
#                        out_data[ctype][type]["fl_x"] = camera["fl_x"]
                    
                else:
                    out_data[ctype][type]["camera_angle_x"] =  bpy.data.objects[camera["name"]].data.angle_x, #2*math.atan(camera["res_x"] /(2*bpy.data.objects[camera["name"]].data.lens))
                #out_data[ctype][type]["fl_x"] = bpy.data.objects[camera["name"]].data.lens
                out_data[ctype][type]["near"] = bpy.data.objects[camera["name"]].data.clip_start
                out_data[ctype][type]["far"] = bpy.data.objects[camera["name"]].data.clip_end
                out_data[ctype][type]["types"] = camera["type"]
                out_data[ctype][type]["points"] = gen_viewpoints(camera)
                out_data[ctype][type]["x_resolutions"] = [camera["res_x"]]*len(out_data[ctype][type]["points"])
                out_data[ctype][type]["y_resolutions"] = [camera["res_y"]]*len(out_data[ctype][type]["points"])
                out_data[ctype][type]["names"] = [camera["name"]]*len(out_data[ctype][type]["points"])
#                        out_data[ctype][type]["frames"] = []

else:
    # keep track which camera types should be used (color, depth, touch)
    for ctype in CAMERA_TYPE:
        out_data[ctype] = {}
        for type in TRAIN_TYPE:
            # keep track of data needed for each training mode (train, test, val)
            out_data[ctype][type] = {}
            out_data[ctype][type]["cameras"] = {}
            out_data[ctype][type]["cameras"]
            out_data[ctype][type]["points"] = []
            out_data[ctype][type]["x_resolutions"] = []
            out_data[ctype][type]["y_resolutions"] = []
            out_data[ctype][type]["names"] = []
            out_data[ctype][type]["frames"] = []
            
            #generate entries for each camera and store them.
            for camera in cameras:
                if ctype in camera["type"]:
                    cam = {
                            "w": camera["res_x"],
                            "h": camera["res_y"],
#                                "camera_angle_x": bpy.data.objects[camera["name"]].data.angle_x, #2*math.atan(camera["res_x"] /(2*bpy.data.objects[camera["name"]].data.lens)),
                            #"fl_x": bpy.data.objects[camera["name"]].data.lens,
                            "near": bpy.data.objects[camera["name"]].data.clip_start,
                            "far": bpy.data.objects[camera["name"]].data.clip_end,
                            }
                    if bpy.data.objects[camera["name"]].data.type == "PANO":
                        cam["camera_angle_x"] = np.deg2rad(camera["angle_x"]).item()
#                            cam["fl_x"] = camera["fl_x"]

                    else:
                        cam["camera_angle_x"] =  bpy.data.objects[camera["name"]].data.angle_x, #2*math.atan(camera["res_x"]
                        
                    out_data[ctype][type]["cameras"][camera["name"]] = cam
                    out_data[ctype][type]["cameras"][camera["name"]]['types'] = camera["type"]
                    cam_points = gen_viewpoints(camera)
                    if ctype == "color":
                        color_points.append(cam_points)
                    if ctype == "depth":
                        depth_points.append(cam_points)
                    if ctype == "touch":
                        touch_points.append(cam_points)
                    out_data[ctype][type]["points"] = out_data[ctype][type]["points"] + cam_points
                    out_data[ctype][type]["x_resolutions"] = out_data[ctype][type]["x_resolutions"] + [camera["res_x"]]*len(cam_points)
                    out_data[ctype][type]["y_resolutions"] = out_data[ctype][type]["y_resolutions"] + [camera["res_y"]]*len(cam_points)
                    out_data[ctype][type]["names"] = out_data[ctype][type]["names"] + [camera["name"]]*len(cam_points)
            
            # shuffle points so there isn't positional bias in the data
            inds = np.arange(0, len(out_data[ctype][type]["x_resolutions"]))
            inds = np.random.permutation(inds)
            out_data[ctype][type]["points"] = list(np.asarray(out_data[ctype][type]["points"])[inds])
            out_data[ctype][type]["x_resolutions"] = list(np.asarray(out_data[ctype][type]["x_resolutions"])[inds])
            out_data[ctype][type]["y_resolutions"] = list(np.asarray(out_data[ctype][type]["y_resolutions"])[inds])
            out_data[ctype][type]["names"] = list(np.asarray(out_data[ctype][type]["names"])[inds])


for ctype in CAMERA_TYPE:
    print("CAMERA TYPE: ", ctype)
#        print("OI OI OI")
    for type in TRAIN_TYPE:
        print("type")
#            print(out_data[ctype])
        if len(out_data[ctype][type]["points"]):
            points = out_data[ctype][type].pop("points")
            x_resolutions = out_data[ctype][type].pop("x_resolutions")
            y_resolutions = out_data[ctype][type].pop("y_resolutions")
            names = out_data[ctype][type].pop("names")
            
            for i, loc in enumerate(points):
                print("using camera: ", names[i])
                print("object type: ", bpy.data.objects[names[i]].data.type)

                scene.render.resolution_x = x_resolutions[i]
                scene.render.resolution_y = y_resolutions[i]
                scene.render.resolution_percentage = 100
                bpy.data.objects[names[i]].select_set(True)
                cam = scene.objects[names[i]]
                bpy.context.scene.camera = cam
                cam.location = loc
                camera_rot = look_at(cam, look_at_pt)
                cam.rotation_euler = camera_rot
                if ctype == "color":
                    camera_orientation_color.append(camera_rot)
                if ctype == "depth":
                    camera_orientation_depth.append(camera_rot)
                if ctype == "touch":
                    camera_orientation_touch.append(camera_rot)
                #camera_orientation.append(camera_rot)
                
                
                
                if bpy.data.objects[names[i]].data.type == "PANO":
                    print("PANORAMIC")
                    bpy.context.scene.render.engine = 'CYCLES'
                    print("focal length: ", bpy.data.objects[names[i]].data.cycles.fisheye_lens)
                    print("field of view:", bpy.data.objects[names[i]].data.cycles.fisheye_fov)
                else:
                    print("PRESPECTIVE")
                    bpy.context.scene.render.engine = 'BLENDER_EEVEE'
                    print("focal length: ", bpy.data.objects[names[i]].data.lens)
                    print("field of view:", bpy.data.objects[names[i]].data.angle_x)
                
                if ctype == 'color':
                    scene.render.filepath = os.path.join(fp, ctype, type, 'r_' + str(i)) #fp + '/r_' + str(i)
                    bpy.ops.render.render(write_still=True)  # render still
                elif ctype == 'depth' or 'touch':
                    scene.render.filepath = os.path.join(fp, ctype, type, 'r_' + str(i))
                    bpy.ops.render.render()
                    get_depth_image(cam_name=names[i], b_invert=False, save_fn=os.path.join(fp, ctype, type, 'r_' + str(i)+'.npy'))
            
            
                frame_data = {
                                'camera': names[i],
                                'file_path': f"./train/r_{i}",
                                'rotation': 0.1,
                                'transform_matrix': listify_matrix(cam.matrix_world)
                }
                out_data[ctype][type]['frames'].append(frame_data)
                gc.collect()
            

            
            with open(os.path.join(fp, ctype, f'transforms_{type}.json'), 'w') as out_file:
                json.dump(out_data[ctype][type], out_file, indent=4)
        print(" ")




path_1 = r"E:\Stanford\Stanford COURSES\First Year\Quarter 2\Research Assistant\depth_camera_orientation"
path_2 = r"E:\Stanford\Stanford COURSES\First Year\Quarter 2\Research Assistant\depth_camera_positions"
print("Length of camera orientation: ", len(camera_orientation_depth))
print("Length of camera positions: ", len(depth_points))

print()

# print(camera_orientation_depth)
# print(depth_points)

print("saving the numpy arrays......")

# to save the depth camera orientation and the camera locations
np.save(path_1, camera_orientation_depth)
np.save(path_2, depth_points)

print()


# Delete all cameras
bpy.ops.object.select_all(action='DESELECT')
for camera in cameras:
    if bpy.context.scene.objects.get(camera["name"]):
        bpy.data.objects[camera["name"]].select_set(True)
        bpy.ops.object.delete()