# code to convert optitrack poses stored in data folder
# into a transform json file for reading into torch-ngp
import argparse
import os
from pathlib import Path
import csv
import numpy as np
import cvxpy as cp 
import json
import re
import imageio
import cv2
from scipy.spatial.transform import Rotation as R
import scipy
#import procrustes
#from procrustes import generic
from scipy import linalg

# helper function that allows sorting of data files by names
def file_sort(name):
    return int(re.findall(r'\d+', name)[0])

# helper function that converts quaternion vectors into rotation matrices
def tf_from_vect(quaternions, translation):
    tf = np.zeros((4,4))

    tf[:3,:3] = R.from_quat(quaternions).as_matrix()
    tf[:3,3] = translation
    tf[3,3] = 1
    return tf

def create_calibration_tf(fname):

    with open(fname) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        tfs_apr_mocap = []
        tfs_cam_mocap  = []
        tfs_cam_april = []
        tfs_cam_mocap_tilda = []
        for row in csv_reader:
            if line_count != 0:
                count, time_sec, time_nano = row[0], row[1], row[2]
                tf_apr_mocap = row[3:10]
                tf_cam_mocap = row[10:17]
                tf_cam_april = row[24:]

                TF_apr_mocap = tf_from_vect(tf_apr_mocap[3:],tf_apr_mocap[:3])
                TF_cam_mocap = tf_from_vect(tf_cam_mocap[3:],tf_cam_mocap[:3])
                TF_cam_april = tf_from_vect(tf_cam_april[3:],tf_cam_mocap[:3])

                tf_cam_mocap_tilda =  TF_apr_mocap@np.linalg.inv(TF_cam_april)
                
                tfs_apr_mocap.append(TF_apr_mocap)
                tfs_cam_mocap.append(TF_cam_mocap)
                tfs_cam_april.append(TF_cam_april)
                tfs_cam_mocap_tilda.append(tf_cam_mocap_tilda)
                
            line_count += 1
            
        tfs_apr_mocap = np.asarray(tfs_apr_mocap)
        tfs_cam_mocap = np.asarray(tfs_cam_mocap)
        tfs_cam_april = np.asarray(tfs_cam_april)
        tfs_cam_mocap_tilda = np.asarray(tfs_cam_mocap_tilda)
        
        trans = np.mean(tfs_cam_mocap_tilda[:,:3,3] - tfs_cam_mocap[:,:3,3],axis=0)
        rot = []
        for i in range(tfs_cam_mocap.shape[0]):
            result = linalg.orthogonal_procrustes(tfs_cam_mocap[i,:3,:3], tfs_cam_mocap_tilda[i,:3,:3])[0]
            rot.append(result)
        rot = np.asarray(rot)
        rot = np.mean(rot,axis=0)
        
        return rot, trans

trans_t = lambda t : torch.Tensor([
    [1,0,0,0],
    [0,1,0,0],
    [0,0,1,t],
    [0,0,0,1]]).float()

# x axis
rot_phi = lambda phi : np.array([
    [1,0,0,0],
    [0,np.cos(phi),-np.sin(phi),0],
    [0,np.sin(phi), np.cos(phi),0],
    [0,0,0,1]])

# y axis
rot_theta = lambda th : np.array([
    [np.cos(th),0,-np.sin(th),0],
    [0,1,0,0],
    [np.sin(th),0, np.cos(th),0],
    [0,0,0,1]])

# z axis
rot_psi = lambda psi : np.array([
    [np.cos(psi),-np.sin(psi),0, 0], 
    [np.sin(psi), np.cos(psi),0, 0], 
    [0, 0, 1, 0], 
    [0,0,0,1]])

def construct_json(args):
    
    print(args)
    
    calib_rot, calib_trans = create_calibration_tf('tfs.csv')
    
    filenames = ['color_pose.csv','depth_pose.csv', 'touch_pose.csv']
    foldernames = ['color','depth','touch']
    offset = [np.array([0,0,0]),
              np.array([0,0,-0.045]),
              np.array([0,0,-.1]),
             ]
    
    CAM_FOV_X_depth = 42 * np.pi / 180
    CAM_FOV_Y_depth = 69 * np.pi / 180
    
    
    focal_x = .5 * 848 / np.tan(.5 * CAM_FOV_X_depth)
    focal_y =  .5 * 480 / np.tan(.5 * CAM_FOV_Y_depth)
    
    file_params = {"color": {"camera_angle_x": 1.1207701965835353,
                             "camera_angle_y": 0.6426270681462697,
                             "fl_x": 1529.9142901380997,
                             "fl_y": 1622.3630079448637,
                             "k1": 0.13397960123760702,
                             "k2": -0.2049359371734879,
                             "p1": 0.0017226206391092774,
                             "p2": -0.0012317764625963917,
                             "cx": 974.4870614226737,
                             "cy": 543.971061330722,
                             "h": 1080,
                             "w": 1920
                          },
                   "depth": {"camera_angle_x": CAM_FOV_X_depth,
                             "camera_angle_y": CAM_FOV_Y_depth,
                             "fl_x": focal_x,
                             "fl_y": focal_y,
                             "k1": 0,
                             "k2": 0,
                             "p1": 0,
                             "p2": 0,
                             "cx": 848/2,
                             "cy": 480/2,
                             "h": 480,
                             "w": 848
                            },
                   "touch": {"camera_angle_x": CAM_FOV_X_depth,
                             "camera_angle_y": CAM_FOV_Y_depth,
                             "fl_x": focal_x,
                             "fl_y": focal_y,
                             "k1": 0,
                             "k2": 0,
                             "p1": 0,
                             "p2": 0,
                             "cx": 570/2,
                             "cy": 570/2,
                             "h": 570,
                             "w": 570
                            }
                    }
    

    all_poses = []
    for i in range(len(filenames)):
        print(os.path.join(args.data_path, foldernames[i], filenames[i]))
        tf_fn = os.path.join(args.data_path, foldernames[i], filenames[i])
    
        fd = open(tf_fn)
        csvreader = csv.reader(fd)
        poses = np.genfromtxt(tf_fn, delimiter=',')
        
        img_dir = os.path.join(args.data_path, foldernames[i], 'images')
        img_names = []
        for path, dirs, files in os.walk(img_dir):
            img_names = files
            break
        img_names.sort(key=file_sort)
        imgs = [imageio.imread(os.path.join(img_dir,img_name)) for img_name in img_names]
        
        tfs = []
        for j in range(poses.shape[0]):
            pose_row = poses[j,:]
            indx = pose_row[0]
            time = pose_row[1]
     
            translation, quaternion = pose_row[2:5], pose_row[5:]
            tf = tf_from_vect(quaternion, translation)
            
            tf[:3,3] = tf[:3,3] + offset[i]
            tf[:3,:3] = calib_rot@tf[:3,:3]
            tf[:3,3] = tf[:3,3] + calib_trans
            tf = tf @ rot_phi(-np.pi/2)
            tf = tf @ rot_psi(np.pi)
            u, s, v = np.linalg.svd(tf[:3,:3])
            R = u@np.eye(3)@v
            tf[:3,:3] = R
            tfs.append(tf)
        
        
        tfs = np.array(tfs)
        
        mean = np.mean(tfs[:,:3,3],axis=0)
        for j in range(tfs.shape[0]):
             tfs[j,:3,3] = tfs[j,:3,3] - mean
                
#         print(np.max(tfs[:,3,2]))
#         print(np.min(tfs[:,3,2]))  
        shift = (np.max(tfs[:,:3,3],axis=0) + np.min(tfs[:,:3,3],axis=0))/2
        for j in range(tfs.shape[0]):
            tfs[j,:3,3] = tfs[j,:3,3] - shift
            
        shift_z = - np.min(tfs[:,2,3],axis=0) #np.max(tfs[:,2,3],axis=0) - np.min(tfs[:,2,3],axis=0)
        print("z-shift")
        print(shift_z)
        print(tfs[:,2,3])
        #print(tfs3,3])
        for j in range(tfs.shape[0]):
            tfs[j,2,3] = tfs[j,2,3] + shift_z
            
        print("")
        print("result")
        print(tfs[:,2,3])
        print(np.max(tfs[:,2,3]))
        print(np.min(tfs[:,2,3])) 
        print(tfs[0,:,:])
#         stop
        frames = []
        for j in range(tfs.shape[0]):
            frame = {}
            frame['file_path'] = os.path.join(args.data_path, foldernames[i], 'images', img_names[j])
            sharpness = cv2.Laplacian(cv2.imread(os.path.join(img_dir,img_names[i])), cv2.CV_64F).var()
            frame['sharpness'] = sharpness
            tf_list = tfs[j].tolist() 
            frame['transform_matrix'] = tf_list #tfs[i].tolist()
            frames.append(frame)
    
        file_params[foldernames[i]]['frames'] = frames
        json_object = json.dumps(file_params[foldernames[i]], indent = 2)
    
        destination = os.path.join(args.data_path, foldernames[i], 'transforms.json')
        with open(destination, "w") as outfile:
            outfile.write(json_object)
       
        all_poses.append(tfs)
    return all_poses
        
    #if args.data_path is None:
    #    raise ValueError('Missing data folder path')
    #if args.image_folder is None:
    #    raise ValueError('Missing image folder path')
    #if args.file_name is None:
    #    raise ValueError('Missing csv file name')
    #if args.destination is None or args.destination == '':
    #    args.destination = os.path.join(args.data_path,'transform_' + Path(args.image_folder).stem + '.json')
    #
    #print(args.data_path)
    #print(args.image_folder)
    #print(args.image_type)
    #print(args.file_name)
    #print(args.destination)
    #print("done!")
    #
    ##prepare json file:
    #json_arguments = {}
    #if os.path.exists(args.destination):
    #    os.remove(args.destination)
    #
    #
    # read in specified csv file
    #tf_fn = os.path.join(args.data_path, args.file_name)
    #print("specified csv file path is: " + tf_fn)
    #fd = open(tf_fn)
    #csvreader = csv.reader(fd)
    #poses = np.genfromtxt(tf_fn, delimiter=',')
    #
    ## read in specified images:
    #img_dir = os.path.join(args.data_path, args.image_folder)
    #img_names = []
    #for path, dirs, files in os.walk(img_dir):
    #    img_names = files
    #    break
    #
    # sort the file names by number
    #img_names.sort(key=file_sort)
    #
    #imgs = [imageio.imread(os.path.join(img_dir,img_name)) for img_name in img_names]
    #
    #get image width, height, and channel depth
    #if args.image_type == 'color':
    #    h,w,c = imgs[0].shape
    #    print(h,w,c)
    #elif args.image_type == 'depth' or args.image_type == 'touch' :
    #    h,w = imgs[0].shape
    #    print(imgs[0].shape)
    #    print(h,w)
    #else:
    #    print('incorrect image type given!')
    #
    #fov_x = float(args.fov_x)
    #
    #if args.fov_y is None or args.fov_y == '':
    #    fov_y = fov_x
    #else:
    #    fov_y = float(args.fov_y)
    #
    # convert fov into radians 
    #print("fovs: ", fov_x, fov_y)
    #CAM_FOV_X = fov_x * np.pi / 180
    #CAM_FOV_Y = fov_y * np.pi / 180
    #
    #json_arguments['camera_angle_x'] = 1.1207701965835353#CAM_FOV_X#1.1151129319802389 #CAM_FOV_X
    #json_arguments['camera_angle_y'] = 0.6426270681462697#CAM_FOV_Y#0.6383814810333517 #CAM_FOV_Y


    # Calculate focal length
    #focal_x = .5 * w / np.tan(.5 * CAM_FOV_X)
    #focal_y = .5 * h / np.tan(.5 * CAM_FOV_Y)

    #json_arguments['fl_x'] = 1529.9142901380997#1.53304323e+03 #1539.57#focal_x
    #json_arguments['fl_y'] = 1622.3630079448637#1.62812148e+03 #1633.93#focal_y

    #CAM_FOV_X = 2*np.atan(w /(2 * focal_x))
    #CAM_FOV_Y = 2*np.atan(w /(2 * focal_x))

    #TODO: figure out what k1, k2, p1, and p2 are actually used for.
    # for now they appear unused so we set them to zero
    #json_arguments['k1'] = 0.13397960123760702#9.70298223e-02#0.134917#0.0
    #json_arguments['k2'] = -0.2049359371734879#2.07592239e-01#-0.218717#0.0
    #json_arguments['p1'] = 0.0017226206391092774#5.21593309e-04#-2.66001e-05#0.0
    #json_arguments['p2'] = -0.0012317764625963917#1.20699410e-03#-0.00224344#0.0
    #
    # set camera image offset and dimensions
    #json_arguments['cx'] = 974.4870614226737#9.89593033e+02#977.638#w/2
    #json_arguments['cy'] = 543.971061330722#5.51238471e+02#544.775#h/2
    #json_arguments['w'] = 1920.0#w
    #json_arguments['h'] = 1080.0#h
    #
    #
    #json_arguments['aabb_scale'] = 16#1#16#1
    #
    #tfs = []
    #for i in range(poses.shape[0]):
    #    pose_row = poses[i,:]
    #    indx = pose_row[0]
    #    time = pose_row[1]
    #    
    #    translation, quaternion = pose_row[2:5], pose_row[5:]
    #    tf = tf_from_vect(quaternion, translation)
    #
    #    tf = tf @ rot_phi(-np.pi/2)
    #    tf = tf @ rot_psi(np.pi)
    #    u, s, v = np.linalg.svd(tf[:3,:3])
    #    R = u@np.eye(3)@v
    #    tf[:3,:3] = R
    #    tfs.append(tf)
    #
    #
    #tfs = np.array(tfs)
    #
    # center the camera poses such that the scene center is on the origin
    #centroid = np.mean(tfs[:,:3,3],axis=0)
    #tfs[:,:3,3] = tfs[:,:3,3] - centroid
    #tfs[:,:3,3] = tfs[:,:3,3] - (np.max(tfs[:,:3,3],axis=0) + np.min(tfs[:,:3,3], axis=0))/2
    #
    #frames = []
    #for i in range(tfs.shape[0]):
    #    frame = {}
    #    frame['file_path'] = os.path.join(args.image_folder,img_names[i])
    #    sharpness = cv2.Laplacian(cv2.imread(os.path.join(img_dir,img_names[i])), cv2.CV_64F).var()
    #    frame['sharpness'] = sharpness
    #    tf_list = tfs[i].tolist() 
    #    frame['transform_matrix'] = tf_list #tfs[i].tolist()
    #    frames.append(frame)
    #
    #json_arguments['frames'] = frames
    #json_object = json.dumps(json_arguments, indent = 2)
    #
    #with open(args.destination, "w") as outfile:
    #    outfile.write(json_object)
    #return tfs


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert optitrack poses into Nerf Compatible Json file')
    parser.add_argument('--data_path', metavar='path', required=True,
                        help='Path to data folder')
    #parser.add_argument('--image_folder', metavar='path', required=True,
    #                    help='Path to corresponding image folder')
    #parser.add_argument('--image_type', required=True,
    #                    help='what is the image type')
    # 
    #parser.add_argument('--file_name', required=True,
    #                    help='name of file to convert')
    #parser.add_argument('--destination', metavar='path', required=False,
    #                    help='destination to place the created file')
    #parser.add_argument('--fov_x', required=True,
    #                    help='camera field of view in x direction (assumed to be in degrees)')
    #parser.add_argument('--fov_y', required=False,
    #                    help='camera field of view in x direction (assumed to be in degrees)')

    args = parser.parse_args() 
    construct_json(args)
    
