'''
To get the graysclae depth images and convert them into RGB Depth images. 

Read all the converted RGB Depth images and extract the RGB channels from camera1, caera2, and camera3 respectively to form the final image. 
'''
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import os
import glob
import sys
from tqdm import tqdm

img_type = ".png"

print("Starting the script: ")

# grayscale depth images
path_original_img = r"C:\Users\hp\AI\Stanford\ARMLab\NeRF_By_Touch\Blender\ARMLab Code - Tejas\NeRF Data\Cube with ridges\data\remapping\original_depth\depth\train"
path_045_remapped = r"C:\Users\hp\AI\Stanford\ARMLab\NeRF_By_Touch\Blender\ARMLab Code - Tejas\NeRF Data\Cube with ridges\data\remapping\depth_0.45_remapped\depth_0.45_remapped\target_045_remapped_images"
path_05_remapped = r"C:\Users\hp\AI\Stanford\ARMLab\NeRF_By_Touch\Blender\ARMLab Code - Tejas\NeRF Data\Cube with ridges\data\remapping\depth_0.5_remapped\depth_0.5_remapped"

# rgb depth images
target_path_original_img = r"C:\Users\hp\AI\Stanford\ARMLab\NeRF_By_Touch\Blender\ARMLab Code - Tejas\NeRF Data\Cube with ridges\data\RGB_depth\RGB_depth_original"
target_045_remapped_path = r"C:\Users\hp\AI\Stanford\ARMLab\NeRF_By_Touch\Blender\ARMLab Code - Tejas\NeRF Data\Cube with ridges\data\RGB_depth\RGB_depth_0.45"
target_05_remapped_path = r"C:\Users\hp\AI\Stanford\ARMLab\NeRF_By_Touch\Blender\ARMLab Code - Tejas\NeRF Data\Cube with ridges\data\RGB_depth\RGB_depth_0.5"



filenames = os.listdir(path_05_remapped)

print("Printing filenames: ", filenames)

for img_name in tqdm(filenames):
    filepath = os.path.join(path_05_remapped, img_name)
    #print(filepath)
    #print("Exiting....")

    #sys.exit()

    # Load the grayscale depth image
    depth_gray = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)

    #print("Dimensions of deoth image: ", depth_gray.shape)

    #cv2.imshow("grayscale_image", depth_gray)
    #cv2.waitKey(1)

    #cv2.destroyAllWindows()

    # Convert the grayscale depth image to a color image
    #depth_color = cv2.applyColorMap(cv2.convertScaleAbs(depth_gray, alpha=0.03), cv2.COLORMAP_JET)
    depth_color = cv2.applyColorMap(depth_gray, cv2.COLORMAP_JET)

    #cv2.imshow('depth_color image', depth_color)
    #print("The dimnensions of the depth color are: ", depth_color.shape)
    #cv2.waitKey(1)

    # if cv2.waitKey(0) == ord("q"):
    #     cv2.destroyAllWindows()

    target_filepath = os.path.join(target_05_remapped_path, img_name)

    print(target_filepath)
    print()

    # Save the RGB depth image
    cv2.imwrite(target_filepath, depth_color)

    # print("Exiting the system....")
    # sys.exit()
