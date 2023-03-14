''''
To combine the RGB depth images taken from a camera from a single location but with pixel intensity remapping

And then combine the R, G, and B channels of camera1, camera2, and camera3 respectively to get the final
RGB depth image. 

Transform the final RGB depth image to a grayscle image
'''
import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import sys


far_plane = 0.75
min_depth_range = 0.1

# to take the paths of rgb depth images
target_path_original_img = r"C:\Users\hp\AI\Stanford\ARMLab\NeRF_By_Touch\Blender\ARMLab Code - Tejas\NeRF Data\Cube with ridges\RGB Depth Data\RGB_depth\RGB_depth_original"
target_045_remapped_path = r"C:\Users\hp\AI\Stanford\ARMLab\NeRF_By_Touch\Blender\ARMLab Code - Tejas\NeRF Data\Cube with ridges\RGB Depth Data\RGB_depth\RGB_depth_0.45"
target_05_remapped_path = r"C:\Users\hp\AI\Stanford\ARMLab\NeRF_By_Touch\Blender\ARMLab Code - Tejas\NeRF Data\Cube with ridges\RGB Depth Data\RGB_depth\RGB_depth_0.5"


final_depth_images_path = r"C:\Users\hp\AI\Stanford\ARMLab\NeRF_By_Touch\Blender\ARMLab Code - Tejas\NeRF Data\Cube with ridges\data\final_depth_images"

equal_weight_path = r"C:\Users\hp\AI\Stanford\ARMLab\NeRF_By_Touch\Blender\ARMLab Code - Tejas\NeRF Data\Cube with ridges\RGB Depth Data\final_depth_images\equal_weight_path"
more_weight_to_red_channel = r"C:\Users\hp\AI\Stanford\ARMLab\NeRF_By_Touch\Blender\ARMLab Code - Tejas\NeRF Data\Cube with ridges\RGB Depth Data\final_depth_images\more_weight_to_red_channel"
more_weight_to_blue_channel = r"C:\Users\hp\AI\Stanford\ARMLab\NeRF_By_Touch\Blender\ARMLab Code - Tejas\NeRF Data\Cube with ridges\RGB Depth Data\final_depth_images\more_weight_to_blue_channel"

# to list the filenames of each rgb depth image
org_img_list = os.listdir(target_path_original_img)
target_045_list = os.listdir(target_045_remapped_path)
target_05_list = os.listdir(target_05_remapped_path)


'''
to make the code more generalized
'''
camera1_path = target_path_original_img
camera2_path = target_045_remapped_path
camera3_path = target_05_remapped_path

camera1_list = org_img_list
camera2_list = target_045_list
camera3_list = target_05_list

print(len(org_img_list), len(target_045_list), len(target_05_list))


for i, j, k in zip(camera1_list, camera2_list, camera3_list):
    
    cam1_img_path = os.path.join(camera1_path, i)
    cam2_img_path = os.path.join(camera2_path, j)
    cam3_img_path = os.path.join(camera3_path, k)

    target_img_path= os.path.join(final_depth_images_path, i)

    # print("Paths:")
    # print(org_img_path)
    # print()
    # print(target_045_path)
    # print()
    # print(target_05_path)
    
    
    img1 = cv2.imread(cam1_img_path, -1)
    img2 = cv2.imread(cam2_img_path, -1)
    img3 = cv2.imread(cam3_img_path, -1)

    # cv2.imshow("Original Image: ", img1)
    # cv2.waitKey(0)
    # cv2.imshow("Image_045", img2)
    # cv2.waitKey(0)
    # cv2.imshow("Image_05", img3)
    # cv2.waitKey(0)

    # if cv2.waitKey(1) == ord("q"):
    #     cv2.destroyAllWindows()


    '''
    to extract the blue, green, and red channels from images taken from camera1,
    camera2, and camera3 respectively. 

    The red channel holds the most significant depth information and so, I am 
    extracting red channels from the 3rd camera as it is giving a good depth image
    '''
    red_channel = img3[:, :, 0]
    green_channel = img2[:, :, 1] * 256
    blue_channel = img1[:, :, 2] * 256 * 256

    cv2.imshow("Red channel: ", red_channel)
    path = os.path.join(more_weight_to_blue_channel, "r_0_red.png")
    cv2.imwrite(path, red_channel)
    print("Red channel saved....")
    cv2.waitKey(0)

    cv2.imshow("Green Channel: ", green_channel)
    path = os.path.join(more_weight_to_blue_channel, "r_0_green.png")
    cv2.imwrite(path, green_channel)
    print("Green channel saved....")
    cv2.waitKey(0)

    cv2.imshow("Blue Channel: ", blue_channel)
    path = os.path.join(more_weight_to_blue_channel, "r_0_blue.png")
    cv2.imwrite(path, blue_channel)
    print("Saved blue channel.....")
    cv2.waitKey(0)

    # to combine these channels and convert them into meters
    #depth = (red_channel + green_channel + blue_channel) * far_plane / (256 * 256 * 256 - 1)
    depth = (red_channel + green_channel + blue_channel).astype(float)

    depth[depth < min_depth_range] = 0.0

    # convert the depth values to meters using the camera parameters
    depth = (depth * far_plane) / (256 * 256 * 256 - 1)

    print("Max value: ", np.max(depth))
    print("Min value: ", np.min(depth))

    # to convert the values to a grayscale depth image
    depth_image = np.uint8(depth / np.max(depth) * 255)

    path = os.path.join(more_weight_to_blue_channel, "r_0_final.png")
    cv2.imwrite(path, depth_image)
    print("Saved the final depth image....")

    # to display the depth image
    cv2.imshow("Final Depth image: ", depth_image)
    cv2.waitKey(0)

    if cv2.waitKey(1) == ord("q"):
        cv2.destroyAllWindows()




    
    break





sys.exit()

