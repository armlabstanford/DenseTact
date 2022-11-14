# Simulated DenseTact

This folder contains simulation of DenseTact in Blender. 


## Install
Before start, please install cv2 and numpy module in your pc and link the library properly. 


### Linux

Please add below codes on the blender script file to link the python packages with blender python.

```
import subprocess
import sys
import os
 
print(sys.exec_prefix)
sys.path.append('./.local/lib/python3.10/site-packages')

```
### Windows

```

```

## Description

### 1. cube_compositing.blend

This file contains the overall generation of touch and depth dataset. The cube stl file can be changed into random object such as stanford bunny stl file. 

Following parameters can be changed:

```
RESOLUTION = 512 # resolution of images - i.e. 512x512x1 for depth image
RESULTS_PATH = 'FINAL_CUBE_TOUCH_IMAGES' 
FORMAT = 'PNG'
obj_scaling_factor = 0.5 # scale object in case the stl file is too small/large
specified_offset_units = 0.01 # offset from vertices [m]
original_rot = Vector((0, 0, 0))
cut_off_index = 10 # the number of dataset
vgrp = 'Group'
camera_initial_loc = Vector((100, 0, 0))
render_engine = 'CYCLES'
render_device = 'GPU'
camera_type = 'PANO'
panorama_type = 'FISHEYE_EQUIDISTANT'
clip_start = 0.001 
clip_end = 0.025 # touch data is recorded from 0.001m to 0.025m(radius of DenseTact V2)
```

### 2. Code for ambient capture

Following two codes captures the position captured images. The arrow is inserted in the position of captured vertices so that the user gets the clear understanding of captured dataset. 

#### Text_arrow_cube_ambient_camera_images.blend

Code for placing a labeled arrow on a single vertex of object.

#### All_arrow_cube_ambient_camera_images.blend

Captures 6 images of the labeled arrow with the object from 6 ambient cameras.

### 3. Mask_python_images_pipeline.ipynb

This code contains a pipeline of converting the depth images into the fisheye lens format.