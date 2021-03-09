import simple_threshold
import otsu_threshold
import cv2
import matplotlib.pyplot as plt
import numpy as np
import sys
import math
import homography
import homography_svd
import custom_math
import itertools

import os
import re

# video_name = 'Tag0.mp4'
# video_name = 'Tag1.mp4'
# video_name = 'Tag2.mp4'
video_name = 'multipleTags.mp4'

video_name_no_extension = video_name[:-4]

# cartoon_mode = True
cartoon_mode = False

cube_mode = True
# cube_mode = False

if cube_mode == True:
    mode_string = 'cube'

if cartoon_mode == True:
    mode_string = 'cartoon'



image_folder = './' + video_name_no_extension + '_annotated_' + mode_string + '/'


images = [img for img in os.listdir(image_folder) if img.endswith(".png")]

images.sort(key=lambda f: int(re.sub('\D', '', f)))

print(images)

frame = cv2.imread(os.path.join(image_folder, images[0]))
height, width, layers = frame.shape

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'MP4V')

if not os.path.exists('./output_annotated_videos/'):
    os.makedirs('./output_annotated_videos/')

out = cv2.VideoWriter('./output_annotated_videos/'+'annotated ' + mode_string + ' ' + video_name, 0x7634706d , 30.0, (width,height))


for image in images:
    print(os.path.join(image_folder, image))
    out.write(cv2.imread(os.path.join(image_folder, image)))
