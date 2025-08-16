from fun import fill_neighbors, generate_mask, get_cluster_at_point, get_superpixels_by_pixels, get_neighboring_superpixels, compute_normalized_histogram, get_unique_filename, graph_maker, helper, helper1, helper2, helper3, similarity_coefficient_calculator_and_value_returner, choose_method
import cv2
from fun import run_method_1, run_method_2
import matplotlib.pyplot as plt
import math
import copy
import networkx as nx
import time
import numpy as np
import sys
image_num = sys.argv[1]
# image_num = 153077
print(f"Processing {image_num}")

printing = False

with open('parameters_and_data.py', 'r') as f:
    code = f.read()

# Execute the code using exec()
exec(code, globals())


ground_img = cv2.imread(ground_truth_path)
plt.imshow(ground_img)
plt.show()
plt.close()

# from parameters_and_data import image_rgb, path_to_add, fpoints, bpoints, labels_slic, result, num_superpixels, image_path, seg_img_path, mask_img_path, num_superpixels_parameter
# from parameters_and_data import image_rgb, scribbling_dimension, fpoints, bpoints, visualization_image_path, scribbled_img_path, ground_truth_path
# from parameters_and_data import only_segmentation_path, segment_generation_time

# hyperparameters
# from parameters_and_data import num_superpixels_parameter, compactness
superpixel_centroids = []
# data
# with open('frontend_ui.py', 'r') as f:
#     code = f.read()

# # Execute the code using exec()
# exec(code)

# scribbles_from = 'one-cut'
# scribbles_from = 'amoe'
# scribbles_from = 'grabcut'
# scribbles_from = 'grabcut_large'
scribbles_from = 'our_markers_images250'

with open('frontend_ui.py', 'r') as f:
    code = f.read()

# Execute the code using exec()
exec(code, globals())
# exit()

image = image_rgb
# Define the parameters for SLIC
# You can adjust these parameters according to your requirements


'''

Algorithm starts here

Steps
1. Calculation of neighboring superpixels for each superpixel:
    - Required for graph creation.
    2 Ways:
        a. Store initially all the neighboring superpixel for each superpixel
        b. Dynamically calculate for each superpixel
    Analyse both the ways of graph creation

2. Calculation of Maximum Spanning Tree then partitioning it.


'''
# Generate the base filename using the image_num
base_name = str(image_num)
extension = "txt"

# Get a unique filename
counter = get_unique_filename(base_name, extension)

# user_choice = choose_method()
user_choice = 1

if (user_choice == 1):
    with open('analysis_1.py', 'r') as f:
        code = f.read()

    # Execute the code using exec()
    exec(code, globals())

    # code = ''
    # with open('analysis_compare.py', 'r') as f:
    #     code = f.read()

    # # Execute the code using exec()
    # exec(code)
elif (user_choice == 2):
    with open('analysis_2.py', 'r') as f:
        code = f.read()

    # Execute the code using exec()
    exec(code)
    # print('Random neighbor function call each time')
    # with open('analysis_3.py', 'r') as f:
    #     code = f.read()

    # # Execute the code using exec()
    # exec(code)
elif (user_choice == 3):
    with open('analysis_5.py', 'r') as f:
        code = f.read()

    # Execute the code using exec()
    exec(code)

    with open('analysis_1.py', 'r') as f:
        code = f.read()

    # Execute the code using exec()
    exec(code)
elif (user_choice == 4):
    with open('analysis_compare.py', 'r') as f:
        code = f.read()

    # Execute the code using exec()
    exec(code)
