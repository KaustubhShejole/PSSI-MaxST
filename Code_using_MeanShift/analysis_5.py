
import json
import time
from parameters_and_data import image_rgb, fpoints, bpoints, labels_slic, result, num_superpixels, image_path, seg_img_path, mask_img_path, num_superpixels_parameter
from parameters_and_data import image_rgb, scribbling_dimension, fpoints, bpoints, visualization_image_path, scribbled_img_path, ground_truth_path, image_num
import networkx as nx
import copy
import math
import matplotlib.pyplot as plt

import cv2
import numpy as np
from fun import ask_to_continue, ask_to_detailed_continue, compute_normalized_histogram4, double_fill_neighbors_6, fill_neighbors, fill_neighbors_5, fill_neighbors_6, fill_neighbors_random_8, find_critical_edges, generate_mask, generate_mask1, get_cluster_at_point, get_superpixels_by_pixels, get_neighboring_superpixels, compute_normalized_histogram, fill_neighbors_4, graph_maker, helper, helper1, helper2, helper3, helper31, helper5, helper6, max_spanning_tree_with_cut, precompute_superpixel_indices, similarity_coefficient_calculator_and_value_returner, similarity_coefficient_calculator_and_value_returner4, abc

image = image_rgb
superpixel_centroids = []


initial_method1_starting_time = time.time()


# Start time
start_time1 = time.time()
start_time_s_i = time.time()
superpixel_indices = precompute_superpixel_indices(labels_slic)
G1 = graph_maker(num_superpixels)
end_time_s_i = time.time()
print(end_time_s_i-start_time_s_i)
neighbors_in_superpixels = fill_neighbors_6(
    labels_slic, superpixel_indices, num_superpixels)

double_neighbors_in_superpixels = double_fill_neighbors_6(
    neighbors_in_superpixels)
neighbors_in_superpixels = double_neighbors_in_superpixels.copy()
# superpixel_information = get_superpixel_information(
#     labels_slic, num_superpixels, neighbors_in_superpixels)

# End time
end_time1 = time.time()
elapsed_time1 = end_time1-start_time1
print('Time elapsed in getting neighbors = '+str(elapsed_time1))


# Start time
start_time2 = time.time()

i = 0
for each_node in G1.nodes():
    # neighbors_of_nodes = get_neighboring_superpixels(labels_slic, each_node)

    neighbors_of_nodes = neighbors_in_superpixels[each_node]

    for each_neighbor in neighbors_of_nodes:
        if not G1.has_edge(each_node, each_neighbor):
            i = i+1
            G1.add_edge(each_node, each_neighbor, weight=similarity_coefficient_calculator_and_value_returner4(
                superpixel_indices, image, each_node, each_neighbor, superpixel_centroids))
print(i)
# End time
end_time2 = time.time()
# Calculate elapsed time
elapsed_time2 = end_time2 - start_time2
print("Elapsed time for graph construction approach 1:", elapsed_time2, "seconds")

# Start time
start_time3 = time.time()
# for i in range(num_superpixels):
#     if G1.has_edge(i, i):
#         G1.remove_edge(i, i)
G1.add_node('background')
G1.add_node('foreground')

# Define the path to the JSON file
file_path_1 = f"{image_num}.json"

# Retrieve the lists from the JSON file
with open(path_to_add+ 'scribbled/flowers/'+file_path_1, 'r') as json_file:
    data = json.load(json_file)
    foreground_superpixels = data["list1"]
    background_superpixels = data["list2"]


for each_superpixel in foreground_superpixels:  # type: ignore
    G1.add_edge('foreground', each_superpixel, weight=1000)
for each_superpixel in background_superpixels:  # type: ignore
    G1.add_edge('background', each_superpixel, weight=1000)
maximum_spanning_tree = nx.maximum_spanning_tree(G1)


# helper(maximum_spanning_tree)
# maximum_spanning_tree = nx.maximum_spanning_tree(G1)

critical_edges = find_critical_edges(
    maximum_spanning_tree, 'background', 'foreground')
helper6(critical_edges, maximum_spanning_tree)
# Your code to be measured goes here

# End time
end_time3 = time.time()

# Calculate elapsed time
elapsed_time3 = end_time3 - start_time3
print("Elapsed time for maximum spanning tree partitioning approach 1:",
      elapsed_time3, "seconds")


initial_method1_ending_time = time.time()

# Calculate elapsed time
elapsed_time = initial_method1_ending_time - initial_method1_starting_time
print("Total time for approach 1:",
      elapsed_time, "seconds")


# helper1(image, maximum_spanning_tree, labels_slic)

# helper2(image, maximum_spanning_tree, labels_slic)
iteration_count_1 = 0

mask1 = generate_mask1(image, maximum_spanning_tree, labels_slic)
plt.imshow(mask1)
plt.imsave(mask_img_path + '_5_1_' +
           str(iteration_count_1)+'.png',
           mask1)
plt.axis('off')
plt.show()
seg_img = helper31(image, maximum_spanning_tree, labels_slic)
# cv2.cvtColor(seg_img, cv2.COLOR_BGR2RGB)
plt.imshow(seg_img)
plt.imsave(seg_img_path + '_5_1_' +
           str(iteration_count_1)+'.png',
           seg_img)
plt.axis('off')
plt.show()

elapsed_time_4_list = []
while True:
    continue_response = ask_to_continue()
    if continue_response:
        '''
        code of another iteration
        '''

        with open('another_iteration_5.py', 'r') as f:
            code = f.read()

        # Execute the code using exec()
        exec(code)
        elapsed_time_4_list.append(elapsed_time4)  # type: ignore
        iteration_count_1 = iteration_count_1 + 1
    else:
        '''
        Print the final results

        '''
        with open('results_5.py', 'r') as f:
            code = f.read()

        # Execute the code using exec()
        exec(code)
        break

# detailed_continue = ask_to_detailed_continue()
# if detailed_continue:
#     with open('reduce_false_positive.py', 'r') as f:
#         code = f.read()
#     # Execute the code using exec()
#     exec(code)
