from fun import similarity_coefficient_calculator_and_value_returner4_from_histograms
import json
import time
# from parameters_and_data import image_rgb, fpoints, bpoints, labels_slic, result, num_superpixels, image_path, seg_img_path, mask_img_path, num_superpixels_parameter
# from parameters_and_data import image_rgb, scribbling_dimension, fpoints, bpoints, visualization_image_path, scribbled_img_path, ground_truth_path, image_num
import networkx as nx
import copy
import math
import matplotlib.pyplot as plt

import cv2
import numpy as np
from fun import ask_to_continue, ask_to_detailed_continue, compute_normalized_histograms, fill_neighbors, fill_neighbors_5, fill_neighbors_6, fill_neighbors_random_8, find_critical_edges, generate_mask, generate_mask1, get_cluster_at_point, get_superpixels_by_pixels, get_neighboring_superpixels, compute_normalized_histogram, fill_neighbors_4, graph_maker, helper, helper1, helper2, helper3, helper31, helper5, helper6, max_spanning_tree_with_cut, precompute_superpixel_indices, similarity_coefficient_calculator_and_value_returner, similarity_coefficient_calculator_and_value_returner4, abc, precompute_superpixel_indices_modified, fill_neighbors_6_optimized, similarity_coefficient_calculator_and_value_returner4_optimized, find_critical_edges_modified, sips_similarity, compute_laplace_histograms

image = image_rgb
superpixel_centroids = []


initial_method1_starting_time = time.time()


# Start time
start_time1 = time.time()
start_time_s_i = time.time()
superpixel_indices = precompute_superpixel_indices_modified(labels_slic)
# G1 = graph_maker(num_superpixels)
end_time_s_i = time.time()
# print(end_time_s_i-start_time_s_i)
# neighbors_in_superpixels1 = fill_neighbors_6_optimized(
#     labels_slic, superpixel_indices, num_superpixels)

from fun import fill_neighbors_6_optimized1, fill_neighbors_custom_offset

neighbors_in_superpixels = fill_neighbors_custom_offset(
    labels_slic, superpixel_indices, num_superpixels)

# check if both are same or contain same list

# superpixel_information = get_superpixel_information(
#     labels_slic, num_superpixels, neighbors_in_superpixels)

# End time
end_time1 = time.time()
elapsed_time1 = end_time1-start_time1
# print('Time elapsed in getting neighbors = '+str(elapsed_time1))


# Start time
start_time2 = time.time()

# i = 0
# for each_node in G1.nodes():
#     # neighbors_of_nodes = get_neighboring_superpixels(labels_slic, each_node)

#     neighbors_of_nodes = neighbors_in_superpixels[each_node]

#     for each_neighbor in neighbors_of_nodes:
#         if not G1.has_edge(each_node, each_neighbor):
#             i = i+1
#             G1.add_edge(each_node, each_neighbor, weight=similarity_coefficient_calculator_and_value_returner4(
#                 superpixel_indices, image, each_node, each_neighbor, superpixel_centroids))
# print(i)

from fun import bhattacharyya_similarity_coefficient_calculator_optimized


num_bins = 8
# (1) Precompute every superpixel’s histogram once:
histograms = compute_normalized_histograms(
    image, superpixel_indices, num_bins=num_bins)


def build_neighbor_matrix(num_bins, lambda_param):
    """
    Build the A matrix for ±1-bin weighted similarity.
    """
    A = np.zeros((num_bins, num_bins), dtype=np.float32)
    for i in range(num_bins):
        if i > 0:
            A[i, i - 1] = lambda_param
        if i < num_bins - 1:
            A[i, i + 1] = lambda_param
    return A


A = build_neighbor_matrix(num_bins=num_bins, lambda_param=0.1)
I_plus_A = np.eye(num_bins) + A


def matrix_form_similarity(histograms, cluster_label, cluster_label1):
    p = np.array(histograms[cluster_label])
    q = np.array(histograms[cluster_label1])
    num_channels, num_bins = p.shape

    sim = []
    for c in range(num_channels):
        val = p[c] @ I_plus_A @ q[c]
        sim.append(np.sqrt(val))

    harmonic_mean = num_channels / np.sum(1.0 / np.array(sim))
    return 100.0 * harmonic_mean

# (2) Initialize graph with all nodes:
G1 = nx.Graph()
G1.add_nodes_from(range(num_superpixels))
# (3) Add each undirected edge exactly once, looking up the histograms:
for u in range(num_superpixels):
    for v in neighbors_in_superpixels[u]:
        if v <= u:
            continue    # skip duplicates and self-loops
        # fast similarity lookup
        # w = similarity_coefficient_calculator_and_value_returner4_from_histograms(
        #     histograms, u, v, 0.2)
        w = matrix_form_similarity(
            histograms, u, v)
        # w = bhattacharyya_similarity_coefficient_calculator_optimized(
        #     histograms, u, v)
        G1.add_edge(u, v, weight=w)


# End time
end_time2 = time.time()
# Calculate elapsed time
elapsed_time2 = end_time2 - start_time2
# print("Elapsed time for graph construction approach 1:", elapsed_time2, "seconds")

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
# with open(path_to_add+ 'scribbled/flowers/'+file_path_1, 'r') as json_file:
#     data = json.load(json_file)
#     foreground_superpixels = data["list1"]
#     background_superpixels = data["list2"]

# visualization_image = np.zeros_like(image_rgb)
# for each_label in foreground_superpixels:
#     visualization_image[labels_slic == each_label] = [
#         0, 255, 0]  # Green color
# # Color the background superpixel (blue)
# for each_label in background_superpixels:
#     visualization_image[labels_slic == each_label] = [0, 0, 255]
# # Convert BGR to RGB for displaying with matplotlib
# visualization_image_rgb = cv2.cvtColor(visualization_image, cv2.COLOR_BGR2RGB)

# # Plot the visualization image
# plt.imshow(visualization_image_rgb)
# # plt.imsave('visualising_superpixels/3.png', visualization_image_rgb)
# plt.axis('off')
# # plt.show()
# plt.close()
# print('visualization_image')



for each_superpixel in foreground_superpixels:  # type: ignore
    G1.add_edge('foreground', each_superpixel, weight=np.inf)
for each_superpixel in background_superpixels:  # type: ignore
    G1.add_edge('background', each_superpixel, weight=np.inf)
maximum_spanning_tree = nx.maximum_spanning_tree(G1)


# helper(maximum_spanning_tree)
# maximum_spanning_tree = nx.maximum_spanning_tree(G1)

critical_edges = find_critical_edges_modified(
    maximum_spanning_tree, 'background', 'foreground')
helper6(critical_edges, maximum_spanning_tree)
# Your code to be measured goes here

# End time
end_time3 = time.time()

# Calculate elapsed time
elapsed_time3 = end_time3 - start_time3
# print("Elapsed time for maximum spanning tree partitioning approach 1:",
#       elapsed_time3, "seconds")


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
plt.imsave('results/mask_' + str(image_num) + '_' +
           str(iteration_count_1) + '_' + scribbles_from + '.png',
           mask1)
plt.axis('off')
plt.show()
plt.close()


seg_img = helper31(image, maximum_spanning_tree, labels_slic)
# cv2.cvtColor(seg_img, cv2.COLOR_BGR2RGB)
plt.imshow(seg_img)
plt.imsave('results/segmentation_' + str(image_num) + '_' +
           str(iteration_count_1)+'_' + scribbles_from + '.png',
           seg_img)
plt.axis('off')
plt.show()
plt.close()

elapsed_time_4_list = []
while True:
    # continue_response = ask_to_continue()
    continue_response = False
    if continue_response:
        '''
        code of another iteration
        '''

        with open('another_iteration_1.py', 'r') as f:
            code = f.read()

        # Execute the code using exec()
        exec(code)
        elapsed_time_4_list.append(elapsed_time4)  # type: ignore
        iteration_count_1 = iteration_count_1 + 1
    else:
        '''
        Print the final results

        '''
        with open('results_1.py', 'r') as f:
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
