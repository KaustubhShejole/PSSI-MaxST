
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
from fun import ask_to_continue, ask_to_detailed_continue, bhattacharyya_similarity_coefficient_calculator, compute_normalized_histogram4, fill_neighbors, fill_neighbors_5, fill_neighbors_6, fill_neighbors_random_8, find_critical_edges, generate_mask, generate_mask1, get_cluster_at_point, get_superpixels_by_pixels, get_neighboring_superpixels, compute_normalized_histogram, fill_neighbors_4, graph_maker, helper, helper1, helper2, helper3, helper31, helper5, helper6, max_spanning_tree_with_cut, precompute_superpixel_indices, similarity_coefficient_calculator_and_value_returner, similarity_coefficient_calculator_and_value_returner4, abc

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
            G1.add_edge(each_node, each_neighbor, weight=bhattacharyya_similarity_coefficient_calculator(
                superpixel_indices, image, each_node, each_neighbor))
print(i)
# End time
end_time2 = time.time()
# Calculate elapsed time
elapsed_time2 = end_time2 - start_time2
print("Elapsed time for graph construction using bhattacharyya similarity measure:", elapsed_time2, "seconds")

# Start time
start_time3 = time.time()
# for i in range(num_superpixels):
#     if G1.has_edge(i, i):
#         G1.remove_edge(i, i)
G1.add_node('background')
G1.add_node('foreground')

# # Define the path to the JSON file
# file_path_1 = f"{image_num}.json"

# # Retrieve the lists from the JSON file
# with open(path_to_add + 'scribbled/flowers/'+file_path_1, 'r') as json_file:
#     data = json.load(json_file)
#     foreground_superpixels = data["list1"]
#     background_superpixels = data["list2"]


for each_superpixel in foreground_superpixels:  # type: ignore
    G1.add_edge('foreground', each_superpixel, weight=1000)
for each_superpixel in background_superpixels:  # type: ignore
    G1.add_edge('background', each_superpixel, weight=1000)
maximum_spanning_tree = nx.maximum_spanning_tree(G1)


# helper(maximum_spanning_tree)
# maximum_spanning_tree = nx.maximum_spanning_tree(G1)

# critical_edges = find_critical_edges(
#     maximum_spanning_tree, 'background', 'foreground')
# helper6(critical_edges, maximum_spanning_tree)
# # Your code to be measured goes here


def graph_maker1(num_superpixels):
    G = nx.DiGraph()
    for i in range(num_superpixels):
        G.add_node(i)
    return G


def convert_to_directed_graph_with_checks1(G):
    directed_G = nx.DiGraph()
    dummy_count = 1

    for u, v, data in G.edges(data=True):
        weight = data.get('weight', 1)  # Default weight is 1 if not specified

        # Apply rule 1 if either node is 'background'
        if u == 'background':
            directed_G.add_edge('background', v, weight=weight)
        elif v == 'background':
            directed_G.add_edge('background', u, weight=weight)

        # Apply rule 2 if either node is 'foreground'
        elif u == 'foreground':
            directed_G.add_edge(v, 'foreground', weight=weight)
        elif v == 'foreground':
            directed_G.add_edge(u, 'foreground', weight=weight)

        # Apply rule 3 for standard nodes
        else:
            # Add the direct edge
            directed_G.add_edge(u, v, weight=weight)

            # Create the dummy node and additional edges
            dummy_node = f"node{dummy_count}"
            directed_G.add_edge(v, dummy_node, weight=weight)
            directed_G.add_edge(dummy_node, u, weight=weight)

            dummy_count += 1

    return directed_G


def apply_translucent_scribble(image, x, y, scribble_color, alpha):
    current_color = image[y, x]

    # Blend the scribble color with the current color using alpha blending
    result_color = [
        (1 - alpha / 255.0) *
        current_color[0] + (alpha / 255.0) * scribble_color[0],
        (1 - alpha / 255.0) *
        current_color[1] + (alpha / 255.0) * scribble_color[1],
        (1 - alpha / 255.0) *
        current_color[2] + (alpha / 255.0) * scribble_color[2]
    ]
    return result_color


def image_mask_max_flow(image, f, b, labels_slic):
    # segmented_image = np.zeros((image_height, image_width, 3), dtype=np.uint8)
    segmented_image = np.copy(image)
    # Assign colors to the connected components

    for node in b:
        coords_superpixel = np.argwhere(
            labels_slic == node)
        for (y, x) in coords_superpixel:
            try:
                segmented_image[y, x] = [0, 0, 0]
                # apply_translucent_scribble(
                #     segmented_image, x, y, [0, 0, 255], alpha=128)
            except Exception as e:
                # print(e)
                pass
    for node in f:
        coords_superpixel = np.argwhere(
            labels_slic == node)
        for (y, x) in coords_superpixel:
            try:
                segmented_image[y, x] = [255, 255, 255]
                # apply_translucent_scribble(
                #     segmented_image, x, y, [0, 0, 255], alpha=128)
            except Exception as e:
                # print(e)
                pass
    # Display the segmented image
    # segmented_image = cv2.cvtColor(segmented_image, cv2.COLOR_BGR2RGB)

    plt.imshow(segmented_image)
    # plt.imsave('flower_segmentation_dataset/segmented/1.png',
    #            segmented_image)
    plt.axis('off')
    # plt.show()
    plt.close()
    return segmented_image


def image_segmentation_max_flow(image, f, b, labels_slic):
    # segmented_image = np.zeros((image_height, image_width, 3), dtype=np.uint8)
    segmented_image = np.copy(image)
    # Assign colors to the connected components

    for node in b:
        coords_superpixel = np.argwhere(
            labels_slic == node)
        for (y, x) in coords_superpixel:
            try:
                segmented_image[y, x] = apply_translucent_scribble(
                    segmented_image, x, y, [0, 0, 255], alpha=128)
            except Exception as e:
                # print(e)
                pass
    # Display the segmented image
    # segmented_image = cv2.cvtColor(segmented_image, cv2.COLOR_BGR2RGB)

    plt.imshow(segmented_image)
    # plt.imsave('flower_segmentation_dataset/segmented/1.png',
    #            segmented_image)
    plt.axis('off')
    # plt.show()
    plt.close()
    return segmented_image


directed_graph1 = convert_to_directed_graph_with_checks1(G1)


'''
Now apply maxflow algorithm to calculate maxflow and 
partition the image accordingly.

'''
# Compute the maximum flow from source 's' to sink 't'
# flow_value, flow_dict = nx.maximum_flow(
#     directed_graph1, 'background', 'foreground', capacity='weight')

cut_value, (set1, set2) = nx.minimum_cut(directed_graph1,
                                         'background', 'foreground', capacity='weight')


# End time
end_time3 = time.time()

# Calculate elapsed time
elapsed_time3 = end_time3 - start_time3
print("Elapsed time for maxflow partitioning approach 1:",
      elapsed_time3, "seconds")


initial_method1_ending_time = time.time()

# Calculate elapsed time
elapsed_time = initial_method1_ending_time - initial_method1_starting_time
print("Total time for first iteration bhattacharyya siilarity coefficient/measure:",
      elapsed_time, "seconds")


# helper1(image, maximum_spanning_tree, labels_slic)

# helper2(image, maximum_spanning_tree, labels_slic)
iteration_count_1 = 0

import numbers

foreground_set = {item for item in set2 if isinstance(item, numbers.Integral)}
background_set = {item for item in set1 if isinstance(item, numbers.Integral)}

mask1 = image_mask_max_flow(
    image, foreground_set, background_set, labels_slic)
plt.imshow(mask1)
plt.imsave(mask_img_path + '_1_1_bh_' +
           str(iteration_count_1)+ scribbles_from + '.png',
           mask1)
plt.axis('off')
# plt.show()
plt.close()

seg_img = image_segmentation_max_flow(
    image, foreground_set, background_set, labels_slic)
# cv2.cvtColor(seg_img, cv2.COLOR_BGR2RGB)
plt.imshow(seg_img)
plt.imsave(seg_img_path + '_1_1_bh_' +
           str(iteration_count_1)+ scribbles_from+ '.png',
           seg_img)
plt.axis('off')
# plt.show()
plt.close()

elapsed_time_4_list = []
while True:
    # continue_response = ask_to_continue()
    continue_response = False
    if continue_response:
        '''
        code of another iteration
        '''
        with open('another_iteration_bh.py', 'r') as f:
            code = f.read()

        # Execute the code using exec()
        exec(code)
        elapsed_time_4_list.append(elapsed_time4)  # type: ignore
        iteration_count_1 = iteration_count_1 + 1
    else:
        '''
        Print the final results

        '''
        with open('results_bh.py', 'r') as f:
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
