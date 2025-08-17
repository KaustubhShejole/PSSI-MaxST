from scipy.ndimage import convolve1d
import time
import heapq
from tkinter import messagebox
import networkx as nx
import copy
import math
import matplotlib.pyplot as plt
import random
import cv2
import numpy as np
# hyperparameters
# from parameters_and_data import num_superpixels_parameter, compactness

# data
# from parameters_and_data import image_rgb, fpoints, bpoints, labels_slic, result, num_superpixels, original_image, num_superpixels_parameter

# Define a function to get the label (cluster) at a specific point
import os

import tkinter as tk
from tkinter import ttk


def on_select(method):
    global selected_method
    selected_method = method
    root.destroy()


def choose_method():
    global root, selected_method
    selected_method = None

    # Create the main window
    root = tk.Tk()
    root.title("Select Method")
    root.geometry("900x500")
    root.configure(bg='lightgreen')  # Set background color

    # Create a label with some padding and border
    label = tk.Label(root, text="Please choose a method:", bg='lightgreen', fg='maroon', font=(
        "Helvetica", 20, "bold"), pady=10, padx=10, borderwidth=2, relief="groove")
    label.pack(pady=40)

    # Create a frame for the buttons to center them with a different background color
    button_frame = tk.Frame(root, bg='lightblue',
                            borderwidth=5, relief="ridge")
    button_frame.pack(pady=20, padx=20)

    # Define hover effect functions with color changes
    def on_enter(event, btn):
        btn['background'] = 'skyblue'

    def on_leave(event, btn):
        btn['background'] = btn.defaultBackground

    # Create buttons for Method1 and Method2 with more colorful styles
    button1 = tk.Button(button_frame, text="Maximum Spanning Tree Based Method",
                        command=lambda: on_select(1), font=("Helvetica", 16), width=40, height=2, relief='raised', borderwidth=5, bg='lightyellow', fg='darkblue')
    button1.grid(row=0, column=0, padx=30, pady=10)
    button1.defaultBackground = button1['background']
    button1.bind("<Enter>", lambda event: on_enter(event, button1))
    button1.bind("<Leave>", lambda event: on_leave(event, button1))

    button2 = tk.Button(button_frame, text="Minimum Cut Based Method",
                        command=lambda: on_select(2), font=("Helvetica", 16), width=40, height=2, relief='raised', borderwidth=5, bg='lightpink', fg='darkgreen')
    button2.grid(row=1, column=0, padx=30, pady=10)

    button3 = tk.Button(button_frame, text="Maximum Spanning Tree Based Method with double nbd",
                        command=lambda: on_select(3), font=("Helvetica", 16), width=40, height=2, relief='raised', borderwidth=5, bg='lightpink', fg='darkgreen')
    button3.grid(row=3, column=0, padx=30, pady=10)
    button3.defaultBackground = button3['background']
    button3.bind("<Enter>", lambda event: on_enter(event, button3))
    button3.bind("<Leave>", lambda event: on_leave(event, button3))

    button4 = tk.Button(button_frame, text="Maximum Spanning Tree Based Method with Bhattacharyya Similarity Measure",
                        command=lambda: on_select(4), font=("Helvetica", 16), width=40, height=2, relief='raised', borderwidth=5, bg='lightpink', fg='darkgreen')
    button4.grid(row=4, column=0, padx=30, pady=10)
    button4.defaultBackground = button4['background']
    button4.bind("<Enter>", lambda event: on_enter(event, button4))
    button4.bind("<Leave>", lambda event: on_leave(event, button4))



    # Adding images to buttons (Optional, requires image files)
    # You need to have image files named 'method1.png' and 'method2.png' in the same directory as this script.
    # Uncomment the lines below if you have these images.
    # method1_img = tk.PhotoImage(file="method1.png")
    # method2_img = tk.PhotoImage(file="method2.png")
    # button1.config(image=method1_img, compound='left')
    # button2.config(image=method2_img, compound='left')

    root.mainloop()

    return selected_method


def run_method_1():
    with open('analysis_1.py', 'r') as f:
        code = f.read()

    # Execute the code using exec()
    exec(code)


def run_method_2():
    with open('analysis_2.py', 'r') as f:
        code = f.read()

    # Execute the code using exec()
    exec(code)


def ask_to_continue():
    # Create a simple dialog box asking the user if they want to continue
    root = tk.Tk()
    root.withdraw()  # Hide the root window
    user_response = messagebox.askyesno(
        "Continue?", "Do you want to continue to the next iteration?")
    root.destroy()
    return user_response


def ask_to_detailed_continue():
    # Create a simple dialog box asking the user if they want to continue
    root = tk.Tk()
    root.withdraw()  # Hide the root window
    user_response = messagebox.askyesno(
        "Continue?", "Do you want to continue for more finer segmentation?")
    root.destroy()
    return user_response


def get_unique_filename(base_name, extension):
    counter = 1
    while True:
        filename = f"{base_name}_{counter}.{extension}"
        if not os.path.exists(filename):
            return counter
        counter += 1


def get_cluster_at_point(x, y):
    return labels_slic[y, x]  # OpenCV uses (x, y) coordinates


def precompute_superpixel_indices(labels_slic):
    superpixel_indices = {}
    num_superpixels = labels_slic.max() + 1
    for label in range(num_superpixels):
        superpixel_indices[label] = np.where(labels_slic == label)
    return superpixel_indices


def precompute_superpixel_indices_modified(labels_slic):
    """
    Returns a dict mapping each superpixel label to a tuple of (rows, cols),
    computed in a single sort-and-split pass.
    """
    H, W = labels_slic.shape
    flat_labels = labels_slic.ravel()
    flat_inds = np.arange(flat_labels.size)

    # Sort pixel indices by their label
    order = np.argsort(flat_labels)
    sorted_lbls = flat_labels[order]

    # Find the boundaries where the label changes
    boundaries = np.nonzero(np.diff(sorted_lbls))[0] + 1

    # Split the ordered indices into one array per label
    groups = np.split(order, boundaries)

    # Build the mapping from label->(row_indices, col_indices)
    superpixel_indices = {}
    # The unique labels in sorted order:
    unique_labels = sorted_lbls[np.concatenate([[0], boundaries])]

    for lbl, inds in zip(unique_labels, groups):
        rows = inds // W
        cols = inds % W
        superpixel_indices[int(lbl)] = (rows, cols)

    return superpixel_indices

def compute_normalized_histogram4(image, superpixel_indices, superpixel_label):
    # sample_fraction = 1.0
    # indices = superpixel_indices[superpixel_label]

    # # Number of pixels in the superpixel

    # # Sample a subset of the indices to reduce complexity
    # sample_size = int(num_pixels * sample_fraction)
    # if sample_size < 1:
    #     sample_size = 1  # Ensure at least one pixel is sampled

    # sampled_indices = np.random.choice(num_pixels, sample_size, replace=False)
    # sampled_pixels = image[indices[0]
    #                        [sampled_indices], indices[1][sampled_indices]]
    # Get all indices of the superpixel
    indices = superpixel_indices[superpixel_label]
    num_pixels = len(indices[0])
    sample_size = num_pixels

    # Use all the pixels in the superpixel
    sampled_pixels = image[indices[0], indices[1]]
    # Calculate histogram for each channel separately
    hist_r, _ = np.histogram(sampled_pixels[:, 0], bins=8, range=(0, 255))
    hist_g, _ = np.histogram(sampled_pixels[:, 1], bins=8, range=(0, 255))
    hist_b, _ = np.histogram(sampled_pixels[:, 2], bins=8, range=(0, 255))

    # Normalize histograms for each channel
    hist_r_normalized = hist_r / sample_size + 0.001
    hist_g_normalized = hist_g / sample_size + 0.001
    hist_b_normalized = hist_b / sample_size + 0.001

    # Concatenate normalized histograms for all channels
    superpixel_histogram_normalized = [
        hist_r_normalized, hist_g_normalized, hist_b_normalized]

    return superpixel_histogram_normalized

    # indices = np.where(labels_slic == superpixel_label)

    # # Number of pixels in the superpixel
    # num_pixels = len(indices[0])

    # # Sample a subset of the indices to reduce complexity
    # sample_size = int(num_pixels * sample_fraction)
    # if sample_size < 1:
    #     sample_size = 1  # Ensure at least one pixel is sampled

    # sampled_indices = np.random.choice(num_pixels, sample_size, replace=False)
    # sampled_pixels = image[indices[0]
    #                        [sampled_indices], indices[1][sampled_indices]]
    # # Mask to select pixels belonging to the specified superpixel
    # mask = labels_slic == superpixel_label

    # # Select pixel values for the specified superpixel
    # superpixel_pixels = image[mask]

    # # Sample a subset of the pixels to reduce complexity
    # num_pixels = len(superpixel_pixels)
    # sample_size = num_pixels
    # if sample_size < 1:
    #     sample_size = 1  # Ensure at least one pixel is sampled

    # sampled_pixels = superpixel_pixels[np.random.choice(
    #     num_pixels, sample_size, replace=False)]

    # # Calculate histogram for each channel separately
    # hist_r, _ = np.histogram(sampled_pixels[:, 0], bins=8, range=(0, 255))
    # hist_g, _ = np.histogram(sampled_pixels[:, 1], bins=8, range=(0, 255))
    # hist_b, _ = np.histogram(sampled_pixels[:, 2], bins=8, range=(0, 255))

    # # Normalize histograms for each channel
    # hist_r_normalized = hist_r / sample_size + 0.001
    # hist_g_normalized = hist_g / sample_size + 0.001
    # hist_b_normalized = hist_b / sample_size + 0.001

    # # Concatenate normalized histograms for all channels
    # superpixel_histogram_normalized = [
    #     hist_r_normalized, hist_g_normalized, hist_b_normalized]

    # return superpixel_histogram_normalized
    return 0


def compute_normalized_histogram4_optimized(image, superpixel_indices, superpixel_label, num_bins=8, alpha=1.0):
    """
    Computes a Laplace-smoothed, normalized RGB histogram for one superpixel.
    Returns a list of three arrays (hist_r, hist_g, hist_b), each of length num_bins.
    """
    rows, cols = superpixel_indices[superpixel_label]
    pixels = image[rows, cols]  # shape (n_pixels, 3)
    n = pixels.shape[0]

    # Compute histograms and apply Laplace smoothing + normalization
    histograms = []
    for c in range(3):  # for R, G, B channels
        h, _ = np.histogram(pixels[:, c], bins=num_bins, range=(0, 255))
        # Laplace smoothing
        h = (h + alpha) / (n + alpha * num_bins)
        histograms.append(h)

    return histograms


import numpy as np


def similarity_coefficient_calculator_and_value_returner4_optimized(
    histograms, cluster_label, cluster_label1, lambda_parameter
):
    # Compute normalized histograms
    superpixel_histogram_normalized = histograms[cluster_label]
    superpixel_histogram_normalized1 =histograms[cluster_label1]

    # Convert to NumPy arrays for vectorized operations
    p = np.array(superpixel_histogram_normalized)
    q = np.array(superpixel_histogram_normalized1)
    num_channels, num_bins = p.shape

    # Parameters
    lambda_parameter = 0.2

    # Base term: p_jk * q_jk for all bins and channels
    base_term = np.sum(p * q, axis=1)

    # Middle bins (1 to num_bins-2): λ * (p_jk * q_j(k+1) + p_jk * q_j(k-1))
    if num_bins > 2:
        middle = p[:, 1:-1] * (lambda_parameter *
                               q[:, :-2] + lambda_parameter * q[:, 2:])
        middle_term = np.sum(middle, axis=1)
    else:
        middle_term = np.zeros(num_channels)

    # First bin: λ * p_j0 * q_j1
    first_bin_term = np.zeros(num_channels)
    if num_bins > 1:
        first_bin_term = lambda_parameter * p[:, 0] * q[:, 1]

    # Combine all terms and compute square root
    total = base_term + middle_term + first_bin_term
    individual_coeffs = np.sqrt(total)

    # Compute harmonic mean efficiently
    harmonic_mean = len(individual_coeffs) / np.sum(1.0 / individual_coeffs)

    return 100 * harmonic_mean

def compute_normalized_histogram(image, labels_slic, superpixel_label):
    # Mask to select pixels belonging to the specified superpixel
    mask = labels_slic == superpixel_label

    # Select pixel values for the specified superpixel
    superpixel_pixels = image[mask]
    if len(superpixel_pixels) == 0:
        print("length is zero")
        print(np.array(superpixel_pixels).shape)
        exit()
    else:
        print(np.array(superpixel_pixels).shape)
        print("Length Not Zero How??")
        exit()

    # Calculate histogram for each channel separately
    hist_r, _ = np.histogram(superpixel_pixels[:, 0], bins=8, range=(0, 255))
    hist_g, _ = np.histogram(superpixel_pixels[:, 1], bins=8, range=(0, 255))
    hist_b, _ = np.histogram(superpixel_pixels[:, 2], bins=8, range=(0, 255))

    # Concatenate histograms for all channels
    # superpixel_histogram = np.concatenate([hist_r, hist_g, hist_b])
    hist_r_normalized = hist_r/len(superpixel_pixels) + 0.001
    hist_b_normalized = hist_b/len(superpixel_pixels) + 0.001
    hist_g_normalized = hist_g/len(superpixel_pixels) + 0.001
    superpixel_histogram_normalized = [
        hist_r_normalized, hist_g_normalized, hist_b_normalized]

    return superpixel_histogram_normalized


def plot_normalized_histogram(rgb_histograms):
    channels = ['red', 'green', 'blue']

    # Create bins for the histograms
    # Assuming all histograms have the same number of bins
    bins = np.arange(len(rgb_histograms[0]))

    # Plot histograms for each channel
    plt.figure(figsize=(12, 4))
    for i in range(3):
        plt.subplot(1, 3, i+1)
        plt.bar(bins, rgb_histograms[i], color=channels[i], alpha=0.7)
        plt.title('Normalized Histogram ({})'.format(channels[i]))
        plt.xlabel('Bin')
        plt.ylabel('Normalized Frequency')
        plt.grid(True)
    plt.tight_layout()
    plt.show()


'''
# Similarity function without taking into consideration neighboring bins
def similarity_coefficient_calculator(superpixel_histogram_normalized, superpixel_histogram_normalized1, cluster_label, cluster_label1):
    # Measure of Similarity : Harmonic mean of the (sum root(pR * p'R), sum root(pG * p'G), sum root(pB * p'B))
    individual_similarity_coefficient = []
    i = 0
    num_bins = (len(superpixel_histogram_normalized[0]))
    for j in range(len(superpixel_histogram_normalized)):
        sum = 0
        for bin in range(num_bins):
            sum = sum + \
                superpixel_histogram_normalized[j][bin] * \
                superpixel_histogram_normalized1[j][bin]
        individual_similarity_coefficient.append(np.sqrt(sum))
    print(individual_similarity_coefficient)

    sum = 0
    for each_coefficient in individual_similarity_coefficient:
        sum = sum + (1/each_coefficient)

    similarity_index_between_superpixels = len(
        individual_similarity_coefficient)/sum
    print(similarity_index_between_superpixels)

    centroid_first_superpixel = (
        superpixel_centroids[cluster_label][0], superpixel_centroids[cluster_label][1])
    centroid_second_superpixel = (
        superpixel_centroids[cluster_label1][0], superpixel_centroids[cluster_label1][1])
    euclidean_distance_between_two_superpixels = np.sqrt(np.square(
        centroid_first_superpixel[0]-centroid_second_superpixel[0]) + np.square(centroid_first_superpixel[1]-centroid_second_superpixel[1]))
    print(centroid_first_superpixel)
    print(centroid_second_superpixel)
    print(euclidean_distance_between_two_superpixels)

    image_height = image.shape[0]
    image_width = image.shape[1]
    print(f'image_height: {image_height}, image_width: {image_width}')
    maximum_distance_possible_between_two_pixels = np.sqrt(
        np.square(image_height)+np.square(image_width))
    normalized_distance_between_superpixels = euclidean_distance_between_two_superpixels / \
        maximum_distance_possible_between_two_pixels
    print(normalized_distance_between_superpixels)
'''


# def similarity_coefficient_calculator(labels_slic, image, cluster_label, cluster_label1, superpixel_centroids):
#     superpixel_histogram_normalized = compute_normalized_histogram(
#         image, labels_slic, cluster_label)
#     superpixel_histogram_normalized1 = compute_normalized_histogram(
#         image, labels_slic, cluster_label1)

#     # Measure of Similarity : Harmonic mean of the (sum root(pR * p'R), sum root(pG * p'G), sum root(pB * p'B))
#     individual_similarity_coefficient = []
#     i = 0
#     lambda_parameter = 0.5
#     num_bins = (len(superpixel_histogram_normalized[0]))
#     for j in range(len(superpixel_histogram_normalized)):
#         sum = 0
#         for bin in range(num_bins):
#             sum = sum + \
#                 superpixel_histogram_normalized[j][bin] * \
#                 superpixel_histogram_normalized1[j][bin]
#             if bin != num_bins-1 and bin != 0:
#                 sum = sum + \
#                     lambda_parameter*((superpixel_histogram_normalized[j][bin] *
#                                        superpixel_histogram_normalized1[j][bin+1]) + (superpixel_histogram_normalized[j][bin] *
#                                                                                       superpixel_histogram_normalized1[j][bin-1]))
#             if bin == 0:
#                 sum = sum + lambda_parameter*(superpixel_histogram_normalized[j][bin] *
#                                               superpixel_histogram_normalized1[j][bin+1])

#         individual_similarity_coefficient.append(np.sqrt(sum))
#     print(individual_similarity_coefficient)

#     sum = 0
#     for each_coefficient in individual_similarity_coefficient:
#         sum = sum + (1/each_coefficient)

#     similarity_index_between_superpixels = len(
#         individual_similarity_coefficient)/sum
#     print(similarity_index_between_superpixels)

#     sum = 0
#     # for each_coefficient in individual_similarity_coefficient:
#     sum = 0.21*individual_similarity_coefficient[0] + 0.72 * \
#         individual_similarity_coefficient[1] + \
#         0.07*individual_similarity_coefficient[2]
#     print(sum)
#     centroid_first_superpixel = (
#         superpixel_centroids[cluster_label][0], superpixel_centroids[cluster_label][1])
#     centroid_second_superpixel = (
#         superpixel_centroids[cluster_label1][0], superpixel_centroids[cluster_label1][1])
#     euclidean_distance_between_two_superpixels = np.sqrt(np.square(
#         centroid_first_superpixel[0]-centroid_second_superpixel[0]) + np.square(centroid_first_superpixel[1]-centroid_second_superpixel[1]))
#     print(centroid_first_superpixel)
#     print(centroid_second_superpixel)
#     print(euclidean_distance_between_two_superpixels)

#     image_height = image.shape[0]
#     image_width = image.shape[1]
#     print(f'image_height: {image_height}, image_width: {image_width}')
#     maximum_distance_possible_between_two_pixels = np.sqrt(
#         np.square(image_height)+np.square(image_width))
#     normalized_distance_between_superpixels = euclidean_distance_between_two_superpixels / \
#         maximum_distance_possible_between_two_pixels
#     print(normalized_distance_between_superpixels)


def get_superpixels_by_pixels(labels_slic, pixels):
    # Find the superpixels containing the given pixels
    result_superpixels = set()
    for each_pixel in pixels:
        result_superpixels.add(labels_slic[each_pixel[1], each_pixel[0]])
    return result_superpixels


# def get_neighboring_superpixels(labels_slic, superpixel_label):
#     # Find the coordinates of all pixels belonging to the given superpixel
#     coords_superpixel = np.argwhere(labels_slic == superpixel_label)

#     # Define the 8-connected neighborhood for each pixel
#     neighborhood = [(i, j) for i in range(-1, 2)
#                     for j in range(-1, 2) if (i != 0 or j != 0)]

#     # Collect neighboring superpixels
#     neighboring_superpixels = set()
#     for (y, x) in coords_superpixel:
#         for (dy, dx) in neighborhood:
#             yy, xx = y + dy, x + dx
#             if 0 <= yy < labels_slic.shape[0] and 0 <= xx < labels_slic.shape[1]:
#                 if labels_slic[yy, xx] not in neighboring_superpixels:
#                     neighboring_superpixels.add(labels_slic[yy, xx])
#                     neighborhood_neigh_superpixel = np.argwhere(
#                         labels_slic == superpixel_label)
#                     for (u, w) in neighborhood_neigh_superpixel:
#                         for (dy, dx) in neighborhood:
#                             uu, ww = u + dy, w + dx
#                             if 0 <= uu < labels_slic.shape[0] and 0 <= ww < labels_slic.shape[1]:
#                                 neighboring_superpixels.add(
#                                     labels_slic[uu, ww])

#     return neighboring_superpixels
def similarity_coefficient_calculator_and_value_returner4(superpixel_indices, image, cluster_label, cluster_label1, superpixel_centroids):
    superpixel_histogram_normalized = compute_normalized_histogram4(
        image, superpixel_indices, cluster_label)
    superpixel_histogram_normalized1 = compute_normalized_histogram4(
        image, superpixel_indices, cluster_label1)

    # Measure of Similarity : Harmonic mean of the (sum root(pR * p'R), sum root(pG * p'G), sum root(pB * p'B))
    individual_similarity_coefficient = []
    i = 0
    lambda_parameter = 0.2
    num_bins = (len(superpixel_histogram_normalized[0]))
    for j in range(len(superpixel_histogram_normalized)):
        sum = 0
        for bin in range(num_bins):
            sum = sum + \
                superpixel_histogram_normalized[j][bin] * \
                superpixel_histogram_normalized1[j][bin]
            if bin != num_bins-1 and bin != 0:
                sum = sum + \
                    lambda_parameter*((superpixel_histogram_normalized[j][bin] *
                                       superpixel_histogram_normalized1[j][bin+1]) + (superpixel_histogram_normalized[j][bin] *
                                                                                      superpixel_histogram_normalized1[j][bin-1]))
            if bin == 0:
                sum = sum + lambda_parameter*(superpixel_histogram_normalized[j][bin] *
                                              superpixel_histogram_normalized1[j][bin+1])

        individual_similarity_coefficient.append(np.sqrt(sum))

    sum = 0
    for each_coefficient in individual_similarity_coefficient:
        sum = sum + (1/each_coefficient)

    similarity_index_between_superpixels = len(
        individual_similarity_coefficient)/sum
    return 100*similarity_index_between_superpixels





# def get_superpixel_information(labels_slic, neighbors_in_superpixels):
#     num_nodes = len(neighbors_in_superpixels)
#     for each_node in range(num_nodes):
#         neighboring_nodes = neighbors_in_superpixels[each_node]

def abc(labels_slic, image, cluster_label, cluster_label1, superpixel_centroids):
    return 100


def get_neighboring_superpixels_random(labels_slic, superpixel_indices, superpixel_label):
    # Find the coordinates of all pixels belonging to the given superpixel
    indices = superpixel_indices[superpixel_label]

    # Number of pixels in the superpixel
    num_pixels = len(indices[0])

    # Sample a subset of the indices to reduce complexity
    sample_size = int(np.sqrt(num_pixels))
    if sample_size < 1:
        sample_size = 1  # Ensure at least one pixel is sampled

    sampled_indices = np.random.choice(
        num_pixels, sample_size, replace=False)
    sampled_pixels = (
        indices[0][sampled_indices], indices[1][sampled_indices])

    # Collect neighboring superpixels
    neighboring_superpixels = set()

    length_to_see = num_superpixels_parameter // 2
    for y, x in zip(sampled_pixels[0], sampled_pixels[1]):
        neighboring_points = [
            [y + length_to_see, x + length_to_see], [y -
                                                     length_to_see, x + length_to_see],
            [y + length_to_see, x - length_to_see], [y -
                                                     length_to_see, x - length_to_see],
            [y, x + length_to_see], [y, x - length_to_see],
            [y + length_to_see, x], [y - length_to_see, x]
        ]
        for each_point in neighboring_points:
            yy, xx = each_point
            if 0 <= yy < labels_slic.shape[0] and 0 <= xx < labels_slic.shape[1]:
                label = labels_slic[yy, xx]
                if label != superpixel_label:
                    neighboring_superpixels.add(label)
    return neighboring_superpixels


def get_neighboring_superpixels_random_8(labels_slic, superpixel_indices, superpixel_label):
    # Find the coordinates of all pixels belonging to the given superpixel
    indices = superpixel_indices[superpixel_label]

    # Number of pixels in the superpixel
    num_pixels = len(indices[0])

    # Sample a subset of the indices to reduce complexity
    sample_size = int(np.sqrt(num_pixels))
    if sample_size < 1:
        sample_size = 1  # Ensure at least one pixel is sampled

    sampled_indices = np.random.choice(
        num_pixels, sample_size, replace=False)
    sampled_pixels = (
        indices[0][sampled_indices], indices[1][sampled_indices])

    # Collect neighboring superpixels
    neighboring_superpixels = set()

    length_to_see = num_superpixels_parameter // 2
    for y, x in zip(sampled_pixels[0], sampled_pixels[1]):
        if len(neighboring_superpixels) >= 8:
            break
        neighboring_points = [
            [y + length_to_see, x + length_to_see], [y -
                                                     length_to_see, x + length_to_see],
            [y + length_to_see, x - length_to_see], [y -
                                                     length_to_see, x - length_to_see],
            [y, x + length_to_see], [y, x - length_to_see],
            [y + length_to_see, x], [y - length_to_see, x]
        ]
        for each_point in neighboring_points:
            yy, xx = each_point
            if 0 <= yy < labels_slic.shape[0] and 0 <= xx < labels_slic.shape[1]:
                label = labels_slic[yy, xx]
                if label != superpixel_label:
                    neighboring_superpixels.add(label)
    return neighboring_superpixels


def double_fill_neighbors_6(neighbors_in_superpixels: list):
    num_superpixels = len(neighbors_in_superpixels)
    neighbors1 = [set() for _ in range(num_superpixels)]
    try:
        for i in range(num_superpixels):
            neighbors1[i] = set(neighbors_in_superpixels[i].copy())
            for each_neighbor in neighbors_in_superpixels[i]:
                neighbors1[i] = neighbors1[i].union(
                    set(neighbors_in_superpixels[each_neighbor]))
            neighbors1[i].discard(i)
    except Exception as e:
        print(e)
    return [list(neighbors) for neighbors in neighbors1]


def fill_neighbors_6(labels_slic, superpixel_indices, num_superpixels):
    neighbors_in_superpixels = [set() for _ in range(num_superpixels)]
    try:
        for i in range(num_superpixels):
            # if len(neighbors_in_superpixels[i]) <= 80:
            indices = superpixel_indices[i]

            # Number of pixels in the superpixel
            num_pixels = len(indices[0])

            # Sample a subset of the indices to reduce complexity
            sample_size = int(np.sqrt(num_pixels))
            if sample_size < 1:
                sample_size = 1  # Ensure at least one pixel is sampled

            # sampled_indices = np.random.choice(
            #     num_pixels, sample_size, replace=False)
            # sampled_pixels = (
            #     indices[0][sampled_indices], indices[1][sampled_indices])
            sampled_pixels = (indices[0], indices[1])

            # Collect neighboring superpixels
            neighboring_superpixels = set()
            num_superpixels_parameter = 13
            length_to_see = num_superpixels_parameter // 4
            for y, x in zip(sampled_pixels[0], sampled_pixels[1]):
                neighboring_points = [
                    [y + length_to_see, x + length_to_see], [y -
                                                             length_to_see, x + length_to_see],
                    [y + length_to_see, x - length_to_see], [y -
                                                             length_to_see, x - length_to_see],
                    [y, x + length_to_see], [y, x - length_to_see],
                    [y + length_to_see, x], [y - length_to_see, x]
                ]
                for each_point in neighboring_points:
                    yy, xx = each_point
                    if 0 <= yy < labels_slic.shape[0] and 0 <= xx < labels_slic.shape[1]:
                        label = labels_slic[yy, xx]
                        if label != i:
                            neighboring_superpixels.add(label)

            neighbors_in_superpixels[i] = neighbors_in_superpixels[i].union(
                neighboring_superpixels)
            for each_neighbor in neighboring_superpixels:
                neighbors_in_superpixels[each_neighbor].add(i)

    except Exception as e:
        print(e)
    return [list(neighbors) for neighbors in neighbors_in_superpixels]


# def fill_neighbors_6_optimized(labels_slic, superpixel_indices, num_superpixels):
#     """
#     Compute the adjacency list for superpixels in a 2D label image.
    
#     Args:
#         labels_slic (ndarray): H×W array of superpixel labels.
#         superpixel_indices (list of tuple): Not used in this implementation,
#             kept for compatibility.
#         num_superpixels (int): Number of superpixel labels (max label + 1).
    
#     Returns:
#         list of lists: neighbors[i] is a list of superpixel labels adjacent to i.
#     """
#     H, W = labels_slic.shape
#     # Initialize adjacency sets
#     adj = [set() for _ in range(num_superpixels)]

#     # Compare each pixel to its right neighbor
#     right_diff = labels_slic[:, :-1] != labels_slic[:, 1:]
#     ys, xs = np.nonzero(right_diff)
#     for y, x in zip(ys, xs):
#         a = labels_slic[y, x]
#         b = labels_slic[y, x + 1]
#         if a < num_superpixels and b < num_superpixels and a != b:
#             adj[a].add(b)
#             adj[b].add(a)

#     # Compare each pixel to its bottom neighbor
#     down_diff = labels_slic[:-1, :] != labels_slic[1:, :]
#     ys, xs = np.nonzero(down_diff)
#     for y, x in zip(ys, xs):
#         a = labels_slic[y, x]
#         b = labels_slic[y + 1, x]
#         if a < num_superpixels and b < num_superpixels and a != b:
#             adj[a].add(b)
#             adj[b].add(a)

#     # Convert sets to sorted lists for consistency
#     neighbors = [sorted(list(s)) for s in adj]
#     return neighbors

import numpy as np

def fill_neighbors_6_optimized1(labels_slic, superpixel_indices, num_superpixels):
    # Precompute constants and image dimensions
    num_superpixels_parameter = 13
    length_to_see = num_superpixels_parameter // 4
    h, w = labels_slic.shape[:2]
    
    # Precompute neighbor offsets as NumPy array
    offsets = np.array([
        (length_to_see, length_to_see),
        (-length_to_see, length_to_see),
        (length_to_see, -length_to_see),
        (-length_to_see, -length_to_see),
        (0, length_to_see),
        (0, -length_to_see),
        (length_to_see, 0),
        (-length_to_see, 0)
    ])
    
    neighbors_in_superpixels = [set() for _ in range(num_superpixels)]
    
    try:
        for i in range(num_superpixels):
            indices = superpixel_indices[i]
            rows, cols = indices[0], indices[1]
            num_pixels = len(rows)
            
            if num_pixels == 0:
                continue
                
            # Create all candidate neighbor coordinates
            neighbor_y = rows[:, None] + offsets[:, 0]
            neighbor_x = cols[:, None] + offsets[:, 1]
            
            # Flatten coordinate arrays
            all_y = neighbor_y.ravel()
            all_x = neighbor_x.ravel()
            
            # Validate coordinates in bulk
            valid_mask = (all_y >= 0) & (all_y < h) & (all_x >= 0) & (all_x < w)
            valid_y = all_y[valid_mask]
            valid_x = all_x[valid_mask]
            
            if len(valid_y) == 0:
                continue
                
            # Get labels at valid positions
            neighbor_labels = labels_slic[valid_y, valid_x]
            
            # Filter out current superpixel and get unique neighbors
            mask_non_i = neighbor_labels != i
            unique_neighbors = set(neighbor_labels[mask_non_i])
            
            # Update neighbor relationships
            neighbors_in_superpixels[i] |= unique_neighbors
            for nbr in unique_neighbors:
                neighbors_in_superpixels[nbr].add(i)
                
    except Exception as e:
        print(f"Error: {e}")
        
    return [list(neighbors) for neighbors in neighbors_in_superpixels]


def fill_neighbors_custom_offset(labels_slic, superpixel_indices, num_superpixels, length_to_see=3):
    """
    Finds neighboring superpixels using fixed-length offsets (not 4- or 8-connectivity).

    Args:
        labels_slic (ndarray): 2D array of superpixel labels.
        superpixel_indices (dict): {label: (row_indices, col_indices)}
        num_superpixels (int): Total number of superpixels.
        length_to_see (int): Offset step for neighbor probing.

    Returns:
        List of neighboring superpixels for each label.
    """
    h, w = labels_slic.shape
    neighbors_in_superpixels = [set() for _ in range(num_superpixels)]

    # Fixed directional offsets (your custom scheme)
    offsets = np.array([
        (length_to_see, length_to_see),
        (-length_to_see, length_to_see),
        (length_to_see, -length_to_see),
        (-length_to_see, -length_to_see),
        (0, length_to_see),
        (0, -length_to_see),
        (length_to_see, 0),
        (-length_to_see, 0)
    ], dtype=np.int32)

    for label, (rows, cols) in superpixel_indices.items():
        if len(rows) == 0:
            continue

        # Convert to array for vector ops
        rows = np.asarray(rows)
        cols = np.asarray(cols)

        # Apply each offset in batch
        for dy, dx in offsets:
            y = rows + dy
            x = cols + dx

            # Keep only valid pixel indices
            valid = (y >= 0) & (y < h) & (x >= 0) & (x < w)
            y = y[valid]
            x = x[valid]

            if len(y) == 0:
                continue

            neighbor_labels = labels_slic[y, x]
            neighbor_labels = neighbor_labels[neighbor_labels != label]

            unique_neighbors = np.unique(neighbor_labels)
            neighbors_in_superpixels[label].update(unique_neighbors)
            for nbr in unique_neighbors:
                neighbors_in_superpixels[nbr].add(label)

    return [list(nbrs) for nbrs in neighbors_in_superpixels]


def fill_neighbors_6_optimized(labels_slic, superpixel_indices, num_superpixels):
    # Precompute constants and image dimensions
    num_superpixels_parameter = 13
    length_to_see = num_superpixels_parameter // 4
    h, w = labels_slic.shape[:2]

    # Precompute neighbor offsets
    offsets = (
        (length_to_see, length_to_see),
        (-length_to_see, length_to_see),
        (length_to_see, -length_to_see),
        (-length_to_see, -length_to_see),
        (0, length_to_see),
        (0, -length_to_see),
        (length_to_see, 0),
        (-length_to_see, 0)
    )

    neighbors_in_superpixels = [set() for _ in range(num_superpixels)]
    try:
        for i in range(num_superpixels):
            indices = superpixel_indices[i]
            rows, cols = indices[0], indices[1]
            num_pixels = len(rows)
            neighboring_superpixels = set()

            # Process all pixels in the superpixel
            for idx in range(num_pixels):
                y, x = rows[idx], cols[idx]
                # Check all neighbor positions
                for dy, dx in offsets:
                    yy = y + dy
                    xx = x + dx
                    if 0 <= yy < h and 0 <= xx < w:
                        label = labels_slic[yy, xx]
                        if label != i:
                            neighboring_superpixels.add(label)

            # Update neighbor relationships
            neighbors_in_superpixels[i].update(neighboring_superpixels)
            for neighbor in neighboring_superpixels:
                neighbors_in_superpixels[neighbor].add(i)

    except Exception as e:
        print(e)

    return [list(neighbors) for neighbors in neighbors_in_superpixels]


def fill_neighbors_random_8(labels_slic, superpixel_indices, num_superpixels):
    neighbors_in_superpixels = [set() for _ in range(num_superpixels)]
    try:
        for i in range(num_superpixels):
            # if len(neighbors_in_superpixels[i]) <= 80:
            indices = superpixel_indices[i]

            # Number of pixels in the superpixel
            num_pixels = len(indices[0])

            # Sample a subset of the indices to reduce complexity
            sample_size = int(np.sqrt(num_pixels))
            if sample_size < 1:
                sample_size = 1  # Ensure at least one pixel is sampled

            sampled_indices = np.random.choice(
                num_pixels, sample_size, replace=False)
            sampled_pixels = (
                indices[0][sampled_indices], indices[1][sampled_indices])

            # Collect neighboring superpixels
            neighboring_superpixels = set()

            length_to_see = num_superpixels_parameter // 2
            for y, x in zip(sampled_pixels[0], sampled_pixels[1]):
                if len(neighboring_superpixels) >= 8:
                    break
                neighboring_points = [
                    [y + length_to_see, x + length_to_see], [y -
                                                             length_to_see, x + length_to_see],
                    [y + length_to_see, x - length_to_see], [y -
                                                             length_to_see, x - length_to_see],
                    [y, x + length_to_see], [y, x - length_to_see],
                    [y + length_to_see, x], [y - length_to_see, x]
                ]
                for each_point in neighboring_points:
                    yy, xx = each_point
                    if 0 <= yy < labels_slic.shape[0] and 0 <= xx < labels_slic.shape[1]:
                        label = labels_slic[yy, xx]
                        if label != i:
                            neighboring_superpixels.add(label)

            neighbors_in_superpixels[i] = neighbors_in_superpixels[i].union(
                neighboring_superpixels)
            for each_neighbor in neighboring_superpixels:
                neighbors_in_superpixels[each_neighbor].add(i)

    except Exception as e:
        print(e)
    return [list(neighbors) for neighbors in neighbors_in_superpixels]


def fill_neighbors_5(labels_slic, superpixel_indices, num_superpixels):
    neighbors_in_superpixels = [[] for _ in range(num_superpixels)]
    try:
        for i in range(num_superpixels):

            if (len(neighbors_in_superpixels[i]) <= 80):

                indices = superpixel_indices[i]

                # Number of pixels in the superpixel
                num_pixels = len(indices[0])

                # Sample a subset of the indices to reduce complexity
                sample_size = int(np.sqrt(num_pixels))
                if sample_size < 1:
                    sample_size = 1  # Ensure at least one pixel is sampled

                sampled_indices = np.random.choice(
                    num_pixels, sample_size, replace=False)
                sampled_pixels = [indices[0]
                                  [sampled_indices], indices[1][sampled_indices]]

                # Collect neighboring superpixels
                neighboring_superpixels = set()

                length_to_see = num_superpixels_parameter//2
                for (y, x) in sampled_pixels:
                    neighboring_points = [[y + length_to_see, x + length_to_see], [y - length_to_see, x + length_to_see],
                                          [y + length_to_see, x - length_to_see], [y -
                                                                                   length_to_see, x - length_to_see],
                                          [y, x+length_to_see], [y,
                                                                 x-length_to_see],
                                          [y+length_to_see, x], [y-length_to_see, x]]
                    for each_point in neighboring_points:
                        yy = each_point[0]
                        xx = each_point[1]
                        if 0 <= yy < labels_slic.shape[0] and 0 <= xx < labels_slic.shape[1]:
                            if labels_slic[yy, xx] not in neighboring_superpixels and labels_slic[yy, xx] != i:
                                neighboring_superpixels.add(
                                    labels_slic[yy, xx])

                neighbors_in_superpixels[i] = list(neighboring_superpixels)
                for each_neighbor in neighbors_in_superpixels:
                    neighbors_in_superpixels[each_neighbor].add(i)

    except Exception as e:
        print(e)
    return neighbors_in_superpixels


def fill_neighbors_4(label_slic, num_superpixels):
    neighbors_in_superpixels = [[] for _ in range(num_superpixels)]
    try:
        for i in range(num_superpixels):
            if (len(neighbors_in_superpixels[i]) <= 80):
                coords_superpixel = np.argwhere(
                    labels_slic == i)
                # Collect neighboring superpixels
                neighboring_superpixels = set()
                num_points_to_select = int(np.sqrt(len(coords_superpixel)))

            # Convert coords_superpixel to a list
            coords_superpixel_list = coords_superpixel.tolist()

            # Randomly select points from coords_superpixel_list
            points = random.sample(
                coords_superpixel_list, num_points_to_select)
            length_to_see = num_superpixels_parameter//2
            for (y, x) in points:
                neighboring_points = [[y + length_to_see, x + length_to_see], [y - length_to_see, x + length_to_see],
                                      [y + length_to_see, x - length_to_see], [y -
                                                                               length_to_see, x - length_to_see],
                                      [y, x+length_to_see], [y, x-length_to_see],
                                      [y+length_to_see, x], [y-length_to_see, x]]
                for each_point in neighboring_points:
                    yy = each_point[0]
                    xx = each_point[1]
                    if 0 <= yy < labels_slic.shape[0] and 0 <= xx < labels_slic.shape[1]:
                        if labels_slic[yy, xx] not in neighboring_superpixels and labels_slic[yy, xx] != i:
                            neighboring_superpixels.add(labels_slic[yy, xx])

            neighbors_in_superpixels[i] = list(neighboring_superpixels)

    except Exception as e:
        print(e)
    return neighbors_in_superpixels


'''
def get_neighboring_superpixels(labels_slic, superpixel_label):
    # Find the coordinates of all pixels belonging to the given superpixel
    coords_superpixel = np.argwhere(labels_slic == superpixel_label)
    # Collect neighboring superpixels
    neighboring_superpixels = set()

    # Assuming coords_superpixel is a list of coordinate tuples
    # For example, coords_superpixel = [(x1, y1), (x2, y2), ...]

    # Calculate the number of points to select (approximately one-third of the total)
    num_points_to_select = len(coords_superpixel) // 3

    # Convert coords_superpixel to a list
    coords_superpixel_list = coords_superpixel.tolist()

    # Randomly select points from coords_superpixel_list
    points = random.sample(coords_superpixel_list, num_points_to_select)
    length_to_see = num_superpixels_parameter//2
    for (y, x) in points:
        neighboring_points = [[y + length_to_see, x + length_to_see], [y - length_to_see, x + length_to_see],
                              [y + length_to_see, x - length_to_see], [y -
                                                                       length_to_see, x - length_to_see],
                              [y, x+length_to_see], [y, x-length_to_see],
                              [y+length_to_see, x], [y-length_to_see, x]]
        for each_point in neighboring_points:
            yy = each_point[0]
            xx = each_point[1]
            if 0 <= yy < labels_slic.shape[0] and 0 <= xx < labels_slic.shape[1]:
                if labels_slic[yy, xx] not in neighboring_superpixels and labels_slic[yy, xx] != superpixel_label:
                    neighboring_superpixels.add(labels_slic[yy, xx])
    # if (len(neighboring_superpixels) >= 5):
    #     print(f'{len(neighboring_superpixels)}')
    return neighboring_superpixels
    # # Find the coordinates of all pixels belonging to the given superpixel
    # coords_superpixel = np.argwhere(labels_slic == superpixel_label)

    # # Calculate the number of points to select (approximately one-third of the total)
    # num_points_to_select = len(coords_superpixel) // 3

    # # Randomly select points from coords_superpixel
    # selected_points = random.sample(
    #     coords_superpixel.tolist(), num_points_to_select)

    # neighboring_superpixels = set()
    # length_to_see = num_superpixels_parameter // 2
    # image_height, image_width = labels_slic.shape

    # for y, x in selected_points:
    #     neighboring_points = [[y + length_to_see, x + length_to_see], [y - length_to_see, x + length_to_see],
    #                           [y + length_to_see, x - length_to_see], [y -
    #                                                                    length_to_see, x - length_to_see],
    #                           [y, x+length_to_see], [y, x-length_to_see],
    #                           [y+length_to_see, x], [y-length_to_see, x]]
    #     for each_point in neighboring_points:
    #         yy = each_point[0]
    #         xx = each_point[1]
    #         if 0 <= yy < image_height and 0 <= xx < image_width and (yy != y or xx != x):
    #             label = labels_slic[yy, xx]
    #             neighboring_superpixels.add(label)
    # # Remove superpixel_label from neighboring_superpixels
    # neighboring_superpixels.discard(superpixel_label)

    # return neighboring_superpixels
    # print('Should be more')
    # similarity_coefficient_calculator(superpixel_histogram_normalized,
    #                                   superpixel_histogram_normalized1, cluster_label, cluster_label1)

    # print('Should be less')
    # similarity_coefficient_calculator(superpixel_histogram_normalized1,
    #                                   superpixel_histogram_normalized2, cluster_label1, cluster_label2)

    # neighboring_superpixels = get_neighboring_superpixels(
    #     labels_slic, cluster_label)
    # print("Neighboring superpixels of superpixel",
    #       superpixel_label, ":", neighboring_superpixels)

    # # Iterate over each neighboring superpixel
    # for each_neighbor in neighboring_superpixels:
    #     print(f'\n\nLabel: {each_neighbor}')
    #     similarity_coefficient_calculator(
    #         labels_slic, image, cluster_label, each_neighbor, superpixel_centroids)

    # def similarity_coefficient_calculator_and_value_returner(labels_slic, image, cluster_label, cluster_label1, superpixel_centroids):
    #     superpixel_histogram_normalized = compute_normalized_histogram(
    #         image, labels_slic, cluster_label)
    #     superpixel_histogram_normalized1 = compute_normalized_histogram(
    #         image, labels_slic, cluster_label1)

    #     # Measure of Similarity : Harmonic mean of the (sum root(pR * p'R), sum root(pG * p'G), sum root(pB * p'B))
    #     individual_similarity_coefficient = []
    #     i = 0
    #     lambda_parameter = 0.2
    #     num_bins = (len(superpixel_histogram_normalized[0]))
    #     for j in range(len(superpixel_histogram_normalized)):
    #         sum = 0
    #         for bin in range(num_bins):
    #             sum = sum + \
    #                 superpixel_histogram_normalized[j][bin] * \
    #                 superpixel_histogram_normalized1[j][bin]
    #             if bin != num_bins-1 and bin != 0:
    #                 sum = sum + \
    #                     lambda_parameter*((superpixel_histogram_normalized[j][bin] *
    #                                        superpixel_histogram_normalized1[j][bin+1]) + (superpixel_histogram_normalized[j][bin] *
    #                                                                                       superpixel_histogram_normalized1[j][bin-1]))
    #             if bin == 0:
    #                 sum = sum + lambda_parameter*(superpixel_histogram_normalized[j][bin] *
    #                                               superpixel_histogram_normalized1[j][bin+1])

    #         individual_similarity_coefficient.append(np.sqrt(sum))
    #     # print(individual_similarity_coefficient)

    #     sum = 0
    #     for each_coefficient in individual_similarity_coefficient:
    #         sum = sum + (1/each_coefficient)

    #     similarity_index_between_superpixels = len(
    #         individual_similarity_coefficient)/sum
    #     # print(similarity_index_between_superpixels)

    #     # sum = 0
    #     # # for each_coefficient in individual_similarity_coefficient:
    #     # sum = 0.21*individual_similarity_coefficient[0] + 0.72 * \
    #     #     individual_similarity_coefficient[1] + \
    #     #     0.07*individual_similarity_coefficient[2]
    #     # print(sum)
    #
    #     # print(normalized_distance_between_superpixels)
    #     return 100*similarity_index_between_superpixels
'''

'''

Functions for getting neighboring superpixels

'''


def get_neighboring_superpixels(labels_slic, superpixel_label):
    # Find the coordinates of all pixels belonging to the given superpixel
    coords_superpixel = np.argwhere(labels_slic == superpixel_label)

    # Define the 8-connected neighborhood for each pixel
    neighborhood = [(i, j) for i in range(-1, 2)
                    for j in range(-1, 2) if (i != 0 or j != 0)]

    # Collect neighboring superpixels
    neighboring_superpixels = set()
    for (y, x) in coords_superpixel:
        for (dy, dx) in neighborhood:
            yy, xx = y + dy, x + dx
            if 0 <= yy < labels_slic.shape[0] and 0 <= xx < labels_slic.shape[1]:
                if labels_slic[yy, xx] not in neighboring_superpixels:
                    neighboring_superpixels.add(labels_slic[yy, xx])
                    neighborhood_neigh_superpixel = np.argwhere(
                        labels_slic == superpixel_label)
                    for (u, w) in neighborhood_neigh_superpixel:
                        for (dy, dx) in neighborhood:
                            uu, ww = u + dy, w + dx
                            if 0 <= uu < labels_slic.shape[0] and 0 <= ww < labels_slic.shape[1]:
                                neighboring_superpixels.add(
                                    labels_slic[uu, ww])

    return neighboring_superpixels


def fill_neighbors(label_slic, num_superpixels):
    neighbors_in_superpixels = [[] for _ in range(num_superpixels)]

    try:
        # if (len(neighbors_in_superpixels[i]) <= 7):
        for superpixel_label in range(num_superpixels):
            coords_superpixel = np.argwhere(labels_slic == superpixel_label)

            # Define the 8-connected neighborhood for each pixel
            neighborhood = [(i, j) for i in range(-1, 2)
                            for j in range(-1, 2) if (i != 0 or j != 0)]

            # Collect neighboring superpixels
            neighboring_superpixels = set()
            for (y, x) in coords_superpixel:
                for (dy, dx) in neighborhood:
                    yy, xx = y + dy, x + dx
                    if 0 <= yy < labels_slic.shape[0] and 0 <= xx < labels_slic.shape[1]:
                        if labels_slic[yy, xx] not in neighboring_superpixels:
                            neighboring_superpixels.add(labels_slic[yy, xx])
                            neighborhood_neigh_superpixel = np.argwhere(
                                labels_slic == superpixel_label)
                            for (u, w) in neighborhood_neigh_superpixel:
                                for (dy, dx) in neighborhood:
                                    uu, ww = u + dy, w + dx
                                    if 0 <= uu < labels_slic.shape[0] and 0 <= ww < labels_slic.shape[1]:
                                        neighboring_superpixels.add(
                                            labels_slic[uu, ww])
            neighbors_in_superpixels[superpixel_label] = list(
                neighboring_superpixels)

    except Exception as e:
        print(e)
    return neighbors_in_superpixels


# def fill_neighbors(label_slic, num_superpixels):
#     neighbors_in_superpixels = [[] for _ in range(num_superpixels)]
#     try:
#         for i in range(num_superpixels):
#             if (len(neighbors_in_superpixels[i]) <= 7):
#                 coords_superpixel = np.argwhere(
#                     labels_slic == i)
#                 # Collect neighboring superpixels
#                 neighboring_superpixels = set()
#                 num_points_to_select = len(coords_superpixel) // 3

#             # Convert coords_superpixel to a list
#             coords_superpixel_list = coords_superpixel.tolist()

#             # Randomly select points from coords_superpixel_list
#             points = random.sample(
#                 coords_superpixel_list, num_points_to_select)
#             length_to_see = num_superpixels_parameter//2
#             for (y, x) in points:
#                 neighboring_points = [[y + length_to_see, x + length_to_see], [y - length_to_see, x + length_to_see],
#                                       [y + length_to_see, x - length_to_see], [y -
#                                                                                length_to_see, x - length_to_see],
#                                       [y, x+length_to_see], [y, x-length_to_see],
#                                       [y+length_to_see, x], [y-length_to_see, x]]
#                 for each_point in neighboring_points:
#                     yy = each_point[0]
#                     xx = each_point[1]
#                     if 0 <= yy < labels_slic.shape[0] and 0 <= xx < labels_slic.shape[1]:
#                         if labels_slic[yy, xx] not in neighboring_superpixels and labels_slic[yy, xx] != i:
#                             neighboring_superpixels.add(labels_slic[yy, xx])
#             neighbors_in_superpixels[i] = list(neighboring_superpixels)

#     except Exception as e:
#         print(e)
#     return neighbors_in_superpixels


# def get_neighboring_superpixels(labels_slic, superpixel_label):
#     # Find the coordinates of all pixels belonging to the given superpixel
#     coords_superpixel = np.argwhere(labels_slic == superpixel_label)
#     # Collect neighboring superpixels
#     neighboring_superpixels = set()

#     # Assuming coords_superpixel is a list of coordinate tuples
#     # For example, coords_superpixel = [(x1, y1), (x2, y2), ...]

#     '''
#     # Calculate the number of points to select (approximately one-third of the total)
#     num_points_to_select = len(coords_superpixel) // 3

#     # Convert coords_superpixel to a list
#     coords_superpixel_list = coords_superpixel.tolist()

#     # Randomly select points from coords_superpixel_list
#     points = random.sample(coords_superpixel_list, num_points_to_select)
#     length_to_see = num_superpixels_parameter//2
#     for (y, x) in points:
#         neighboring_points = [[y + length_to_see, x + length_to_see], [y - length_to_see, x + length_to_see],
#                               [y + length_to_see, x - length_to_see], [y -
#                                                                        length_to_see, x - length_to_see],
#                               [y, x+length_to_see], [y, x-length_to_see],
#                               [y+length_to_see, x], [y-length_to_see, x]]
#         for each_point in neighboring_points:
#             yy = each_point[0]
#             xx = each_point[1]
#             if 0 <= yy < labels_slic.shape[0] and 0 <= xx < labels_slic.shape[1]:
#                 if labels_slic[yy, xx] not in neighboring_superpixels and labels_slic[yy, xx] != superpixel_label:
#                     neighboring_superpixels.add(labels_slic[yy, xx])
#     # if (len(neighboring_superpixels) >= 5):
#     #     print(f'{len(neighboring_superpixels)}')
#     return neighboring_superpixels


#     '''
#     # Find the coordinates of all pixels belonging to the given superpixel
#     coords_superpixel = np.argwhere(labels_slic == superpixel_label)

#     # Calculate the number of points to select (approximately one-third of the total)
#     num_points_to_select = len(coords_superpixel) // 3

#     # Randomly select points from coords_superpixel
#     selected_points = random.sample(
#         coords_superpixel.tolist(), num_points_to_select)

#     neighboring_superpixels = set()
#     length_to_see = num_superpixels_parameter // 2
#     image_height, image_width = labels_slic.shape

#     for y, x in selected_points:
#         neighboring_points = [[y + length_to_see, x + length_to_see], [y - length_to_see, x + length_to_see],
#                               [y + length_to_see, x - length_to_see], [y -
#                                                                        length_to_see, x - length_to_see],
#                               [y, x+length_to_see], [y, x-length_to_see],
#                               [y+length_to_see, x], [y-length_to_see, x]]
#         for each_point in neighboring_points:
#             yy = each_point[0]
#             xx = each_point[1]
#             if 0 <= yy < image_height and 0 <= xx < image_width and (yy != y or xx != x):
#                 label = labels_slic[yy, xx]
#                 neighboring_superpixels.add(label)
#     # Remove superpixel_label from neighboring_superpixels
#     neighboring_superpixels.discard(superpixel_label)

#     return neighboring_superpixels
def color_superpixels(image,superpixel_indices):
    colored_image = image.copy()
    num_superpixels = labels_slic.max() + 1
    for label in range(num_superpixels):
        indices = superpixel_indices[label]
        # Generate a random color for the current superpixel
        color = [random.randint(0, 255) for _ in range(3)]  # RGB color

        # Color all the pixels belonging to the current superpixel with the generated color
        for row_index, col_index in zip(indices[0], indices[1]):
            colored_image[row_index, col_index] = color
    plt.imshow(colored_image)
    plt.title('Superpixels Colored with Random Colors')
    plt.show()
def quantize_rgb(pixel,num_bins=16):
    """Quantize RGB values from 0-255 to 0-15."""
    return pixel // (256 // num_bins)
def compute_histogram(indices,image):
    """Compute the 4096-bin color histogram for an image region."""
    # Initialize the histogram with 4096 bins (16x16x16)
    histogram = np.zeros(4096)

    # print(indices)

    # exit()
    # Process each pixel in the region
    for row_index, col_index in zip(indices[0],indices[1]):
        each_pixel = image[row_index,col_index]
        num_bins = 16
        # Quantize the RGB values
        r, g, b = quantize_rgb(each_pixel,num_bins)
        r = np.int32(r)
        g = np.int32(g)
        b = np.int32(b)
        # Compute the bin index
        bin_index = r * num_bins * num_bins + g * num_bins + b

        # Increment the corresponding bin in the histogram
        histogram[bin_index] += 1

    # Normalize the histogram
    histogram /= (len(indices[0]))

    return histogram


# def compute_normalized_histogram4(image, superpixel_indices, superpixel_label):
#     sample_fraction = 0.1
#     indices = superpixel_indices[superpixel_label]

#     # Number of pixels in the superpixel
#     num_pixels = len(indices[0])

#     # Sample a subset of the indices to reduce complexity
#     sample_size = int(num_pixels * sample_fraction)
#     if sample_size < 1:
#         sample_size = 1  # Ensure at least one pixel is sampled

#     sampled_indices = np.random.choice(num_pixels, sample_size, replace=False)
#     sampled_pixels = image[indices[0]
#                            [sampled_indices], indices[1][sampled_indices]]
#     # Calculate histogram for each channel separately
#     hist_r, _ = np.histogram(sampled_pixels[:, 0], bins=8, range=(0, 255))
#     hist_g, _ = np.histogram(sampled_pixels[:, 1], bins=8, range=(0, 255))
#     hist_b, _ = np.histogram(sampled_pixels[:, 2], bins=8, range=(0, 255))

#     # Normalize histograms for each channel
#     hist_r_normalized = hist_r / sample_size + 0.001
#     hist_g_normalized = hist_g / sample_size + 0.001
#     hist_b_normalized = hist_b / sample_size + 0.001

#     # Concatenate normalized histograms for all channels
#     superpixel_histogram_normalized = [
#         hist_r_normalized, hist_g_normalized, hist_b_normalized]

#     return superpixel_histogram_normalized


def compute_modified_normalized_histogram(image, superpixel_indices, superpixel_label):
    indices = superpixel_indices[superpixel_label]

    # Number of pixels in the superpixel
    num_pixels = len(indices[0])

    return compute_histogram(indices,image)

def bhattacharyya_similarity_coefficient_calculator(superpixel_indices, image, cluster_label, cluster_label1):
    # color_superpixels(image,superpixel_indices)
    # exit()
    superpixel_histogram_normalized = compute_modified_normalized_histogram(
        image, superpixel_indices, cluster_label)
    superpixel_histogram_normalized1 = compute_modified_normalized_histogram(
        image, superpixel_indices, cluster_label1)
    
    return np.sum(np.sqrt(superpixel_histogram_normalized * superpixel_histogram_normalized1))


def bhattacharyya_similarity_coefficient_calculator_optimized(histograms, cluster_label, cluster_label1):
    # color_superpixels(image,superpixel_indices)
    # exit()
    superpixel_histogram_normalized = histograms[cluster_label]
    superpixel_histogram_normalized1 = histograms[cluster_label1]
    
    return np.sum(np.sqrt(superpixel_histogram_normalized * superpixel_histogram_normalized1))

def similarity_coefficient_calculator_and_value_returner(labels_slic, image, cluster_label, cluster_label1, superpixel_centroids):
    superpixel_histogram_normalized = compute_normalized_histogram(
        image, labels_slic, cluster_label)
    superpixel_histogram_normalized1 = compute_normalized_histogram(
        image, labels_slic, cluster_label1)

    # Measure of Similarity : Harmonic mean of the (sum root(pR * p'R), sum root(pG * p'G), sum root(pB * p'B))
    individual_similarity_coefficient = []
    i = 0
    lambda_parameter = 0.2
    num_bins = (len(superpixel_histogram_normalized[0]))
    for j in range(len(superpixel_histogram_normalized)):
        sum = 0
        for bin in range(num_bins):
            sum = sum + \
                superpixel_histogram_normalized[j][bin] * \
                superpixel_histogram_normalized1[j][bin]
            if bin != num_bins-1 and bin != 0:
                sum = sum + \
                    lambda_parameter*((superpixel_histogram_normalized[j][bin] *
                                       superpixel_histogram_normalized1[j][bin+1]) + (superpixel_histogram_normalized[j][bin] *
                                                                                      superpixel_histogram_normalized1[j][bin-1]))
            if bin == 0:
                sum = sum + lambda_parameter*(superpixel_histogram_normalized[j][bin] *
                                              superpixel_histogram_normalized1[j][bin+1])

        individual_similarity_coefficient.append(np.sqrt(sum))
    # print(individual_similarity_coefficient)

    sum = 0
    for each_coefficient in individual_similarity_coefficient:
        sum = sum + (1/each_coefficient)

    similarity_index_between_superpixels = len(
        individual_similarity_coefficient)/sum
    # print(similarity_index_between_superpixels)

    # sum = 0
    # # for each_coefficient in individual_similarity_coefficient:
    # sum = 0.21*individual_similarity_coefficient[0] + 0.72 * \
    #     individual_similarity_coefficient[1] + \
    #     0.07*individual_similarity_coefficient[2]
    # print(sum)
    '''
    centroid_first_superpixel = (
        superpixel_centroids[cluster_label][0], superpixel_centroids[cluster_label][1])
    centroid_second_superpixel = (
        superpixel_centroids[cluster_label1][0], superpixel_centroids[cluster_label1][1])
    euclidean_distance_between_two_superpixels = np.sqrt(np.square(
        centroid_first_superpixel[0]-centroid_second_superpixel[0]) + np.square(centroid_first_superpixel[1]-centroid_second_superpixel[1]))
    # print(centroid_first_superpixel)
    # print(centroid_second_superpixel)
    # print(euclidean_distance_between_two_superpixels)

    image_height = image.shape[0]
    image_width = image.shape[1]
    # print(f'image_height: {image_height}, image_width: {image_width}')
    maximum_distance_possible_between_two_pixels = np.sqrt(
        np.square(image_height)+np.square(image_width))
    normalized_distance_between_superpixels = euclidean_distance_between_two_superpixels / \
        maximum_distance_possible_between_two_pixels
    '''
    # print(normalized_distance_between_superpixels)
    return 100*similarity_index_between_superpixels


def graph_maker(num_superpixels):
    G = nx.Graph()
    for i in range(num_superpixels):
        G.add_node(i)
    return G


class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, u):
        if self.parent[u] != u:
            self.parent[u] = self.find(self.parent[u])
        return self.parent[u]

    def union(self, u, v):
        root_u = self.find(u)
        root_v = self.find(v)
        if root_u != root_v:
            if self.rank[root_u] > self.rank[root_v]:
                self.parent[root_v] = root_u
            elif self.rank[root_u] < self.rank[root_v]:
                self.parent[root_u] = root_v
            else:
                self.parent[root_v] = root_u
                self.rank[root_u] += 1


def max_spanning_tree_with_cut(graph, node_a, node_b):
    # Step 1: Create the maximum spanning tree
    mst = nx.maximum_spanning_tree(graph)

    # Step 2: Convert node names to indices for Union-Find
    node_map = {node: idx for idx, node in enumerate(mst.nodes())}
    idx_a = node_map[node_a]
    idx_b = node_map[node_b]

    # Step 3: Sort edges by weight in ascending order
    edges = [(data['weight'], u, v) for u, v, data in mst.edges(data=True)]
    heapq.heapify(edges)

    # Step 4: Initialize Union-Find
    uf = UnionFind(len(mst.nodes()))

    # Initially, all nodes are in the same component
    for u, v in mst.edges():
        uf.union(node_map[u], node_map[v])

    # Step 5: Find the minimum cost edge to cut
    min_cut_edge = None

    while edges:
        weight, u, v = heapq.heappop(edges)
        if uf.find(idx_a) != uf.find(idx_b):
            min_cut_edge = (u, v, weight)
            break
        uf.union(node_map[u], node_map[v])
        mst.remove_edge(u, v)

    # Output the results
    if min_cut_edge:
        u, v, edge_cost = min_cut_edge
        print(f"Minimum cost edge to cut: ({u}, {v}) with cost {edge_cost}")
        mst.remove_edge(u, v)
    else:
        print("No valid edge found to cut that separates 'a' and 'b'.")

    return mst


def helper5(maximum_spanning_tree, foreground_label='foreground', background_label='background'):
    min_cut_edge = None
    min_cut_weight = float('inf')
    sorted_edges = sorted(maximum_spanning_tree.edges(
        data=True), key=lambda x: x[2]['weight'])

    # Create a mapping from nodes to indices for the Union-Find structure
    node_map = {node: idx for idx, node in enumerate(
        maximum_spanning_tree.nodes())}
    uf = UnionFind(len(node_map))

    # Union the edges in the MST initially
    for u, v, data in sorted_edges:
        uf.union(node_map[u], node_map[v])

    # Initially check if 'foreground_node' and 'background_node' are connected
    if uf.find(node_map[foreground_label]) == uf.find(node_map[background_label]):
        # Iterate over the sorted edges
        for edge in sorted_edges:
            source, target, data = edge
            weight = data['weight']

            # Temporarily remove the edge from the MST by performing union-find operations
            uf_temp = UnionFind(len(node_map))
            uf_temp.parent = uf.parent[:]
            uf_temp.rank = uf.rank[:]
            uf_temp.parent[node_map[source]] = node_map[source]
            uf_temp.parent[node_map[target]] = node_map[target]

            # Check if the foreground and background nodes are now disconnected
            if uf_temp.find(node_map[foreground_label]) != uf_temp.find(node_map[background_label]):
                if weight < min_cut_weight:
                    min_cut_edge = (source, target)
                    min_cut_weight = weight
                break

    print(min_cut_edge)
    print(min_cut_weight)

    # Remove the smallest edge that separates foreground and background into components
    if min_cut_edge is not None:
        maximum_spanning_tree.remove_edge(min_cut_edge[0], min_cut_edge[1])

    return maximum_spanning_tree


def helper6(critical_edges, maximum_spanning_tree):
    sorted_edges = sorted(critical_edges, key=lambda x: x[2])
    min_cut_edge = sorted_edges[0]
    print(min_cut_edge)
    if min_cut_edge is not None:
        maximum_spanning_tree.remove_edge(min_cut_edge[0], min_cut_edge[1])


def helper(maximum_spanning_tree):
    s = time.time()
    min_cut_edge = None
    min_cut_weight = float('inf')
    sorted_edges = sorted(maximum_spanning_tree.edges(
        data=True), key=lambda x: x[2]['weight'])
    is_done = False
    i = 0
    t = time.time()
    print(t-s)
    print(len(sorted_edges))
    for edge in sorted_edges:
        i = i+1
        if not is_done:
            # Remove the current edge from the maximum spanning tree
            temp_tree = copy.deepcopy(maximum_spanning_tree)
            source, target, _ = edge  # Unpack the edge tuple
            temp_tree.remove_edge(source, target)

            components = list(nx.connected_components(temp_tree))
            foreground_component = None
            background_component = None
            for component in components:
                if 'foreground' in component:
                    foreground_component = component
                elif 'background' in component:
                    background_component = component

            if foreground_component is not None and background_component is not None \
                    and foreground_component != background_component \
                    and 'foreground' not in background_component \
                    and 'background' not in foreground_component:
                if len(components) == 2:
                    weight = maximum_spanning_tree[edge[0]][edge[1]]['weight']
                    print('$$$$$$$'+str(weight) + '  '+str(edge))
                    if weight < min_cut_weight:
                        min_cut_edge = edge
                        min_cut_weight = weight
                    is_done = True
            # # Check if foreground and background are separated into two components
            # components = list(nx.connected_components(temp_tree))
            # if {'foreground', 'background'} in components and len(components) == 2:
            #     weight = maximum_spanning_tree[edge[0]][edge[1]]['weight']
            #     print('$$$$$$$'+str(weight) + '  '+str(edge))
            #     if weight < min_cut_weight:
            #         min_cut_edge = edge
            #         min_cut_weight = weight
        else:
            break
    print(i)

    print(min_cut_edge)
    print(min_cut_weight)

    # Remove the smallest edge that separates foreground and background into components
    if min_cut_edge is not None:
        maximum_spanning_tree.remove_edge(min_cut_edge[0], min_cut_edge[1])


def helper1(image, maximum_spanning_tree, label_slic):
    image_height, image_width = image.shape[:2]
    segmented_image = np.zeros((image_height, image_width, 3), dtype=np.uint8)
    components = list(nx.connected_components(maximum_spanning_tree))
    # Assign colors to the connected components
    for component in components:
        color = (0, 0, 255) if 'foreground' in component else (0, 255, 0)
        print(color)
        if 'foreground' in component:
            print('$$$foreground$$$$')
        else:
            print('$$$background$$$$')
        # Assign the color to all nodes in the component
        for node in component:
            # print(node)
            if node != 'foreground' and node != 'background':
                x = node
                segmented_image[label_slic == x] = color

    # Display the segmented image
    plt.imshow(segmented_image)
    plt.axis('off')
    plt.show()


def helper2(image, maximum_spanning_tree, label_slic):
    image_height, image_width = image.shape[:2]
    # segmented_image = np.zeros((image_height, image_width, 3), dtype=np.uint8)
    segmented_image = np.copy(image)
    components = list(nx.connected_components(maximum_spanning_tree))
    # Assign colors to the connected components
    for component in components:
        color = (255, 255, 255)
        foreground = False
        if 'foreground' in component:
            print('$$$foreground$$$$')
            foreground = True
        else:
            print('$$$background$$$$')
        # Assign the color to all nodes in the component
        for node in component:
            # print(node)
            if node != 'foreground' and node != 'background':
                x = node
                if not foreground:
                    segmented_image[label_slic == x] = color

    # Display the segmented image
    segmented_image = cv2.cvtColor(segmented_image, cv2.COLOR_BGR2RGB)
    plt.imshow(segmented_image)
    plt.axis('off')
    plt.show()


def helper3(image, maximum_spanning_tree, label_slic):
    image_height, image_width = image.shape[:2]
    # segmented_image = np.zeros((image_height, image_width, 3), dtype=np.uint8)
    segmented_image = np.copy(image)
    components = list(nx.connected_components(maximum_spanning_tree))
    # Assign colors to the connected components
    for component in components:
        color = (255, 255, 255)
        foreground = False
        if 'foreground' in component:
            print('$$$foreground$$$$')
            foreground = True
        else:
            print('$$$background$$$$')
        # Assign the color to all nodes in the component
        for node in component:
            # print(node)
            if node != 'foreground' and node != 'background':
                x = node
                if not foreground:
                    coords_superpixel = np.argwhere(
                        labels_slic == x)
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
    plt.show()
    return segmented_image


def helper31(image, maximum_spanning_tree, labels_slic):
    alpha = 0.5
    color = np.array([0, 0, 255], dtype=np.float32)
    image_height, image_width = image.shape[:2]
    # segmented_image = np.zeros((image_height, image_width, 3), dtype=np.uint8)
    segmented_image = np.copy(image)
    components = list(nx.connected_components(maximum_spanning_tree))
    # Assign colors to the connected components
    for component in components:
        foreground = False
        if 'foreground' in component:
            print('$$$foreground$$$$')
            foreground = True
        else:
            print('$$$background$$$$')
        # Assign the color to all nodes in the component
        for node in component:
            # print(node)
            if node != 'foreground' and node != 'background':
                x = node
                if not foreground:
                    segmented_image[labels_slic == x] = (alpha * color +
                                                         (1 - alpha) * segmented_image[labels_slic == x].astype(np.float32)).astype(np.uint8)
                    # coords_superpixel = np.argwhere(
                    #     labels_slic == x)
                    # for (y, x) in coords_superpixel:
                    #     try:
                    #         segmented_image[y, x] = apply_translucent_scribble(
                    #             segmented_image, x, y, [0, 0, 255], alpha=128)
                    #     except Exception as e:
                    #         # print(e)
                    #         pass

    # Display the segmented image
    # segmented_image = cv2.cvtColor(segmented_image, cv2.COLOR_BGR2RGB)

    # plt.imshow(segmented_image)
    # # plt.imsave('flower_segmentation_dataset/segmented/1.png',
    # #            segmented_image)
    # plt.axis('off')
    # plt.show()
    return segmented_image


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


def apply_translucent_to_image(image, alpha=128):
    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            image[y, x] = apply_translucent_scribble(
                image, x, y, [0, 0, 255], alpha)
    plt.title("translucent color application: blue")
    plt.imshow(image)
    plt.axis("off")
    plt.show()


def generate_mask(image, maximum_spanning_tree, label_slic):
    segmented_image = np.copy(image)
    components = list(nx.connected_components(maximum_spanning_tree))
    # Assign colors to the connected components
    for component in components:
        foreground = False
        if 'foreground' in component:
            print('$$$foreground$$$$')
            foreground = True
        else:
            print('$$$background$$$$')
        # Assign the color to all nodes in the component
        for node in component:
            # print(node)
            if node != 'foreground' and node != 'background':
                x = node
                if not foreground:
                    coords_superpixel = np.argwhere(
                        labels_slic == x)
                    for (y, x) in coords_superpixel:
                        segmented_image[y, x] = [0, 0, 0]
                        # apply_translucent_scribble(
                        #     segmented_image, x, y, [0, 0, 255], alpha=128)
                else:
                    coords_superpixel = np.argwhere(
                        labels_slic == x)
                    for (y, x) in coords_superpixel:
                        try:
                            segmented_image[y, x] = [255, 255, 255]
                        except Exception as e:
                            # print(e)
                            pass

    # Display the segmented image
    segmented_image = cv2.cvtColor(segmented_image, cv2.COLOR_BGR2RGB)
    plt.imshow(segmented_image)
    # plt.imsave('images/visvesvaraya_kaustubh_1_mask.jpg',
    #            segmented_image)
    # plt.imsave('images/Mona_Lisa_mask.jpg',
    #            segmented_image)
    plt.axis('off')
    plt.show()
    return segmented_image


def generate_mask1(image, maximum_spanning_tree, label_slic):
    segmented_image = np.copy(image)
    components = list(nx.connected_components(maximum_spanning_tree))
    # Assign colors to the connected components
    for component in components:
        foreground = False
        if 'foreground' in component:
            print('$$$foreground$$$$')
            foreground = True
        else:
            print('$$$background$$$$')
        # Assign the color to all nodes in the component
        for node in component:
            # print(node)
            if node != 'foreground' and node != 'background':
                x = node
                if not foreground:
                    segmented_image[label_slic == node] = [0, 0, 0]
                else:
                    segmented_image[label_slic == node] = [255, 255, 255]

    # Display the segmented image
    segmented_image = cv2.cvtColor(segmented_image, cv2.COLOR_BGR2RGB)
    plt.imshow(segmented_image)
    # plt.imsave('images/visvesvaraya_kaustubh_1_mask.jpg',
    #            segmented_image)
    # plt.imsave('images/Mona_Lisa_mask.jpg',
    #            segmented_image)
    plt.axis('off')
    # plt.show()
    plt.close()
    return segmented_image


def find_critical_edges(mst, a, b):

    # Step 3: Find the path between 'a' and 'b' in the MST
    path = nx.shortest_path(mst, source=a, target=b, weight='weight')

    # Step 4: Identify the critical edges on the path
    critical_edges = []
    for i in range(len(path) - 1):
        u = path[i]
        v = path[i + 1]
        weight = mst[u][v]['weight']
        critical_edges.append((u, v, weight))

    return critical_edges


def find_critical_edges_modified(mst, a, b):
    """
    Since `mst` is a tree, there is exactly one simple path from a to b.
    We do an unweighted DFS to recover that path, then collect edge weights.
    """
    # Build a lightweight adjacency list
    adj = {u: list(mst[u]) for u in mst.nodes()}

    # DFS to find parents
    parent = {a: None}
    stack = [a]
    while stack and b not in parent:
        u = stack.pop()
        for v in adj[u]:
            if v not in parent:
                parent[v] = u
                stack.append(v)

    # Reconstruct the unique path a → … → b
    if b not in parent:
        raise ValueError(f"No path found between {a} and {b} in the MST")
    path = []
    node = b
    while node is not None:
        path.append(node)
        node = parent[node]
    path.reverse()

    # Gather the edges (u, v, weight) along that path
    critical_edges = [
        (u, v, mst[u][v]['weight'])
        for u, v in zip(path, path[1:])
    ]
    return critical_edges


def compute_laplace_histograms(image, superpixel_indices, num_bins=8, alpha=1.0):
    """
    For each superpixel label, compute a Laplace-smoothed normalized histogram
    of each channel. Returns an array of shape (num_superpixels, 3, num_bins).
    
    Args:
        image: H×W×3 uint8 image array.
        superpixel_indices: dict[label] -> (rows, cols) of that superpixel.
        num_bins: number of bins per channel.
        alpha: Laplace smoothing constant.
    """
    num_superpixels = max(superpixel_indices.keys()) + 1
    histograms = np.zeros((num_superpixels, 3, num_bins), dtype=np.float32)

    # Build each superpixel’s histogram
    for lbl, (rows, cols) in superpixel_indices.items():
        pixels = image[rows, cols]  # shape (n_pixels, 3)
        n = pixels.shape[0]
        for c in range(3):
            # Raw histogram
            h, _ = np.histogram(pixels[:, c], bins=num_bins, range=(0, 255))
            # Laplace smoothing + normalize
            histograms[lbl, c] = (h + alpha) / (n + alpha * num_bins)

    return histograms


def compute_normalized_histograms(image, superpixel_indices, num_bins=8):
    """
    Compute normalized histograms (without Laplace smoothing) for all superpixels.
    Returns an array of shape (num_superpixels, 3, num_bins).

    Args:
        image: H×W×3 uint8 image array.
        superpixel_indices: dict[label] -> (rows, cols) for each superpixel.
        num_bins: Number of histogram bins per channel (default 8).

    Returns:
        histograms: ndarray of shape (num_superpixels, 3, num_bins), containing
                    normalized histograms with small constant offset added.
    """
    num_superpixels = max(superpixel_indices.keys()) + 1
    histograms = np.zeros((num_superpixels, 3, num_bins), dtype=np.float32)

    for label, (rows, cols) in superpixel_indices.items():
        sampled_pixels = image[rows, cols]  # shape: (n_pixels, 3)
        num_pixels = sampled_pixels.shape[0]
        if num_pixels == 0:
            continue  # avoid divide-by-zero

        for c in range(3):  # R, G, B
            hist, _ = np.histogram(
                sampled_pixels[:, c], bins=num_bins, range=(0, 255))
            histograms[label, c] = hist / num_pixels + \
                0.001  # normalization with offset

    return histograms

def sips_similarity(histograms, i, j, lambda_param=0.2):
    """
    Compute the SIPS similarity between superpixels i and j.

    Args:
        histograms: array of shape (num_superpixels, 3, num_bins)
        i, j: indices of the two superpixels.
        lambda_param: weight for neighboring-bin contribution.

    Returns:
        sips: float similarity in [0,1].
    """
    hi = histograms[i]   # shape (3, B)
    hj = histograms[j]   # shape (3, B)
    kernel = np.array([lambda_param, 1.0, lambda_param], dtype=hi.dtype)

    sim_channels = []
    # for each color channel
    for c in range(3):
        # include ±1-bin contributions via 1D convolution
        w1 = convolve1d(hi[c], kernel, mode='constant', cval=0.0)
        w2 = convolve1d(hj[c], kernel, mode='constant', cval=0.0)
        # sum of sqrt products
        sim_c = np.sqrt(w1 * w2).sum()
        sim_channels.append(sim_c)

    sim_arr = np.array(sim_channels)
    # harmonic mean across channels
    sips = len(sim_arr) / np.sum(1.0 / sim_arr)
    return sips


def similarity_coefficient_calculator_and_value_returner4_from_histograms(
    histograms, cluster_label, cluster_label1, lambda_parameter=0.2
):
    """
    Computes the SIPS similarity between two superpixels using precomputed histograms.

    Args:
        histograms: dict[label] -> [hist_r, hist_g, hist_b], each hist of shape (num_bins,)
        cluster_label: index of first superpixel
        cluster_label1: index of second superpixel
        lambda_parameter: bin-neighbor smoothing factor (default 0.2)

    Returns:
        similarity index scaled by 100
    """
    hist1 = histograms[cluster_label]
    hist2 = histograms[cluster_label1]

    num_channels = len(hist1)
    num_bins = len(hist1[0])
    individual_similarity_coefficient = []

    for channel in range(num_channels):
        h1 = hist1[channel]
        h2 = hist2[channel]
        channel_sum = 0.0

        for b in range(num_bins):
            main_term = h1[b] * h2[b]
            left_term = h1[b] * h2[b - 1] if b > 0 else 0
            right_term = h1[b] * h2[b + 1] if b < num_bins - 1 else 0
            channel_sum += main_term + \
                lambda_parameter * (left_term + right_term)

        individual_similarity_coefficient.append(np.sqrt(channel_sum))

    # Harmonic mean of the similarity coefficients
    inv_sum = sum(1.0 / coeff for coeff in individual_similarity_coefficient)
    similarity_index_between_superpixels = num_channels / inv_sum

    return 100.0 * similarity_index_between_superpixels
