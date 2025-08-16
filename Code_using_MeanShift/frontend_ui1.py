import os
import json
import matplotlib.pyplot as plt
import cv2
import numpy as np
import numpy as np
from fun import get_superpixels_by_pixels
# from parameters_and_data import image_rgb, scribbling_dimension, fpoints, bpoints, scribbled_img_path
# from parameters_and_data import image_rgb, fpoints, bpoints, labels_slic, result, num_superpixels, image_path, seg_img_path, mask_img_path, num_superpixels_parameter
# from parameters_and_data import image_rgb, scribbling_dimension, fpoints, bpoints, visualization_image_path, scribbled_img_path, ground_truth_path


visualization_image = np.zeros_like(image_rgb)
image1 = np.copy(image_rgb)
# def calculate_intensity(pixel):
#     return 0.2126 * pixel[0] + 0.7152 * pixel[1] + 0.0722 * pixel[2]


# for x in range(image_rgb.shape[0]):
#     for y in range(image_rgb.shape[1]):
#         print(image_rgb[x][y])
# Create a 10x10 image with RGB channels
image = np.zeros((image_rgb.shape[0], image_rgb.shape[1], 3), dtype=np.uint8)
# image_with_intensities = np.zeros(
#     (image_rgb.shape[0], image_rgb.shape[1], 1), dtype=np.uint8)


# Function to initialize variables

# background_clicked_points = []
# foreground_clicked_points = []
# scribble_color = [0, 0, 255]


# def init(image):
#     global is_scribbling
#     is_scribbling = False
#     global points_list
#     points_list = []
#     global image_width
#     global image_height
#     image_width = image.shape[0]
#     image_height = image.shape[1]
#     global scribbling_length
#     scribbling_length = scribbling_dimension

# # Function to handle mouse press events


# def on_press(event, clicked_points):
#     global points_list
#     global is_scribbling
#     if event.xdata is not None and event.ydata is not None:
#         x = int(round(event.xdata))
#         y = int(round(event.ydata))
#         is_scribbling = True
#         # print('press' + str(x) + ' ' + str(y) + ' ' + str(clicked_points))
#         points_list.append([x, y])
#         # scribble1(x, y, clicked_points)

# # Function to handle mouse move events (while button is pressed)


# def on_move(event, clicked_points):
#     global points_list
#     if is_scribbling:
#         if event.xdata is not None and event.ydata is not None:
#             x = int(round(event.xdata))
#             y = int(round(event.ydata))
#             # print('move' + str(x) + ' ' + str(y) + ' ' + str(clicked_points))
#             points_list.append([x, y])
#             # scribble1(x, y, clicked_points)

# # Function to handle mouse release events


# def on_release(event, clicked_points):
#     global is_scribbling
#     is_scribbling = False
#     global points_list
#     scribble2(points_list, clicked_points)
#     points_list = []
#     plt.imshow(image)
#     plt.draw()


# # Function to perform scribbling (coloring)

# def scribble2(points_list, clicked_points):
#     for each_point in points_list:
#         scribble3(each_point[0], each_point[1], clicked_points)


# def scribble3(x, y, clicked_points):
#     global image_width
#     global image_height
#     global scribbling_length
#     pixels = []
#     for i in range(-1*scribbling_length, scribbling_length+1):
#         for j in range(-1*scribbling_length, scribbling_length+1):
#             if (i != 0 or j != 0):
#                 if (((x+i) >= 0 and (x+i) < image_height) and ((y+j) >= 0 and (y+j) < image_width)):
#                     pixels.append([x+i, y+j])
#     for each_pixel in pixels:
#         scribble(each_pixel[0], each_pixel[1], clicked_points)


# def scribble1(x, y, clicked_points):
#     pixels = []
#     for i in range(-5, 6):
#         for j in range(-5, 6):
#             if (i != 0 or j != 0):
#                 pixels.append([x+i, y+j])
#     for each_pixel in pixels:
#         scribble(each_pixel[0], each_pixel[1], clicked_points)
#     plt.imshow(image)
#     plt.draw()


# def scribble(x, y, clicked_points):
#     pixel = image[y, x]  # Get pixel of clicked point
#     # intensity = 0.2126 * pixel[0] + 0.7152 * pixel[1] + 0.0722 * pixel[2]

#     pixel_color = image[y, x]
#     if (x, y) not in [(point[0], point[1]) for point in clicked_points] and \
#         (pixel_color.tolist() != [0, 0, 255]) and \
#             (pixel_color.tolist() != [255, 0, 0]):
#         # Store clicked point if conditions are met
#         # clicked_points.append((x, y, intensity))
#         clicked_points.append((x, y))

#         image[y, x] = scribble_color


# # Initialize variables
# init(image)

# for i in range(image_rgb.shape[0]):
#     for j in range(image_rgb.shape[1]):
#         image[i, j] = image_rgb[i][j]

# # print("foreground")
# # Display the image and connect the mouse events
# fig, ax = plt.subplots()
# ax.set_title("Click foreground points")
# ax.imshow(image)
# ax.axis('off')
# fig.canvas.mpl_connect('button_press_event', lambda event: on_press(
#     event, foreground_clicked_points))
# fig.canvas.mpl_connect('button_release_event', lambda event: on_release(
#     event, foreground_clicked_points))
# fig.canvas.mpl_connect('motion_notify_event', lambda event: on_move(
#     event, foreground_clicked_points))

# plt.show()

# # print("background")
# scribble_color = [0, 255, 0]

# fig1, ax1 = plt.subplots()
# ax1.set_title("Click background points")
# ax1.imshow(image)
# ax1.axis('off')
# fig1.canvas.mpl_connect('button_press_event',
#                         lambda event: on_press(event, background_clicked_points))
# fig1.canvas.mpl_connect('button_release_event', lambda event: on_release(
#     event, background_clicked_points))
# fig1.canvas.mpl_connect('motion_notify_event', lambda event: on_move(
#     event, background_clicked_points))
# plt.show()

# # print("foregound pixels: ")
# # print(foreground_clicked_points)

# # print("background_pixels:")
# # print(background_clicked_points)

# for (x, y) in foreground_clicked_points:
#     fpoints.append((x, y))
# for (x, y) in background_clicked_points:
#     bpoints.append((x, y))
# plt.imshow(image)
# plt.imsave(scribbled_img_path + '_1.png',
#            image)
# plt.axis('off')
# plt.show()

# file_path_1 = f"{image_num}.json"
# data1 = {
#     'f': fpoints,
#     'b': bpoints
# }
# with open(path_to_add+ 'scribbled/images/'+file_path_1, 'w') as json_file:
#     json.dump(data1, json_file)

# Create a new image to visualize the original superpixel and its neighbors


# Taking data from other scribbles

# Load the scribble image
if scribbles_from == 'one-cut':
    amoe_scribble_image = cv2.imread(onecut_path)
elif scribbles_from == 'amoe':
    amoe_scribble_image = cv2.imread(amoe_path)
elif scribbles_from == 'ssn-cut':
    amoe_scribble_image = cv2.imread(ssncut_path)
elif scribbles_from == 'our_markers_images250' or scribbles_from == 'our_markers_images250_graphcut' or scribbles_from == 'our_markers_images250_slic' or scribbles_from == 'our_markers_images250_maxst_bh' or scribbles_from == 'our_markers_images250_graphcut_bh' or scribbles_from == 'our_markers_images250_slic_bh' or scribbles_from == 'images250_analyzing_neighborhood_0_1':
    amoe_scribble_image = cv2.imread(our_markers_images250_path)
else:
    amoe_scribble_image = cv2.imread(grabcut_path)

# Keep a copy of the original for visualization
amoe_scribble_image_rgb = cv2.cvtColor(amoe_scribble_image, cv2.COLOR_BGR2RGB)  # for plotting

# Define scribble colors (BGR for OpenCV)
green_bgr = [0, 255, 0]
blue_bgr = [255, 0, 0]

# Create binary masks for green and blue scribbles
green_mask = np.all(amoe_scribble_image == green_bgr, axis=-1).astype(np.uint8)
blue_mask = np.all(amoe_scribble_image == blue_bgr, axis=-1).astype(np.uint8)

# Fix: missing colon
if scribbles_from == 'one-cut':
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
elif scribbles_from == 'amoe':
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
elif scribbles_from == 'ssn-cut':
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
else:
    kernel = None  # In case you want to skip erosion for other cases

# Erode to thin the scribbles
if scribbles_from != 'grabcut' and kernel is not None:
    green_mask_thin = cv2.erode(green_mask, kernel, iterations=1)
    blue_mask_thin = cv2.erode(blue_mask, kernel, iterations=1)

    # Remove original scribbles and apply thinned ones
    thinned_image = amoe_scribble_image.copy()
    thinned_image[green_mask == 1] = [0, 0, 0]
    thinned_image[blue_mask == 1] = [0, 0, 0]
    thinned_image[green_mask_thin == 1] = green_bgr
    thinned_image[blue_mask_thin == 1] = blue_bgr

    # Convert to RGB for matplotlib display
    thinned_image_rgb = cv2.cvtColor(thinned_image, cv2.COLOR_BGR2RGB)

    # Plot both original and thinned images
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.title("Original Scribbles")
    plt.imshow(amoe_scribble_image_rgb)
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title("Thinned Scribbles")
    plt.imshow(thinned_image_rgb)
    plt.axis('off')
    plt.tight_layout()
    # plt.show()
    plt.close()

    # Extract points from thinned image
    green_rgb = green_bgr[::-1]  # Convert BGR to RGB
    blue_rgb = blue_bgr[::-1]

    green_mask_final = np.all(thinned_image_rgb == green_rgb, axis=-1)
    blue_mask_final = np.all(thinned_image_rgb == blue_rgb, axis=-1)

else:
    # Note: amoe_scribble_image is still in BGR
    green_mask_final = np.all(amoe_scribble_image == green_bgr, axis=-1)
    blue_mask_final = np.all(amoe_scribble_image == blue_bgr, axis=-1)

# Load the grabcut box image
box_image = cv2.imread(box_path)

# Create a mask for white pixels (BGR)
white_mask = np.all(box_image == [255, 255, 255], axis=-1)

# Extract coordinates of white pixels as (x, y)
white_points = np.column_stack(np.where(white_mask))[:, ::-1]  # (x, y)

# Extract coordinates as (x, y)
foreground_points = np.column_stack(np.where(green_mask_final))[:, ::-1]
background_points = np.column_stack(np.where(blue_mask_final))[:, ::-1]

# Combine blue scribble points with white box points for background
background_points = np.vstack([background_points, white_points])

print(f"Foreground points: {foreground_points.shape[0]}")
print(f"Background points: {background_points.shape[0]}")

# Get superpixels for each point
foreground_superpixels = get_superpixels_by_pixels(labels_slic, list(foreground_points))
background_superpixels = get_superpixels_by_pixels(labels_slic, list(background_points))

# Remove overlap between superpixels
set1 = set(int(i) for i in foreground_superpixels)
set2 = set(int(i) for i in background_superpixels)
common = set1 & set2

foreground_superpixels = list(set1 - common)
background_superpixels = list(set2 - common)

file_path_1 = f"{image_num}.json"

if not os.path.exists(f'scribbled/{scribbles_from}/'):
    # Create the folder
    os.makedirs(f'scribbled/{scribbles_from}/')
    print(f"Folder 'scribbled/{scribbles_from}/' created.")
else:
    print(f"Folder 'scribbled/{scribbles_from}/' already exists.")
    # Combine the lists into a dictionary


# data = {
#     "list1": foreground_points,
#     "list2": background_points
# }

# # Store the lists into the JSON file
# with open(path_to_add + f'scribbled/{scribbles_from}/'+file_path_1, 'w') as json_file:
#     json.dump(data, json_file)

# Color the foreground superpixels (green)
# visualization_image = np.zeros_like(image_rgb)
for each_label in foreground_superpixels:
    visualization_image[labels_slic == each_label] = [
        0, 255, 0]  # Green color
# Color the background superpixel (blue)
for each_label in background_superpixels:
    visualization_image[labels_slic == each_label] = [0, 0, 255]
# Convert BGR to RGB for displaying with matplotlib
visualization_image_rgb = cv2.cvtColor(visualization_image, cv2.COLOR_BGR2RGB)

# Plot the visualization image
plt.imshow(visualization_image_rgb)
# plt.imsave('visualising_superpixels/3.png', visualization_image_rgb)
plt.axis('off')
# plt.show()
plt.close()
# Overlay contour image on the original image
# overlay_image = cv2.addWeighted(image1, 0.5, visualization_image, 0.5, 0)
# overlay_image = cv2.add(image1, visualization_image)

# Convert visualization image to grayscale
visualization_gray = cv2.cvtColor(visualization_image, cv2.COLOR_BGR2GRAY)

# Create a mask where non-zero pixels in visualization_gray are 255
_, mask = cv2.threshold(visualization_gray, 1, 255, cv2.THRESH_BINARY)
# Invert the mask
mask_inv = cv2.bitwise_not(mask)

# Apply the mask to image1 using bitwise AND
image1_masked = cv2.bitwise_and(image1, image1, mask=mask_inv)

# Add the masked image1 and visualization image
overlay_image = cv2.add(image1_masked, visualization_image)
# # Apply the mask to image1 using bitwise AND
# image1_masked = cv2.bitwise_and(image1, image1, mask=mask)

# # Add the masked image1 and visualization image
# overlay_image = cv2.add(image1_masked, visualization_image)
# Convert BGR to RGB for displaying with matplotlib
overlay_image_rgb = (overlay_image)

# Plot the overlaid image
plt.imshow(overlay_image_rgb)
plt.imsave(visualization_image_path, overlay_image_rgb)
plt.axis('off')
# plt.show()
plt.close()
