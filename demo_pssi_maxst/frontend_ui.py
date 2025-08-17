import os
import json
import matplotlib.pyplot as plt
import cv2
import numpy as np
import numpy as np
from fun import get_superpixels_by_pixels
# from parameters_and_data import image_rgb, scribbling_dimension, fpoints, bpoints, scribbled_img_path, image_num
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

background_clicked_points = []
foreground_clicked_points = []
scribble_color = [0, 0, 255]


def init(image):
    global is_scribbling
    is_scribbling = False
    global points_list
    points_list = []
    global image_width
    global image_height
    image_width = image.shape[0]
    image_height = image.shape[1]
    global scribbling_length
    scribbling_length = scribbling_dimension

# Function to handle mouse press events


def on_press(event, clicked_points):
    global points_list
    global is_scribbling
    if event.xdata is not None and event.ydata is not None:
        x = int(round(event.xdata))
        y = int(round(event.ydata))
        is_scribbling = True
        # print('press' + str(x) + ' ' + str(y) + ' ' + str(clicked_points))
        points_list.append([x, y])
        # scribble1(x, y, clicked_points)

# Function to handle mouse move events (while button is pressed)


def on_move(event, clicked_points):
    global points_list
    if is_scribbling:
        if event.xdata is not None and event.ydata is not None:
            x = int(round(event.xdata))
            y = int(round(event.ydata))
            # print('move' + str(x) + ' ' + str(y) + ' ' + str(clicked_points))
            points_list.append([x, y])
            # scribble1(x, y, clicked_points)

# Function to handle mouse release events


def on_release(event, clicked_points):
    global is_scribbling
    is_scribbling = False
    global points_list
    scribble2(points_list, clicked_points)
    points_list = []
    plt.imshow(image)
    plt.draw()


# Function to perform scribbling (coloring)

def scribble2(points_list, clicked_points):
    for each_point in points_list:
        scribble3(each_point[0], each_point[1], clicked_points)


def scribble3(x, y, clicked_points):
    global image_width
    global image_height
    global scribbling_length
    pixels = []
    for i in range(-1*scribbling_length, scribbling_length+1):
        for j in range(-1*scribbling_length, scribbling_length+1):
            if (i != 0 or j != 0):
                if (((x+i) >= 0 and (x+i) < image_height) and ((y+j) >= 0 and (y+j) < image_width)):
                    pixels.append([x+i, y+j])
    for each_pixel in pixels:
        scribble(each_pixel[0], each_pixel[1], clicked_points)


def scribble1(x, y, clicked_points):
    pixels = []
    for i in range(-5, 6):
        for j in range(-5, 6):
            if (i != 0 or j != 0):
                pixels.append([x+i, y+j])
    for each_pixel in pixels:
        scribble(each_pixel[0], each_pixel[1], clicked_points)
    plt.imshow(image)
    plt.draw()


def scribble(x, y, clicked_points):
    pixel = image[y, x]  # Get pixel of clicked point
    # intensity = 0.2126 * pixel[0] + 0.7152 * pixel[1] + 0.0722 * pixel[2]

    pixel_color = image[y, x]
    if (x, y) not in [(point[0], point[1]) for point in clicked_points] and \
        (pixel_color.tolist() != [0, 0, 255]) and \
            (pixel_color.tolist() != [255, 0, 0]):
        # Store clicked point if conditions are met
        # clicked_points.append((x, y, intensity))
        clicked_points.append((x, y))

        image[y, x] = scribble_color


# Initialize variables
init(image)

for i in range(image_rgb.shape[0]):
    for j in range(image_rgb.shape[1]):
        image[i, j] = image_rgb[i][j]

# print("foreground")
# Display the image and connect the mouse events
fig, ax = plt.subplots()
ax.set_title("Click foreground points")
ax.imshow(image)
ax.axis('off')
fig.canvas.mpl_connect('button_press_event', lambda event: on_press(
    event, foreground_clicked_points))
fig.canvas.mpl_connect('button_release_event', lambda event: on_release(
    event, foreground_clicked_points))
fig.canvas.mpl_connect('motion_notify_event', lambda event: on_move(
    event, foreground_clicked_points))

plt.show()

# print("background")
scribble_color = [0, 255, 0]

fig1, ax1 = plt.subplots()
ax1.set_title("Click background points")
ax1.imshow(image)
ax1.axis('off')
fig1.canvas.mpl_connect('button_press_event',
                        lambda event: on_press(event, background_clicked_points))
fig1.canvas.mpl_connect('button_release_event', lambda event: on_release(
    event, background_clicked_points))
fig1.canvas.mpl_connect('motion_notify_event', lambda event: on_move(
    event, background_clicked_points))
plt.show()

# print("foregound pixels: ")
# print(foreground_clicked_points)

# print("background_pixels:")
# print(background_clicked_points)

for (x, y) in foreground_clicked_points:
    fpoints.append((x, y))
for (x, y) in background_clicked_points:
    bpoints.append((x, y))

os.makedirs('scribbles', exist_ok=True)

plt.imshow(image)
plt.imsave('scribbles/' + str(image_num) + '_scribbled.png',
           image)
plt.axis('off')
plt.show()

file_path_1 = f"{image_num}.json"
data1 = {
    'f': fpoints,
    'b': bpoints
}
with open(path_to_add+ 'scribbled/images/'+file_path_1, 'w') as json_file:
    json.dump(data1, json_file)

# Convert original RGB image to BGR
image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

# Make a black copy of the image (same shape, all zeros)
image_bgr = np.zeros_like(image_bgr)

# Mark foreground pixels as green: (0, 255, 0)
for (x, y) in foreground_clicked_points:
    image_bgr[int(y), int(x)] = [0, 255, 0]  # BGR

# Mark background pixels as blue: (255, 0, 0)
for (x, y) in background_clicked_points:
    image_bgr[int(y), int(x)] = [255, 0, 0]  # BGR

# Save the modified image as BMP
scribble_bmp_path = 'scribbles/' + str(image_num) + '_marker.bmp'
cv2.imwrite(scribble_bmp_path, image_bgr)
print(f"Saved BMP image with scribbles at: {scribble_bmp_path}")

# Create a new image to visualize the original superpixel and its neighbors

# # Load the grabcut box image
# grabcut_box_image = cv2.imread(box_path)

# # Create a mask for white pixels (BGR)
# white_mask = np.all(grabcut_box_image == [255, 255, 255], axis=-1)

# # Extract coordinates of white pixels as (x, y)
# white_points = np.column_stack(np.where(white_mask))[:, ::-1]  # (x, y)

# bpoints = np.vstack([bpoints, white_points])

foreground_superpixels = get_superpixels_by_pixels(labels_slic, list(fpoints))
background_superpixels = get_superpixels_by_pixels(labels_slic, list(bpoints))


# if not os.path.exists('scribbled/flowers/'):
#     # Create the folder
#     os.makedirs('scribbled/flowers/')
#     print(f"Folder {'scribbled/flowers/'} created.")
# else:
#     print(f"Folder {'scribbled/flowers/'} already exists.")
    # Combine the lists into a dictionary

list1 = [int(i) for i in foreground_superpixels]
list2 = [int(i) for i in background_superpixels]
data = {
    "list1": list1,
    "list2": list2
}

# Store the lists into the JSON file
# with open(path_to_add+ 'scribbled/flowers/'+file_path_1, 'w') as json_file:
#     json.dump(data, json_file)

# Color the foreground superpixels (green)
for each_label in foreground_superpixels:
    visualization_image[labels_slic == each_label] = [
        0, 255, 0]  # Green color
# Color the background superpixel (blue)
for each_label in background_superpixels:
    visualization_image[labels_slic == each_label] = [0, 0, 255]
# Convert BGR to RGB for displaying with matplotlib
visualization_image_rgb = cv2.cvtColor(visualization_image, cv2.COLOR_BGR2RGB)
os.makedirs('visualising_superpixels', exist_ok=True)

# Plot the visualization image
plt.imshow(visualization_image_rgb)
# plt.imsave('visualising_superpixels/'+ str(image_num) + '.png', visualization_image_rgb)
plt.axis('off')
plt.show()

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
plt.imsave('visualising_superpixels/' +
           str(image_num) + '.png', overlay_image_rgb)
plt.axis('off')
plt.show()
