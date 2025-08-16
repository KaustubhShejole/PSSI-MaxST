# from test_on_10_images import image_path
import time
from skimage import graph
from skimage import segmentation, color
import cv2
import numpy as np
import matplotlib.pyplot as plt


import os

# Define the parent directory
parent_dir = "./../../results"

# List of directories to create inside the parent directory
directories = [
    "visualising_superpixels",
    "visualising_superpixels/flowers",
    "analysis",
    "analysis/neighborhood",
    "masks",
    "masks/flowers",
    "masks1",
    "masks1/flowers",
    "only_segmentations",
    "only_segmentations/1",
    "only_segmentations/2",
    "only_segmentations/3",
    "scribbled_images",
    "scribbled_images/flowers",
    "scribbled_images1",
    "scribbled_images1/flowers",
    "segmentation_results",
    "segmentation_results/flowers",
    "segmentation_results1",
    "segmentation_results1/flowers",
    "superpixels_scribbled",
    "superpixels_scribbled/flowers",
    "superpixels_visualization",
    "superpixels_visualization/flowers",
    "text_results"
]

# # Loop over the directories and create them under the parent directory
# for directory in directories:
#     dir_path = os.path.join(parent_dir, directory)
#     if not os.path.exists(dir_path):
#         os.makedirs(dir_path)
#         print(f"Created directory: {dir_path}")
#     else:
#         print(f"Directory already exists: {dir_path}")

# dir_path = "./../../" +"scribbled/flowers"
# if not os.path.exists(dir_path):
#     os.makedirs(dir_path)
#     print(f"Created {dir_path}")
# else:
#     print(f"Directory {dir_path} already exists")


scribbling_dimension = 1
fpoints = []
bpoints = []

# num_superpixels_parameter = 13
# compactness = 10.0

path_to_add_to_get_image = "../../image_data/imagesgrabcut/"
path_to_add = "../../"

superpixel_img_path = path_to_add + 'results/' + \
    'superpixels_visualization/flowers/' + image_num + '.jpg'

mask_img_path = path_to_add + 'results/'+'masks/flowers/' + image_num
mask_img_path2 = path_to_add + 'results/masks1/flowers/' + image_num + '_2_'
seg_img_path2 = path_to_add + 'results/segmentation_results1/flowers/' + image_num + '_2_'

only_segmentation_path = path_to_add+ 'results/only_segmentations'
markers_save_path = "../../image_data/markers_ours_images250/" + image_num

image_path = path_to_add_to_get_image + "images/" + image_num + '.jpg'
# image_path = path_to_add_to_get_image + "/images/" + image_num + '.bmp'
scribbled_img_path = path_to_add + 'results/'+'scribbled_images/flowers/' + image_num
scribbled_img_path2 = path_to_add + 'results/'+'scribbled_images1/flowers/' + image_num + '_2_'

visualization_image_path = path_to_add + 'results/' + \
    'visualising_superpixels/flowers/'+ image_num + '.png'
seg_img_path = path_to_add + 'results/'+'segmentation_results/flowers/' + image_num
image_rgb = cv2.imread(image_path)
num_pixels = image_rgb.shape[0] * image_rgb.shape[1]


ground_truth_path = path_to_add_to_get_image + "GT/" + image_num + '.bmp' # + '.png'
num_superpixels_parameter = int(np.sqrt(np.sqrt(num_pixels/2)/2))

amoe_path = path_to_add_to_get_image + "amoemarker/" + f'{image_num}_marker.bmp'
onecut_path = path_to_add_to_get_image + "one-cut/" + f'{image_num}_marker.bmp'
ssncut_path = path_to_add_to_get_image + \
    "ssncutmarker/" + f'{image_num}_marker.bmp'
our_markers_images250_path = path_to_add_to_get_image + \
    "markers_ours_images250/" + f'{image_num}_marker.bmp'
grabcut_path = path_to_add_to_get_image + "markers/" + f'{image_num}_marker.bmp'
box_path = path_to_add_to_get_image + "box/" + f'{image_num}_box.bmp'
# grabcut_box_path = path_to_add_to_get_image + "box/" + f'{image_num}_box.bmp'
'''
if num_pixels >= 600*800:
    num_superpixels_parameter = num_superpixels_parameter
    scribbling_dimension *= 2
print(f'Superpixel parameter={num_superpixels_parameter}')

'''
# image_rgb = cv2.imread(
#     'images/Mona_Lisa.jpg')
image_lab = cv2.cvtColor(image_rgb, cv2.COLOR_BGR2Lab)
image_rgb = cv2.cvtColor(image_rgb, cv2.COLOR_BGR2RGB)
# Convert the image to Lab color space
image = np.copy(image_rgb)
original_image = np.copy(image_rgb)
print(f'{image_lab.shape[1]} * {image_lab.shape[0]}')


height, width = image.shape[:2]

# # Compute region_size for ~1000 superpixels
# target_superpixels = 1000
region_size = 13 #int(np.sqrt((height * width) / target_superpixels))
region_size = max(region_size, 1)  # avoid zero

s = time.time()
# Apply SLIC algorithm
# Create SLIC superpixels
slic = cv2.ximgproc.createSuperpixelSLIC(
    image_lab,
    algorithm=cv2.ximgproc.SLICO,
    region_size=region_size,
    ruler=10.0  # compactness; tweak between 5–20 if needed
)
slic.iterate(10)

# Retrieve the labels of the superpixels
labels_slic = slic.getLabels()

num_superpixels = slic.getNumberOfSuperpixels()
t = time.time()
segment_generation_time = (t - s)
print(t-s)
print(f'number of superpixels is equal to {num_superpixels}')

# superpixels = [[] for _ in range(num_superpixels)]
# for i in range(labels_slic.shape[0]):
#     for j in range(labels_slic.shape[1]):
#         superpixel_label = labels_slic[i, j]
#         # Store pixel coordinates and values
#         superpixels[superpixel_label].append(((i, j), image[i, j]))

# # Find the centroid of each superpixel
# superpixel_centroids = []
# for superpixel_label in range(num_superpixels):
#     sp = superpixels[superpixel_label]
#     if sp:
#         # Calculate mean position of pixels to find centroid
#         sp_array = np.array([coord for coord, _ in sp])
#         centroid = np.mean(sp_array, axis=0)
#         # Include pixel value at centroid
#         superpixel_centroids.append(
#             [int(centroid[0]), int(centroid[1]), sp[0][1]])


# Create a mask for superpixel boundaries
mask = slic.getLabelContourMask()


mask_slic_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
result = cv2.addWeighted(image, 0.5, mask_slic_rgb, 0.5, 0)
# # Overlay the mask on the original image
# segmented_image = cv2.bitwise_and(image, image, mask=mask)

# # Convert BGR to RGB for displaying with matplotlib
# segmented_image_rgb = cv2.cvtColor(segmented_image, cv2.COLOR_BGR2RGB)

# Display the result using matplotlib
plt.imshow(result)
plt.imsave(superpixel_img_path, result)
plt.axis('off')
# plt.show()
plt.close()

# s = time.time()
# Apply mean shift segmentation using quickshift
# segments = segmentation.quickshift(
#     image_lab, kernel_size=3, max_dist=6, ratio=0.5)
# segments = segmentation.quickshift(
#     image_lab,
#     kernel_size=2.5,      # ↓ smaller kernel gives finer segments
#     max_dist=5,         # ↓ smaller max_dist increases sensitivity to color/texture
#     ratio=0.5           # ↓ lower ratio favors color over spatial proximity
# )


# h = image_lab.shape[0]
# w = image_lab.shape[1]
# area = h * w

# if area <= 160000:       # ~< 400x400
#     segments = segmentation.quickshift(
#         image_lab,
#         kernel_size=2.25,      # ↓ smaller kernel gives finer segments
#         max_dist=3,         # ↓ smaller max_dist increases sensitivity to color/texture
#         ratio=0.5           # ↓ lower ratio favors color over spatial proximity
#     )
# elif area <= 300000:     # ~< 550x550
#     segments = segmentation.quickshift(
#         image_lab,
#         kernel_size=2.5,      # ↓ smaller kernel gives finer segments
#         max_dist=4,         # ↓ smaller max_dist increases sensitivity to color/texture
#         ratio=0.5           # ↓ lower ratio favors color over spatial proximity
#     )
# else:
#     segments = segmentation.quickshift(
#         image_lab,
#         kernel_size=2.5,      # ↓ smaller kernel gives finer segments
#         max_dist=5,         # ↓ smaller max_dist increases sensitivity to color/texture
#         ratio=0.5           # ↓ lower ratio favors color over spatial proximity
#     )


# if image_lab.shape[0] <= 460 and image_lab.shape[1] <= 460:
#     segments = segmentation.quickshift(
#         image_lab,
#         kernel_size=2.25,      # ↓ smaller kernel gives finer segments
#         max_dist=3,         # ↓ smaller max_dist increases sensitivity to color/texture
#         ratio=0.5           # ↓ lower ratio favors color over spatial proximity
#     )
# else:
#     segments = segmentation.quickshift(
#         image_lab,
#         kernel_size=2.5,      # ↓ smaller kernel gives finer segments
#         max_dist=5,         # ↓ smaller max_dist increases sensitivity to color/texture
#         ratio=0.5           # ↓ lower ratio favors color over spatial proximity
#     )


# t = time.time()
# segment_generation_time = (t - s)
# print(t-s)
# Convert the segmented image to RGB
# segmented_image = color.label2rgb(segments, image, kind='avg')

# image_with_boundaries = segmentation.mark_boundaries(
#     image, segments, color=(1, 0, 0), mode='thick')
# plt.imshow(image_with_boundaries)
# plt.imsave(superpixel_img_path, image_with_boundaries)
# plt.axis('off')
# # plt.show()
# plt.close()

# num_superpixels = segments.max() + 1
print(f'number of superpixels = {num_superpixels}')
# labels_slic = segments

# result = segmented_image
