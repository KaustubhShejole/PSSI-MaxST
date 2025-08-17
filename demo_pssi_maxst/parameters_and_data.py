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

num_superpixels_parameter = 13
compactness = 10.0

# path_to_add_to_get_image = "images/"
path_to_add = "../../"

superpixel_img_path = path_to_add + 'results/' + \
    'superpixels_visualization/flowers/' + image_num + '.jpg'

mask_img_path = path_to_add + 'results/'+'masks/flowers/' + image_num
mask_img_path2 = path_to_add + 'results/masks1/flowers/' + image_num + '_2_'
seg_img_path2 = path_to_add + 'results/segmentation_results1/flowers/' + image_num + '_2_'

only_segmentation_path = path_to_add+ 'results/only_segmentations'
markers_save_path = "../../image_data/markers_ours_images250/" + image_num

# image_path = path_to_add_to_get_image + image_num + '.jpg'
# image_path = path_to_add_to_get_image + "/images/" + image_num + '.bmp'
scribbled_img_path = path_to_add + 'results/'+'scribbled_images/flowers/' + image_num
scribbled_img_path2 = path_to_add + 'results/'+'scribbled_images1/flowers/' + image_num + '_2_'

visualization_image_path = path_to_add + 'results/' + \
    'visualising_superpixels/flowers/'+ image_num + '.png'
seg_img_path = path_to_add + 'results/'+'segmentation_results/flowers/' + image_num
image_rgb = cv2.imread(image_path)
num_pixels = image_rgb.shape[0] * image_rgb.shape[1]


ground_truth_path = "GT/" + image_num + '.bmp' # + '.png'
num_superpixels_parameter = int(np.sqrt(np.sqrt(num_pixels/2)/2))

image_lab = cv2.cvtColor(image_rgb, cv2.COLOR_BGR2Lab)
image_rgb = cv2.cvtColor(image_rgb, cv2.COLOR_BGR2RGB)
# Convert the image to Lab color space
image = np.copy(image_rgb)
original_image = np.copy(image_rgb)
print(f'{image_lab.shape[1]} * {image_lab.shape[0]}')

h = image_lab.shape[0]
w = image_lab.shape[1]
area = h * w


s = time.time()
if area <= 160000:       # ~< 400x400
    segments = segmentation.quickshift(
        image_lab,
        kernel_size=2.25,      # ↓ smaller kernel gives finer segments
        max_dist=3,         # ↓ smaller max_dist increases sensitivity to color/texture
        ratio=0.5           # ↓ lower ratio favors color over spatial proximity
    )
elif area <= 300000:     # ~< 550x550
    segments = segmentation.quickshift(
        image_lab,
        kernel_size=2.5,      # ↓ smaller kernel gives finer segments
        max_dist=4,         # ↓ smaller max_dist increases sensitivity to color/texture
        ratio=0.5           # ↓ lower ratio favors color over spatial proximity
    )
else:
    segments = segmentation.quickshift(
        image_lab,
        kernel_size=2.5,      # ↓ smaller kernel gives finer segments
        max_dist=5,         # ↓ smaller max_dist increases sensitivity to color/texture
        ratio=0.5           # ↓ lower ratio favors color over spatial proximity
    )


t = time.time()
segment_generation_time = (t - s)
print(t-s)
# Convert the segmented image to RGB
segmented_image = color.label2rgb(segments, image, kind='avg')

image_with_boundaries = segmentation.mark_boundaries(
    image, segments, color=(1, 0, 0), mode='thick')
plt.imshow(image_with_boundaries)
plt.imsave(superpixel_img_path, image_with_boundaries)
plt.axis('off')
# plt.show()
plt.close()

num_superpixels = segments.max() + 1
print(f'number of superpixels = {num_superpixels}')
labels_slic = segments

result = segmented_image
