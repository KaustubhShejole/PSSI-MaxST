import os
import cv2
import subprocess
import tqdm
import shutil

# image_folder = '../../image_data/imagesgrabcut1/small/images'
image_folder = '../../image_data/imagesgrabcut1/small/images'


# target_root = '../../image_data/imagesgrabcut1'

# def get_size_class(h, w):
#     area = h * w
#     if area <= 160000:       # ~< 400x400
#         return 'small'
#     elif area <= 300000:     # ~< 550x550
#         return 'medium'
#     else:
#         return 'large'

# image_sizes = []
# moved_count = 0
image_nums = []
for file in os.listdir(image_folder):
    if file.lower().endswith('.jpg'):
        image_path = os.path.join(image_folder, file)
        image_name = os.path.splitext(file)[0]
        image_nums.append(image_name)

print(image_nums)