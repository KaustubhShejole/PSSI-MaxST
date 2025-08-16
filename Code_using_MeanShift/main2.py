import random
import os
import cv2
import subprocess
import tqdm
import shutil

image_folder = '../../image_data/imagesgrabcut/images/'


image2_folder = '../../results/text_results/ssn-cut'
image2_nums = [
    entry for entry in os.listdir(image2_folder)
    if os.path.isdir(os.path.join(image2_folder, entry))
]
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
        # img = cv2.imread(image_path)

        # if img is not None:
        #     h, w = img.shape[:2]
        #     # image_sizes.append((h, w))
        #     # size_class = get_size_class(h, w)

        #     # # Make sure the target directory exists
        #     # target_dir = os.path.join(target_root, size_class)
        #     # os.makedirs(target_dir, exist_ok=True)

        #     # # Move the image
        #     # shutil.move(image_path, os.path.join(target_dir, file))
        #     # moved_count += 1
        # else:
        #     print(f"Warning: Unable to read {image_path}")

# # Print unique sizes
# unique_sizes = set(image_sizes)
# print("Unique image sizes:")
# for size in unique_sizes:
#     print(size)

# print(f"\n✅ Moved {moved_count} images into small/medium/large directories.")

# image_nums = image_nums[:1]
# print(image_nums)

# Run script for each image
# image_nums = list(set(image_nums) - set(image2_nums))
# # print(len(image_nums))

# image_nums = random.sample(image_nums, 50)
# image_nums = ['']
for image_num in tqdm.tqdm(image_nums):
    subprocess.run(["../../.venv/Scripts/python", "main1.py", image_num])
