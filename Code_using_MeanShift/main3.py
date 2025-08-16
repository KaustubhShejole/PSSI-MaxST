import os
import cv2
import subprocess
import tqdm
import shutil

image_folder = '../../image_data/images/'

image_nums = []
for file in os.listdir(image_folder):
    if file.lower().endswith('.jpg'):
        image_path = os.path.join(image_folder, file)
        image_name = os.path.splitext(file)[0]
        image_nums.append(image_name)

for image_num in tqdm.tqdm(image_nums[-13:]):
    subprocess.run(["../../.venv/Scripts/python", "main4.py", image_num])