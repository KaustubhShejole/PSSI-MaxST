import csv
import os
import cv2
import subprocess
import tqdm

csv_file = '../../results/text_results/optimal_scribbles.csv'
image_names = []

with open(csv_file, 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        image_names.append(row['image_name'])

# Print or use the list
# print("Image names:", image_names)

for image_num in tqdm.tqdm(image_names):
    subprocess.run(["../../.venv/Scripts/python", "main1.py", image_num])
