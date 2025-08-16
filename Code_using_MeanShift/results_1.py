import os
import cv2
import numpy as np
# from parameters_and_data import image_rgb, fpoints, bpoints, labels_slic, result, num_superpixels, image_path, seg_img_path, mask_img_path, num_superpixels_parameter
# from parameters_and_data import image_rgb, scribbling_dimension, fpoints, bpoints, visualization_image_path, scribbled_img_path, ground_truth_path, image_num

# Load your binary segmented image and ground truth image here
segmented_image = cv2.imread(
    mask_img_path + '_1_1_'+str(iteration_count_1)+ scribbles_from + '.png', cv2.IMREAD_GRAYSCALE)  # Convert to grayscale
ground_truth = cv2.imread(ground_truth_path,
                          cv2.IMREAD_GRAYSCALE)  # Convert to grayscale
our_mask = cv2.imread(mask_img_path + '_1_1_'+str(iteration_count_1)+ scribbles_from + '.png')
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
result = cv2.bitwise_and(image, our_mask)

plt.imshow(result)
plt.imsave(only_segmentation_path + '/1/'+image_num+ scribbles_from + '.png',
           result)
plt.axis('off')
# plt.show()
# plt.close()

# === Compute Confusion Matrix ===
TP = np.sum(np.logical_and(segmented_image == 255, ground_truth == 255))
FP = np.sum(np.logical_and(segmented_image == 255, ground_truth == 0))
TN = np.sum(np.logical_and(segmented_image == 0, ground_truth == 0))
FN = np.sum(np.logical_and(segmented_image == 0, ground_truth == 255))
print(f'TP: {TP}, FP: {FP}, TN: {TN}, FN: {FN}')

# === Compute Metrics ===
precision = TP / (TP + FP) if (TP + FP) > 0 else 0
# True Positive Rate (TPR)
recall = TP / (TP + FN) if (TP + FN) > 0 else 0
f1_score = 2 * precision * recall / \
    (precision + recall) if (precision + recall) > 0 else 0
iou = TP / (TP + FP + FN) if (TP + FP + FN) > 0 else 0       # Jaccard Index
fnr = FN / (FN + TP) if (FN + TP) > 0 else 0
tnr = TN / (TN + FP) if (TN + FP) > 0 else 0
fpr = FP / (FP + TN) if (FP + TN) > 0 else 0
accuracy = (TP + TN) / (TP + FP + TN + FN) if (TP + FP + TN + FN) > 0 else 0
error_rate = (FP + FN) / (TP + FP + TN + FN) if (TP + FP + TN + FN) > 0 else 0

# === Print Metrics ===
print(f"Jaccard Index (IoU): {iou}")
print(f"Precision: {precision}")
print(f"Recall (TPR): {recall}")
print(f"F1-Score (Dice): {f1_score}")
print(f"False Negative Rate (FNR): {fnr}")
print(f"True Negative Rate (TNR): {tnr}")
print(f"False Positive Rate (FPR): {fpr}")
print(f"Accuracy: {accuracy}")
print(f"Error Rate: {error_rate}")

# === Define Output Path ===
unique_filename = f"{base_name}_{counter}_1.{extension}"
output_dir = os.path.join(
    path_to_add, 'results/text_results', scribbles_from, base_name)

# === Create Folder if It Doesn’t Exist ===
if not os.path.exists(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    print(f"Folder '{base_name}' created.")
else:
    print(f"Folder '{base_name}' already exists.")

# === Save All Metrics to File ===
with open(os.path.join(output_dir, unique_filename), 'w') as file:
    file.write(f"{segment_generation_time}\n")  # Optional pre-metric info
    file.write(f"{iou}\n")
    file.write(f"{recall}\n")       # TPR
    file.write(f"{fpr}\n")
    file.write(f"{precision}\n")
    file.write(f"{f1_score}\n")
    file.write(f"{accuracy}\n")
    file.write(f"{fnr}\n")
    file.write(f"{tnr}\n")
    file.write(f"{error_rate}\n")

    # Optional timings (e.g. graph construction, MST, iterations)
    file.write(f"{elapsed_time1}\n")
    file.write(f"{elapsed_time2}\n")
    file.write(f"{elapsed_time3}\n")
    file.write(f"{elapsed_time}\n")  # Iteration 1 time
    for elapsed_time4 in elapsed_time_4_list:
        file.write(f"{elapsed_time4}\n")
print(f"Analysis of approach 1 written to {unique_filename}")