

import os
import cv2
# Load your binary segmented image and ground truth image here
segmented_image = cv2.imread(mask_img_path2+str(iteration_count_2) +
                             '.png', cv2.IMREAD_GRAYSCALE)  # Convert to grayscale
ground_truth = cv2.imread(ground_truth_path,
                          cv2.IMREAD_GRAYSCALE)  # Convert to grayscale
# Calculate confusion matrix
TP = np.sum(np.logical_and(segmented_image == 255, ground_truth == 255))
FP = np.sum(np.logical_and(segmented_image == 255, ground_truth == 0))
TN = np.sum(np.logical_and(segmented_image == 0, ground_truth == 0))
FN = np.sum(np.logical_and(segmented_image == 0, ground_truth == 255))
print(f'TP: {TP}, FP: {FP}, TN: {TN}, FN: {FN}')

# Calculate True Positive Rate (TPR) and False Positive Rate (FPR)
TPR = TP / (TP + FN)
FPR = FP / (FP + TN)
print("True Positive Rate (TPR):", TPR)
print("False Positive Rate (FPR):", FPR)

image = cv2.imread(image_path)

mask = cv2.imread(ground_truth_path)
our_mask = cv2.imread(mask_img_path2+str(iteration_count_2) +
                      '.png')

result = cv2.bitwise_and(image, our_mask)
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
result = cv2.bitwise_and(image, our_mask)

plt.imshow(result)
plt.imsave(only_segmentation_path + '/2/'+image_num+'.png',
           result)
plt.axis('off')
plt.show()


unique_filename = f"{base_name}_{counter}_2.{extension}"
if not os.path.exists('results/text_results/'+base_name):
    # Create the folder
    os.makedirs('results/text_results/'+base_name)
    print(f"Folder '{base_name}' created.")
else:
    print(f"Folder '{base_name}' already exists.")
# Write the variables to the unique file
with open(path_to_add + 'results/text_results/'+base_name+'/'+unique_filename, 'w') as file:
    file.write(f"{segment_generation_time}\n")
    file.write(f"{TPR}\n")
    file.write(f"{FPR}\n")
    # finding neighbors of ech superpixel
    file.write(f"{elapsed_time1}\n")
    file.write(f"{elapsed_time2}\n")  # Graph Construction
    # Maximum Spanning Tree Partitioning
    file.write(f"{elapsed_time3}\n")
    file.write(f"{elapsed_time}\n")  # Iteration 1 time
    for elapsed_time4 in elapsed_time_4_list:
        file.write(f"{elapsed_time4}\n")
print(f"Analysis of approach 2 written to {unique_filename}")
