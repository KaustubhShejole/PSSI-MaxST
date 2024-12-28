# ImageSegmentationUsingGraphTheory

This project was done by Mr. Kaustubh Shivshankar Shejole under the guidance of Dr. Gaurav Mishra at Computer Science and Engineering,
VNIT Nagpur. 
The work proposes a novel maximum spanning tree based Image segmentation leveraging low-level segmentation as Mean-Shift segmentation.

The work also proposes an efficient similarity measure between pixel segments that have a complexity linear in the number of bins in color space.

The results demonstrates the effectiveness of the method.

We worked with two low-level segmentation approaches, Mean-Shift and SLIC and found out that Mean-Shift gives better segments.


We used two datasets:
1. Oxford 102 flower dataset: https://www.robots.ox.ac.uk/~vgg/data/flowers/102/
2. EG1800 Portrait Dataset available at: https://drive.google.com/file/d/18xM3jU2dSp1DiDqEM6PVXattNMZvsX4z/view?usp=sharing


# Title: Mean-Shift meets Maximum Spanning Tree: An Efficient Interactive Graph-based Image Segmentation Technique

Image segmentation plays a crucial role in applications such as object detection, object identification, tracking, and digital image analysis. Existing deep learning based approaches excel in automation and accuracy but require annotated datasets which may not be available in critical scenarios whereas traditional methods such as edge-based, graph-based, etc. are interpretable but can be computationally rich and low in accuracy. This paper presents a graph based iterative and interactive image segmentation method that can be effectively used in such scenarios where annotated data is lacking while maintaining the accuracy, reducing the computational cost and giving the control of segmentation to the user making it suitable for critical scenarios. In this method, low-level segmentation is used to generate the segments and a segment graph is constructed using an efficient similarity measure between pixel segments and maximum spanning tree-based partitioning is used for segmenting images into foreground and background. Our proposed similarity measure for constructing graph is both time-efficient and effective in segmentation results. The maximum spanning tree captures the strongly connected local neighborhood information and is thus helpful in image segmentation. The experimental results demonstrate that the proposed algorithm outperforms the popular segmentation methods such as YOLO11 in terms of Average IoU and Average F1 score. To the best of our knowledge, this is the first demonstration of interactive image segmentation using segment graph and maximum spanning tree-based partitioning.
