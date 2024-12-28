# Title: Mean-Shift meets Maximum Spanning Tree: An Efficient Interactive Graph-based Image Segmentation Technique

Image segmentation plays a crucial role in applications such as object detection, object identification, tracking, and digital image analysis. Existing deep learning based approaches excel in automation and accuracy but require annotated datasets which may not be available in critical scenarios whereas traditional methods such as edge-based, graph-based, etc. are interpretable but can be computationally rich and low in accuracy. This paper presents a graph based iterative and interactive image segmentation method that can be effectively used in such scenarios where annotated data is lacking while maintaining the accuracy, reducing the computational cost and giving the control of segmentation to the user making it suitable for critical scenarios. In this method, low-level segmentation is used to generate the segments and a segment graph is constructed using an efficient similarity measure between pixel segments and maximum spanning tree-based partitioning is used for segmenting images into foreground and background. Our proposed similarity measure for constructing graph is both time-efficient and effective in segmentation results. The maximum spanning tree captures the strongly connected local neighborhood information and is thus helpful in image segmentation. The experimental results demonstrate that the proposed algorithm outperforms the popular segmentation methods such as YOLO11 in terms of Average IoU and Average F1 score. To the best of our knowledge, this is the first demonstration of interactive image segmentation using segment graph and maximum spanning tree-based partitioning.

We used two datasets:
1. Oxford 102 flower dataset: https://www.robots.ox.ac.uk/~vgg/data/flowers/102/
2. EG1800 Portrait Dataset available at: https://drive.google.com/file/d/18xM3jU2dSp1DiDqEM6PVXattNMZvsX4z/view?usp=sharing

---

## **Setup the project**
## **Setup Instructions**

### Step 1: Clone the Repository
```bash
git clone https://github.com/KaustubhShejole/ImageSegmentationUsingGraphTheory/
```
```bash
cd ImageSegmentationUsingGraphTheory/Code_using_MeanShift
```

### Step 2: Install Requirements
Install the required Python packages listed in `requirements.txt`:

```bash
pip install -r requirements.txt
```

### Step 3: Verify Dependencies
Ensure you have the following installed:
- Python 3.6 or higher
- OpenCV
- NumPy
- Matplotlib
- Scikit-Image

You can verify installation with:
```bash
python -c "import cv2, numpy, matplotlib, skimage"
```

---

## **Running the Scripts**
### Step 1: Prepare Input Data
- Place your flower images in `../../final_research/flower_segmentation_dataset/flowers/`.
- Place your corresponding masks in `../../final_research/flower_segmentation_dataset/masks/`.

i.e. 

Place your input images in the following path:

```

├── final_research/
│   ├── flower_segmentation_dataset/
│   │   ├── flowers/
│   │   │   ├── image1.jpg
│   │   │   ├── image2.jpg
│   │   │   └── (other images...)
│   │   ├── masks/
│   │   │   ├── mask1.jpg
│   │   │   ├── mask2.jpg
│   │   │   └── (other masks...)
├──ImageSegmentationUsingGraphTheory/
│   ├── ImageSegmentationUsingGraphTheory/
│   │   ├── Code_using_MeanShift/

You can change the input paths according to your convenience.
```
### Step 2: Creating Necessary Output Directories
The `parameters_and_data.py` script automatically creates the required directories under `results/` if they don't exist:

```
Code_using_MeanShift/
├── results/
│   ├── visualising_superpixels/
│   ├── analysis/
│   ├── masks/
│   ├── scribbled_images/
│   ├── segmentation_results/
│   └── (other directories...)
```

Ensure the images and corresponding masks are named appropriately for consistent segmentation results.

You can customize these directories by editing the `parameters_and_data.py` file.

### Step 3: Run the Main Script
Run the segmentation and visualization script:
```bash
python main.py
```
This file runs the process of interactive and iterative image segmentation.

### Step 4: Visualize Outputs
Results are saved in the following locations:
- **Superpixel/Segments Visualizations**: `results/visualising_superpixels/flowers/`
- **Scribbled Images**: `results/scribbled_images/flowers/`
- **Segmentation Results**: `results/segmentation_results/flowers/`

Use any image viewer or Matplotlib to inspect the outputs.

---

## **Key Parameters**
The kernel_size 
The max_dist 
The ratio 
- `kernel_size`: determines the resolution or granularity of segmentation. (3)
- `max_dist`: governs the threshold for segment connectivity. (6)
- `ratio`: controls the relative importance of spatial proximity versus color similarity. (0.6)

You can modify these in `parameters_and_data.py`.

---

## **Troubleshooting**

- Check directory paths in `parameters_and_data.py`.
- Install any missing Python packages as prompted.

---

## **License**
This project is licensed under the MIT License.

---

## **Contact**
For questions or issues, please contact [your-email@example.com].
