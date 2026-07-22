# NitinsaiKaruturi-CS898BA-Project1
**Author:** Nitin Sai Karuturi  
**Course:** CS 898BA – Image Analysis and Computer Vision  
**Assignment:** Homework 1, Homework 2 & Homework 3

---

## Project Overview

This project applies fundamental image analysis, processing, and segmentation 
techniques to an image captured by a doorbell camera. Homework 1 focused on 
color space transformations, affine transformations, blurring, and edge 
detection. Homework 2 builds on this pipeline to isolate the figure in the 
image using classical and optimization-based segmentation techniques.

---

## Setup & Installation

### Requirements
- Python 3.10+
- OpenCV, NumPy, SciPy, Matplotlib

### Install Dependencies
\`\`\`bash
pip install -r requirements.txt
\`\`\`

### Run the Project — Homework 1
Run scripts in this order from the `src/` directory:
\`\`\`bash
python Hello_World.py
python channel_image_statistics.py
python color_spectrum_conversions.py
python luminance_equalizer.py
python spatial_warper.py
python blur_pipeline.py
python subset_partitioner.py
python boundary_extractor.py
python visual_report.py
\`\`\`

### Run the Project — Homework 2
Run scripts in this order from the `src/` directory (after Homework 1 scripts):
\`\`\`bash
python multichannel_normalizer.py
python threshold_segmentation.py
python kmeans_segmentation.py
python segmentation_evaluator.py
python comparison_visualizer.py

\`\`\`

### Run the Project — Homework 3
Run scripts in this order from the `src/` directory (after HW1/HW2 scripts, and after placing the Fish dataset in `hw3_data/Fish/`):
\`\`\`bash
python fish_data_pipeline.py
python fish_cnn_baseline.py
python fish_hyperparameter_tuning.py
python fish_model_evaluator.py
python fish_results_dashboard.py
\`\`\`

---

## File Descriptions

### Homework 1

| File | Purpose |
|---|---|
| `Hello_World.py` | Initial commit script |
| `channel_image_statistics.py` | Computes min, max, mean, median, mode, skew, range, std, variance per RGB channel |
| `color_spectrum_conversions.py` | Converts image to Grayscale, Binary, HSV, CIELAB, HLS |
| `luminance_equalizer.py` | Normalizes brightness via histogram equalization on HSV V channel |
| `spatial_warper.py` | Applies 14 unique affine transformations across 7 images |
| `blur_pipeline.py` | Applies Gaussian blur at 7 sigma levels to all 21 images |
| `subset_partitioner.py` | Splits 168 images into 4 random subsets of 42 |
| `boundary_extractor.py` | Runs Sobel, Laplacian, Canny, Prewitt edge detection on 42 images |
| `visual_report.py` | Generates 42 comparison plots, saves 6 randomly to readme_plots/ |

### Homework 2

| File | Purpose |
|---|---|
| `multichannel_normalizer.py` | Equalizes all 3 BGR channels independently for full-spectrum normalization |
| `threshold_segmentation.py` | Applies Otsu's global thresholding and Adaptive thresholding segmentation |
| `kmeans_segmentation.py` | Applies K-Means clustering segmentation in HSV color space |
| `segmentation_evaluator.py` | Calculates IoU and Dice Coefficient against a manual ground truth mask |
| `comparison_visualizer.py` | Generates the 6-panel segmentation comparison plot for the README |

---

### Homework 3

| File | Purpose |
|---|---|
| `fish_data_pipeline.py` | Loads Fish dataset, creates stratified 70/15/15 train/val/test splits, resizes/normalizes images, applies augmentation |
| `fish_cnn_baseline.py` | Builds and trains the baseline 3-conv-layer CNN, saves model and training curves |
| `fish_hyperparameter_tuning.py` | Grid search over learning rate, batch size, and dropout; saves best model based on validation loss |
| `fish_model_evaluator.py` | Evaluates baseline and tuned models on the test set; generates classification reports and confusion matrix |
| `fish_results_dashboard.py` | Builds the combined training-curve/confusion-matrix visualization for the README |

---

## Image Count Summary (Homework 1)

| Stage | Images |
|---|---|
| Original image | 1 |
| After color conversions + normalization | 7 |
| After affine transformations | 21 |
| After Gaussian blur pipeline | 168 |
| After edge detection | 210 |

---

# Homework 1: Image Analysis & Computer Vision

## Part 2 Results

### Channel Statistics
The original image was analyzed across all 3 RGB channels.
Each channel showed different intensity distributions reflecting
the dark and low-contrast nature of the doorbell camera image.

### Color Conversions
The image was successfully converted into 5 color spaces:
- **Grayscale** — removes color, keeps luminance
- **Binary** — black and white only using threshold of 127
- **HSV** — separates color (hue) from brightness (value)
- **CIELAB** — perceptual color space mimicking human vision
- **HLS** — separates hue from lightness and saturation

### Lighting Normalization
Histogram equalization was applied to the V channel of the HSV image.
This spread out the brightness values across the full range,
making the image significantly clearer and easier to analyze.

### Gaussian Blur Sigma Analysis

| Sigma | Effect |
|---|---|
| 0.5 | Very subtle blur, fine details preserved |
| 1.0 | Mild smoothing, small textures begin to blend |
| 1.5 | Moderate blur, edges start losing sharpness |
| 2.0 | Noticeable softening, medium detail fades |
| 2.5 | Strong blur, only dominant shapes remain |
| 3.0 | Heavy blur, fine details fully removed |
| 3.5 | Extreme blur, only broad color regions remain |

As sigma increases the Gaussian kernel widens and averages over more
neighboring pixels, progressively removing high frequency detail like
edges and textures while keeping low frequency structure like large shapes.

---

## Part 3 Results

### Edge Detection Analysis

| Technique | Pros | Cons |
|---|---|---|
| **Sobel** | Fast, directional sensitivity, good edge magnitude | Thick edges, sensitive to noise |
| **Laplacian** | Detects edges in all directions, rotation invariant | Very noise sensitive, needs pre-blurring |
| **Canny** | Thin clean edges, noise robust, configurable thresholds | Requires threshold tuning |
| **Prewitt** | Simple, slightly smoother than Sobel | Less accurate than Sobel, less robust than Canny |

### Best Technique for This Image
**Canny** performed best on this image set because the doorbell camera 
image is low contrast and noisy. Canny's built-in Gaussian pre-blur 
and two-threshold hysteresis produced the cleanest and most useful 
edges while suppressing background noise effectively.

---

## Sample Comparison Plots
Each plot shows: Original | Sobel | Laplacian | Canny | Prewitt

![Plot 1](readme_plots/readme_plot_01.png)
![Plot 2](readme_plots/readme_plot_02.png)
![Plot 3](readme_plots/readme_plot_03.png)
![Plot 4](readme_plots/readme_plot_04.png)
![Plot 5](readme_plots/readme_plot_05.png)
![Plot 6](readme_plots/readme_plot_06.png)

---

# Homework 2: Image Segmentation

## Part 2 Results

### Multi-Channel Normalization
All 3 color channels (B, G, R) were independently histogram-equalized
and merged back together. Compared to Homework 1's single-channel (V only)
normalization, this multi-channel approach more aggressively balanced
contrast across all color information, producing a fully normalized
color image used as the input for all segmentation tasks below.

---

## Part 3 & 4 Results

### Segmentation Methods Applied
- **Otsu's Global Thresholding** — automatically calculates a single optimal
  threshold value to separate foreground from background based on the
  grayscale histogram.
- **Adaptive Thresholding (Gaussian)** — calculates a local threshold for
  each region of the image, intended to better handle uneven illumination.
- **K-Means Clustering** — clusters pixels in HSV color space into K groups
  (tested K = 3, 4, 5) and isolates the cluster best matching the figure.
  A Gaussian blur pre-processing step and morphological opening/closing
  cleanup were applied to reduce grass texture noise in the resulting mask.

### Quantitative Results

| Method | IoU | Dice Coefficient |
|---|---|---|
| Otsu Thresholding | 0.0449 | 0.0859 |
| Adaptive Thresholding | 0.0787 | 0.1459 |
| K-Means Clustering | 0.1979 | 0.3305 |

---

## Part 5: Evaluation and Analysis

### Qualitative Analysis

**Otsu Thresholding** performed the worst of the three methods (IoU 0.0449). 
Otsu calculates a single global threshold for the entire image, which struggles 
with this scene's uneven outdoor lighting — the dark grass, varying shadow areas, 
and bright sky all competed for the same brightness range as the figure, causing 
significant background noise to be classified as foreground while parts of the 
figure itself were misclassified as background.

**Adaptive Thresholding** improved on Otsu (IoU 0.0787) by calculating local 
thresholds for small regions rather than one global value, which helped handle 
some of the uneven lighting. However, it still struggled significantly with 
grass texture noise, since the local brightness variation in grass blades was 
similar in scale to the brightness variation at the figure's edges.

**K-Means Clustering** clearly outperformed both threshold-based methods 
(IoU 0.1979, more than double Adaptive's score). By clustering in HSV color 
space rather than relying on simple brightness thresholds, K-Means could 
separate the figure's clothing/skin tones from the grass and sky based on 
hue and saturation differences, not just brightness. Applying Gaussian blur 
before clustering and morphological cleanup (opening/closing) afterward 
further reduced grass texture noise that initially fragmented the mask.

**Impact of multi-channel normalization:** Compared to Homework 1's edge 
detection results (which used only single-channel V normalization), the 
full 3-channel normalization in this assignment produced more balanced 
contrast across the color image, which directly benefited K-Means clustering 
since it depends on color information across all channels. The threshold-based 
methods (Otsu, Adaptive) benefited less from this normalization since they 
only use the grayscale-converted version of the image, discarding the color 
information that multi-channel normalization improved.

**Best performing method:** K-Means clustering was the best performer by a 
clear margin in both IoU and Dice metrics. This matches what we observed 
visually — the K-Means mask preserved a recognizable full-body silhouette 
of the figure, while Otsu and Adaptive masks were dominated by background 
texture noise that overwhelmed the relatively small ground truth figure area.

---

## Comparison Visualization
Each plot shows: Original | Multi-Channel Normalized | Otsu | Adaptive | K-Means | Ground Truth

![Segmentation Comparison](readme_plots/segmentation_comparison.png)

---

## Homework 3: Deep Learning for Fish Classification

### Overview
This assignment builds a CNN to classify 6 species of fish (Bete, Cray, Discuss, Gold, Guppy, Oscar) from a dataset of 1016 labeled images, then performs systematic hyperparameter tuning to attempt to improve performance.

### Data Pipeline
- Images were split into stratified train/val/test sets (70/15/15) preserving class proportions.
- All images were resized to 128x128 and normalized to [0, 1].
- Training data was augmented with random horizontal flips, small rotations, and brightness adjustments to reduce overfitting.
- **Class imbalance note:** Cray had only 80 total images, notably fewer than other classes (145-207), which affected model performance on that class throughout this assignment.

### Baseline CNN
A custom 3-convolutional-layer CNN (32, 64, 128 filters) with max-pooling and a dense hidden layer was trained for 20 epochs using Adam (lr=0.001, batch size=32).

- Training accuracy reached 96%, but validation accuracy plateaued around 83-85%.
- Validation loss decreased initially then increased in later epochs while training loss kept dropping — a clear overfitting pattern (see `hw3_plots/baseline_training_curves.png`).

### Hyperparameter Tuning
A grid search was performed over:
- Learning rate: 0.01, 0.001, 0.0001
- Batch size: 32, 64
- Dropout rate: 0.3, 0.5

Each of the 12 combinations was trained for 10 epochs, and the best configuration was selected based on validation loss.

**Best configuration:** learning rate = 0.001, batch size = 64, dropout = 0.3 (final val_loss = 0.587, final val_accuracy = 80.9%)

### Evaluation on Test Set

| Model    | Test Accuracy | Macro F1 |
|----------|---------------|----------|
| Baseline | 87.6%         | 0.853    |
| Tuned    | 80.4%         | 0.766    |

Full per-class classification reports are saved in `hw3_output/classification_reports.txt`.

**Key finding:** despite the tuned model achieving a lower validation loss during training (and visibly less overfitting — see `hw3_plots/results_dashboard.png`), it underperformed the baseline on the held-out test set. This highlights an important distinction: optimizing for validation loss does not guarantee better generalization on unseen test data, particularly with a small and imbalanced dataset. The **Cray** class (only 80 total images, the smallest class) suffered the most in the tuned model, with recall dropping from 0.750 (baseline) to 0.417 (tuned) — visible in `hw3_plots/confusion_matrix_tuned.png`, where several Cray images were misclassified as Guppy.

### Qualitative Analysis
- **Augmentation:** random flips and rotations helped the model generalize across fish orientation, though brightness augmentation required a value-range fix to avoid pixel clipping errors.
- **Dropout (0.3 vs 0.5):** 0.3 was selected as part of the best configuration, suggesting the network benefited from mild regularization without excessive information loss during training.
- **Batch size (64 vs 32):** the best configuration used batch size 64, which may have contributed to smoother gradient updates but also gave the smallest class (Cray) fewer updates per epoch, worsening its recall.
- **Learning rate:** 0.001 was selected as optimal, consistent with typical Adam optimizer defaults for CNNs of this scale.

### Visualizations
- `hw3_plots/augmentation_sample.png` - sample of augmented training images
- `hw3_plots/baseline_training_curves.png` - baseline model loss/accuracy curves
- `hw3_plots/results_dashboard.png` - combined baseline vs. tuned model training curves
- `hw3_plots/confusion_matrix_tuned.png` - confusion matrix for the tuned model on the test set

## AI Usage
See [AI_Log.md](AI_Log.md) for full AI usage tracking across both assignments.