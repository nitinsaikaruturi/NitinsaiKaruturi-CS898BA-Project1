# CS 898BA - Homework 2
# comparison_visualizer.py
# Purpose: Create a side-by-side comparison plot of original, normalized,
# and all 3 segmentation masks for the README

import cv2
import matplotlib.pyplot as plt
import os

INPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "input")
SEG_DIR   = os.path.join(os.path.dirname(__file__), "..", "output", "segmentation")
README_DIR = os.path.join(os.path.dirname(__file__), "..", "readme_plots")
os.makedirs(README_DIR, exist_ok=True)


def run():
    original   = cv2.cvtColor(cv2.imread(os.path.join(INPUT_DIR, "image.png")), cv2.COLOR_BGR2RGB)
    normalized = cv2.cvtColor(cv2.imread(os.path.join(SEG_DIR, "normalized_multichannel.png")), cv2.COLOR_BGR2RGB)
    otsu       = cv2.imread(os.path.join(SEG_DIR, "otsu_mask.png"), cv2.IMREAD_GRAYSCALE)
    adaptive   = cv2.imread(os.path.join(SEG_DIR, "adaptive_mask.png"), cv2.IMREAD_GRAYSCALE)
    kmeans     = cv2.imread(os.path.join(SEG_DIR, "kmeans_mask.png"), cv2.IMREAD_GRAYSCALE)
    ground_truth = cv2.imread(os.path.join(INPUT_DIR, "ground_truth_mask.png"), cv2.IMREAD_GRAYSCALE)

    panels = [
        (original, "Original", None),
        (normalized, "Multi-Channel Normalized", None),
        (otsu, "Otsu Threshold", "gray"),
        (adaptive, "Adaptive Threshold", "gray"),
        (kmeans, "K-Means Cluster", "gray"),
        (ground_truth, "Ground Truth", "gray"),
    ]

    fig, axes = plt.subplots(1, 6, figsize=(26, 5))
    for ax, (img, title, cmap) in zip(axes, panels):
        ax.imshow(img, cmap=cmap)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.axis("off")

    plt.tight_layout()
    save_path = os.path.join(README_DIR, "segmentation_comparison.png")
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()

    print(f"Saved: {save_path}")
    print("[comparison_visualizer.py] Done.")


if __name__ == "__main__":
    run()