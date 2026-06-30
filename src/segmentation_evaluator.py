# CS 898BA - Homework 2
# segmentation_evaluator.py
# Purpose: Calculate IoU and Dice Coefficient for each segmentation method
# against a manually created ground truth mask

import cv2
import numpy as np
import os

SEG_DIR = os.path.join(os.path.dirname(__file__), "..", "output", "segmentation")
GROUND_TRUTH_PATH = os.path.join(os.path.dirname(__file__), "..", "input", "ground_truth_mask.png")


def load_binary_mask(path):
    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Mask not found at: {path}")
    _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    return binary


def compute_iou(mask_a, mask_b):
    intersection = np.logical_and(mask_a == 255, mask_b == 255).sum()
    union = np.logical_or(mask_a == 255, mask_b == 255).sum()
    if union == 0:
        return 0.0
    return intersection / union


def compute_dice(mask_a, mask_b):
    intersection = np.logical_and(mask_a == 255, mask_b == 255).sum()
    total = (mask_a == 255).sum() + (mask_b == 255).sum()
    if total == 0:
        return 0.0
    return (2 * intersection) / total


def run():
    ground_truth = load_binary_mask(GROUND_TRUTH_PATH)
    gt_h, gt_w = ground_truth.shape

    methods = {
        "Otsu":     os.path.join(SEG_DIR, "otsu_mask.png"),
        "Adaptive": os.path.join(SEG_DIR, "adaptive_mask.png"),
        "K-Means":  os.path.join(SEG_DIR, "kmeans_mask.png"),
    }

    print("=" * 50)
    print("  SEGMENTATION EVALUATION RESULTS")
    print("=" * 50)

    results = {}
    for name, path in methods.items():
        mask = load_binary_mask(path)

        if mask.shape != (gt_h, gt_w):
            mask = cv2.resize(mask, (gt_w, gt_h))

        iou = compute_iou(mask, ground_truth)
        dice = compute_dice(mask, ground_truth)
        results[name] = {"IoU": iou, "Dice": dice}

        print(f"\n{name}:")
        print(f"  IoU (Jaccard Index): {iou:.4f}")
        print(f"  Dice Coefficient:    {dice:.4f}")

    print("\n[segmentation_evaluator.py] Done.")
    return results


if __name__ == "__main__":
    run()