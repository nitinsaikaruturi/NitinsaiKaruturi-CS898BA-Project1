# CS 898BA - Homework 2
# optimal_k_analyzer.py
# Purpose: Compare IoU/Dice across K=3, 4, 5 to justify the chosen K value

import cv2
import numpy as np
import os

np.random.seed(42)
cv2.setRNGSeed(42)

INPUT_PATH = os.path.join(os.path.dirname(__file__), "..", "output", "segmentation", "normalized_multichannel.png")
GT_PATH = os.path.join(os.path.dirname(__file__), "..", "input", "ground_truth_mask.png")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "output", "segmentation")

K_VALUES_TO_TEST = [3, 4, 5]


def run_kmeans(image_hsv, k):
    h, w = image_hsv.shape[:2]
    pixels = image_hsv.reshape((-1, 3)).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    return labels.reshape((h, w))


def clean_mask(mask):
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


def compute_iou_dice(pred_mask, gt_mask):
    pred = pred_mask > 0
    gt = gt_mask > 0
    intersection = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    iou = intersection / union if union > 0 else 0
    dice = (2 * intersection) / (pred.sum() + gt.sum()) if (pred.sum() + gt.sum()) > 0 else 0
    return iou, dice


def run():
    image = cv2.imread(INPUT_PATH)
    gt_mask = cv2.imread(GT_PATH, cv2.IMREAD_GRAYSCALE)

    if image is None or gt_mask is None:
        raise FileNotFoundError("Missing normalized image or ground truth mask.")

    blurred = cv2.GaussianBlur(image, (9, 9), 2.0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    print(f"{'K':<5}{'Best Cluster':<15}{'IoU':<10}{'Dice':<10}")
    results = []

    for k in K_VALUES_TO_TEST:
        labels = run_kmeans(hsv, k)
        best_iou, best_dice, best_cluster = 0, 0, -1

        for cluster_id in range(k):
            mask = np.uint8(labels == cluster_id) * 255
            mask = clean_mask(mask)
            iou, dice = compute_iou_dice(mask, gt_mask)
            if iou > best_iou:
                best_iou, best_dice, best_cluster = iou, dice, cluster_id

        print(f"{k:<5}{best_cluster:<15}{best_iou:<10.4f}{best_dice:<10.4f}")
        results.append({"k": k, "best_cluster": best_cluster, "iou": best_iou, "dice": best_dice})

    print("\n=== Summary ===")
    best_overall = max(results, key=lambda r: r["iou"])
    print(f"Best K = {best_overall['k']} (cluster {best_overall['best_cluster']}) "
          f"with IoU={best_overall['iou']:.4f}, Dice={best_overall['dice']:.4f}")

    return results


if __name__ == "__main__":
    run()