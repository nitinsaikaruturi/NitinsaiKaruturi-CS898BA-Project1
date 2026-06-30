# CS 898BA - Homework 2
# kmeans_segmentation.py
# Purpose: Use K-Means clustering in HSV color space to segment
# the image into K regions, then isolate the cluster matching the figure

import cv2
import numpy as np
import os

np.random.seed(42)
cv2.setRNGSeed(42)

INPUT_PATH = os.path.join(os.path.dirname(__file__), "..", "output", "segmentation", "normalized_multichannel.png")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "output", "segmentation")
os.makedirs(OUTPUT_DIR, exist_ok=True)

K_VALUES_TO_TEST = [3, 4, 5]
CHOSEN_K = 4
CHOSEN_CLUSTER_INDEX = 1


def run_kmeans(image_hsv, k):
    h, w = image_hsv.shape[:2]
    pixels = image_hsv.reshape((-1, 3)).astype(np.float32)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, centers = cv2.kmeans(
        pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS
    )

    labels = labels.reshape((h, w))
    return labels, centers


def save_all_clusters(image, labels, k, output_prefix):
    for cluster_id in range(k):
        mask = np.uint8(labels == cluster_id) * 255
        # Clean up small noise specks using morphological opening
        kernel = np.ones((5, 5), np.uint8)
        mask_clean = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_CLOSE, kernel)
        cv2.imwrite(os.path.join(OUTPUT_DIR, f"{output_prefix}_k{k}_cluster{cluster_id}_mask.png"), mask_clean)


def run():
    image = cv2.imread(INPUT_PATH)
    if image is None:
        raise FileNotFoundError(f"Normalized image not found at: {INPUT_PATH}. "
                                f"Run multichannel_normalizer.py first.")

    # Smooth the image first to reduce grass texture noise before clustering
    blurred = cv2.GaussianBlur(image, (9, 9), 2.0)

    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    print("Testing K values:", K_VALUES_TO_TEST)
    for k in K_VALUES_TO_TEST:
        labels, centers = run_kmeans(hsv, k)
        save_all_clusters(image, labels, k, "kmeans_test")
        print(f"  Saved all {k} cluster masks for K={k}")

    print("\n>>> STOP: Go look at the saved cluster masks in output/segmentation/")
    print(">>> Find which K and which cluster number best matches the figure.")
    print(">>> Then update CHOSEN_K and CHOSEN_CLUSTER_INDEX at the top of this file.\n")

    labels, centers = run_kmeans(hsv, CHOSEN_K)
    final_mask = np.uint8(labels == CHOSEN_CLUSTER_INDEX) * 255

    # Clean up the final mask too
    kernel = np.ones((5, 5), np.uint8)
    final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_OPEN, kernel)
    final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, kernel)

    cv2.imwrite(os.path.join(OUTPUT_DIR, "kmeans_mask.png"), final_mask)

    kmeans_foreground = cv2.bitwise_and(image, image, mask=final_mask)
    cv2.imwrite(os.path.join(OUTPUT_DIR, "kmeans_foreground.png"), kmeans_foreground)

    print(f"Saved final: kmeans_mask.png and kmeans_foreground.png (K={CHOSEN_K}, cluster={CHOSEN_CLUSTER_INDEX})")
    print("[kmeans_segmentation.py] Done.")

    return final_mask


if __name__ == "__main__":
    run()