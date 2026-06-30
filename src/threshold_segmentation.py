# CS 898BA - Homework 2
# threshold_segmentation.py
# Purpose: Apply Otsu's global thresholding and Adaptive thresholding
# to the normalized image to separate foreground from background

import cv2
import os

INPUT_PATH = os.path.join(os.path.dirname(__file__), "..", "output", "segmentation", "normalized_multichannel.png")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "output", "segmentation")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def run():
    image = cv2.imread(INPUT_PATH)
    if image is None:
        raise FileNotFoundError(f"Normalized image not found at: {INPUT_PATH}. "
                                f"Run multichannel_normalizer.py first.")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # ── 1. Otsu's Global Thresholding ──────────────────────────────────
    _, otsu_mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cv2.imwrite(os.path.join(OUTPUT_DIR, "otsu_mask.png"), otsu_mask)

    otsu_foreground = cv2.bitwise_and(image, image, mask=otsu_mask)
    cv2.imwrite(os.path.join(OUTPUT_DIR, "otsu_foreground.png"), otsu_foreground)

    print("Saved: otsu_mask.png and otsu_foreground.png")

    # ── 2. Adaptive Thresholding (Gaussian) ────────────────────────────
    adaptive_mask = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=11,
        C=2
    )
    cv2.imwrite(os.path.join(OUTPUT_DIR, "adaptive_mask.png"), adaptive_mask)

    adaptive_foreground = cv2.bitwise_and(image, image, mask=adaptive_mask)
    cv2.imwrite(os.path.join(OUTPUT_DIR, "adaptive_foreground.png"), adaptive_foreground)

    print("Saved: adaptive_mask.png and adaptive_foreground.png")
    print("[threshold_segmentation.py] Done.")

    return otsu_mask, adaptive_mask


if __name__ == "__main__":
    run()