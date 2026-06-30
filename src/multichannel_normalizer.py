# CS 898BA - Homework 2
# multichannel_normalizer.py
# Purpose: Equalize all 3 color channels independently to normalize
# illumination across the entire color spectrum (not just brightness)

import cv2
import os

IMAGE_PATH = os.path.join(os.path.dirname(__file__), "..", "input", "image.png")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "output", "segmentation")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def run():
    image = cv2.imread(IMAGE_PATH)
    if image is None:
        raise FileNotFoundError(f"Image not found at: {IMAGE_PATH}")

    b, g, r = cv2.split(image)

    b_eq = cv2.equalizeHist(b)
    g_eq = cv2.equalizeHist(g)
    r_eq = cv2.equalizeHist(r)

    normalized = cv2.merge([b_eq, g_eq, r_eq])

    out_path = os.path.join(OUTPUT_DIR, "normalized_multichannel.png")
    cv2.imwrite(out_path, normalized)
    print(f"Saved: {out_path}")
    print("[multichannel_normalizer.py] Done.")

    return normalized


if __name__ == "__main__":
    run()