# CS 898BA - Homework 1
# boundary_extractor.py
# Purpose: Apply 4 edge detection techniques to the 42-image working subset
# Methods: Sobel, Laplacian, Canny, Prewitt
# Result: 42 originals + 168 edge images = 210 total

import cv2
import numpy as np
import os
import random

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "output")
EDGE_DIR   = os.path.join(OUTPUT_DIR, "edges")
os.makedirs(EDGE_DIR, exist_ok=True)
RANDOM_SEED = 42


def apply_sobel(gray):
    sx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    return cv2.convertScaleAbs(cv2.magnitude(sx, sy))


def apply_laplacian(gray):
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    return cv2.convertScaleAbs(lap)


def apply_canny(gray):
    blurred = cv2.GaussianBlur(gray, (5, 5), 1.0)
    return cv2.Canny(blurred, 50, 150)


def apply_prewitt(gray):
    kx = np.array([[ 1, 0,-1],[ 1, 0,-1],[ 1, 0,-1]], dtype=np.float32)
    ky = np.array([[ 1, 1, 1],[ 0, 0, 0],[-1,-1,-1]], dtype=np.float32)
    px = cv2.filter2D(gray.astype(np.float32), -1, kx)
    py = cv2.filter2D(gray.astype(np.float32), -1, ky)
    return cv2.convertScaleAbs(np.sqrt(px**2 + py**2))


def run():
    import glob
    all_images = sorted(glob.glob(os.path.join(OUTPUT_DIR, "*.png")))
    all_images = [f for f in all_images if "subset" not in f]

    random.seed(RANDOM_SEED)
    random.shuffle(all_images)
    subset = all_images[:42]

    print(f"Running edge detection on 42 images...")

    techniques = [
        ("sobel",     apply_sobel),
        ("laplacian", apply_laplacian),
        ("canny",     apply_canny),
        ("prewitt",   apply_prewitt),
    ]

    for img_path in subset:
        img  = cv2.imread(img_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        base = os.path.splitext(os.path.basename(img_path))[0]

        cv2.imwrite(os.path.join(EDGE_DIR, f"{base}_original.png"), img)

        for name, fn in techniques:
            result = fn(gray)
            cv2.imwrite(os.path.join(EDGE_DIR, f"{base}_{name}.png"), result)

    print(f"\n[boundary_extractor.py] Done. Images saved to output/edges/")
    print(f"Total images: 210")


if __name__ == "__main__":
    run()