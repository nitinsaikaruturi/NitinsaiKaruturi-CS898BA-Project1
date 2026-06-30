# CS 898BA - Homework 1
# spectrum_converter.py
# Purpose: Convert the original image into 5 different color spaces
# Outputs: Grayscale, Binary, HSV, CIELAB, HLS → saved to /output

import cv2
import os

IMAGE_PATH = os.path.join(os.path.dirname(__file__), "..", "image.png")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def run():
    image = cv2.imread(IMAGE_PATH)
    if image is None:
        raise FileNotFoundError(f"Image not found at: {IMAGE_PATH}")

    # Save original
    cv2.imwrite(os.path.join(OUTPUT_DIR, "00_original.png"), image)
    print("Saved: 00_original.png")

    # 1. Grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(os.path.join(OUTPUT_DIR, "01_grayscale.png"), gray)
    print("Saved: 01_grayscale.png")

    # 2. Binary (threshold grayscale at midpoint 127)
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    cv2.imwrite(os.path.join(OUTPUT_DIR, "02_binary.png"), binary)
    print("Saved: 02_binary.png")

    # 3. HSV (Hue, Saturation, Value)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    cv2.imwrite(os.path.join(OUTPUT_DIR, "03_hsv.png"), hsv)
    print("Saved: 03_hsv.png")

    # 4. CIELAB (perceptual color space)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    cv2.imwrite(os.path.join(OUTPUT_DIR, "04_cielab.png"), lab)
    print("Saved: 04_cielab.png")

    # 5. HLS (Hue, Lightness, Saturation)
    hls = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
    cv2.imwrite(os.path.join(OUTPUT_DIR, "05_hls.png"), hls)
    print("Saved: 05_hls.png")

    print("\n[spectrum_converter.py] Done. 6 images saved (including original).")
    return image, gray, binary, hsv, lab, hls


if __name__ == "__main__":
    run()
