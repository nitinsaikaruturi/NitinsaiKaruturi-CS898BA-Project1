# CS 898BA - Homework 1
# luminance_equalizer.py
# Purpose: Normalize lighting on the HSV image by equalizing the V (Value/brightness) channel
# Then convert back to RGB and save as the 7th image

import cv2
import os

IMAGE_PATH = os.path.join(os.path.dirname(__file__), "..", "input", "image.png")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def run():
    image = cv2.imread(IMAGE_PATH)
    if image is None:
        raise FileNotFoundError(f"Image not found at: {IMAGE_PATH}")

    # Convert original BGR image to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Split into individual H, S, V channels
    h, s, v = cv2.split(hsv)

    # Equalize only the V (brightness/luminance) channel
    v_equalized = cv2.equalizeHist(v)

    # Merge channels back with equalized V
    hsv_equalized = cv2.merge([h, s, v_equalized])

    # Convert normalized HSV back to BGR
    normalized_bgr = cv2.cvtColor(hsv_equalized, cv2.COLOR_HSV2BGR)

    # Save as image #7
    cv2.imwrite(os.path.join(OUTPUT_DIR, "06_normalized_rgb.png"), normalized_bgr)
    print("Saved: 06_normalized_rgb.png")

    print("\n[luminance_equalizer.py] Done. You now have 7 images total.")
    return normalized_bgr


if __name__ == "__main__":
    run()