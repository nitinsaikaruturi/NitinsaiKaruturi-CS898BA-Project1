# CS 898BA - Homework 1
# channel_profiler.py
# Purpose: Load the original image and compute statistics for each RGB channel
# Stats: min, max, mean, median, mode, skew, range, std deviation, variance

import cv2
import numpy as np
from scipy import stats
import os

IMAGE_PATH = os.path.join(os.path.dirname(__file__), "..", "image.png")


def compute_channel_stats(channel, name):
    """Compute and print all required statistics for a single image channel."""
    flat = channel.flatten()
    mode_result = stats.mode(flat, keepdims=True)

    print(f"\n--- {name} Channel ---")
    print(f"  Min:            {np.min(flat)}")
    print(f"  Max:            {np.max(flat)}")
    print(f"  Mean:           {np.mean(flat):.4f}")
    print(f"  Median:         {np.median(flat):.4f}")
    print(f"  Mode:           {mode_result.mode[0]}")
    print(f"  Skewness:       {stats.skew(flat):.4f}")
    print(f"  Range:          {np.ptp(flat)}")
    print(f"  Std Deviation:  {np.std(flat):.4f}")
    print(f"  Variance:       {np.var(flat):.4f}")


def run():
    image = cv2.imread(IMAGE_PATH)
    if image is None:
        raise FileNotFoundError(f"Image not found at: {IMAGE_PATH}")

    print("=" * 40)
    print("  IMAGE CHANNEL STATISTICS REPORT")
    print("=" * 40)
    print(f"  Image Shape: {image.shape}")
    print(f"  Image Size:  {image.shape[1]} x {image.shape[0]} pixels")

    # OpenCV loads as BGR — split into B, G, R
    b, g, r = cv2.split(image)

    compute_channel_stats(r, "Red (R)")
    compute_channel_stats(g, "Green (G)")
    compute_channel_stats(b, "Blue (B)")

    print("\n[channel_profiler.py] Done.")


if __name__ == "__main__":
    run()
