# CS 898BA - Homework 1
# blur_pipeline.py
# Purpose: Apply Gaussian blur at 7 sigma levels to all 21 images
# Result: 21 x 7 = 147 new images → 168 total

import cv2
import os
import glob

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

SIGMA_LEVELS = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]


def compute_kernel_size(sigma):
    k = int(6 * sigma + 1)
    return k if k % 2 == 1 else k + 1


def apply_gaussian_blur(image, sigma):
    ksize = compute_kernel_size(sigma)
    return cv2.GaussianBlur(image, (ksize, ksize), sigmaX=sigma, sigmaY=sigma)


def run():
    all_images = sorted(glob.glob(os.path.join(OUTPUT_DIR, "*.png")))

    if len(all_images) != 21:
        raise ValueError(f"Expected 21 images, found {len(all_images)}. "
                         f"Run spatial_warper.py first.")

    print(f"Applying {len(SIGMA_LEVELS)} blur levels to {len(all_images)} images...")

    count = 0
    for img_path in all_images:
        img = cv2.imread(img_path)
        base_name = os.path.splitext(os.path.basename(img_path))[0]

        for sigma in SIGMA_LEVELS:
            blurred = apply_gaussian_blur(img, sigma)
            sigma_label = str(sigma).replace(".", "pt")
            out_name = f"{base_name}_blur_s{sigma_label}.png"
            cv2.imwrite(os.path.join(OUTPUT_DIR, out_name), blurred)
            count += 1

    print(f"\n[blur_pipeline.py] Done. {count} blurred images. Total: {21 + count} images.")


if __name__ == "__main__":
    run()