# CS 898BA - Homework 1
# spatial_warper.py
# Purpose: Apply 2 unique affine transformations to each of the 7 base images
# Result: 14 new images added → 21 total images

import cv2
import numpy as np
import os
import glob

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def rotate_image(image, angle):
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(image, M, (w, h))


def translate_image(image, tx, ty):
    M = np.float32([[1, 0, tx], [0, 1, ty]])
    h, w = image.shape[:2]
    return cv2.warpAffine(image, M, (w, h))


def scale_image(image, fx, fy):
    h, w = image.shape[:2]
    scaled = cv2.resize(image, None, fx=fx, fy=fy)
    result = np.zeros_like(image)
    sh, sw = scaled.shape[:2]
    ch, cw = min(sh, h), min(sw, w)
    result[:ch, :cw] = scaled[:ch, :cw]
    return result


def shear_image(image, shear_x):
    h, w = image.shape[:2]
    M = np.float32([[1, shear_x, 0], [0, 1, 0]])
    new_w = w + int(abs(shear_x) * h)
    return cv2.warpAffine(image, M, (new_w, h))


TRANSFORM_PLAN = [
    (rotate_image,    (45,),        "rot_045deg"),
    (translate_image, (50, 30),     "trans_50x30"),
    (rotate_image,    (186,),       "rot_186deg"),
    (scale_image,     (1.3, 1.3),   "scale_up_1pt3"),
    (translate_image, (-40, 60),    "trans_neg40x60"),
    (shear_image,     (0.25,),      "shear_pos0pt25"),
    (rotate_image,    (90,),        "rot_090deg"),
    (scale_image,     (0.75, 0.9),  "scale_down_0pt75"),
    (shear_image,     (-0.3,),      "shear_neg0pt3"),
    (translate_image, (0, 80),      "trans_0x80"),
    (rotate_image,    (270,),       "rot_270deg"),
    (scale_image,     (1.5, 1.0),   "scale_wide_1pt5"),
    (shear_image,     (0.4,),       "shear_pos0pt4"),
    (translate_image, (70, -50),    "trans_70x_neg50"),
]


def run():
    base_images = sorted(glob.glob(os.path.join(OUTPUT_DIR, "0[0-6]_*.png")))

    if len(base_images) != 7:
        raise ValueError(f"Expected 7 base images, found {len(base_images)}. "
                         f"Run spectrum_converter and luminance_equalizer first.")

    print(f"Found {len(base_images)} base images. Applying 2 transforms each...")

    transform_index = 0
    for img_path in base_images:
        img = cv2.imread(img_path)
        base_name = os.path.splitext(os.path.basename(img_path))[0]

        for i in range(2):
            fn, args, label = TRANSFORM_PLAN[transform_index]
            warped = fn(img, *args)
            out_name = f"{base_name}_warp_{label}.png"
            cv2.imwrite(os.path.join(OUTPUT_DIR, out_name), warped)
            print(f"  Saved: {out_name}")
            transform_index += 1

    print(f"\n[spatial_warper.py] Done. 14 new images saved. Total: 21 images.")


if __name__ == "__main__":
    run()