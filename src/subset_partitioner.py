# CS 898BA - Homework 1
# subset_partitioner.py
# Purpose: Randomly split all 168 images into 4 equal subsets of 42

import os
import glob
import random

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "output")
RANDOM_SEED = 42


def run():
    all_images = sorted(glob.glob(os.path.join(OUTPUT_DIR, "*.png")))

    if len(all_images) != 168:
        raise ValueError(f"Expected 168 images, found {len(all_images)}. "
                         f"Run blur_pipeline.py first.")

    random.seed(RANDOM_SEED)
    shuffled = all_images[:]
    random.shuffle(shuffled)

    subsets = [shuffled[i * 42:(i + 1) * 42] for i in range(4)]

    for i, subset in enumerate(subsets):
        list_path = os.path.join(OUTPUT_DIR, f"subset_{i}_filelist.txt")
        with open(list_path, "w") as f:
            f.write("\n".join(subset))
        print(f"Subset {i}: {len(subset)} images saved to subset_{i}_filelist.txt")

    print(f"\n[subset_partitioner.py] Done. Using Subset 0 for edge detection.")
    return subsets[0]


if __name__ == "__main__":
    run()