# CS 898BA - Homework 1
# visual_report.py
# Purpose: Create 42 five-panel comparison plots
# Saves 6 random plots to readme_plots/ for the README

import cv2
import matplotlib.pyplot as plt
import os
import glob
import random
import shutil

EDGE_DIR   = os.path.join(os.path.dirname(__file__), "..", "output", "edges")
PLOTS_DIR  = os.path.join(os.path.dirname(__file__), "..", "output", "plots")
README_DIR = os.path.join(os.path.dirname(__file__), "..", "readme_plots")
os.makedirs(PLOTS_DIR,  exist_ok=True)
os.makedirs(README_DIR, exist_ok=True)
RANDOM_SEED = 42


def create_comparison_plot(base_name, save_path):
    def load(suffix):
        path = os.path.join(EDGE_DIR, f"{base_name}_{suffix}.png")
        return cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    images = [
        (load("original"),  "Original"),
        (load("sobel"),     "Sobel"),
        (load("laplacian"), "Laplacian"),
        (load("canny"),     "Canny"),
        (load("prewitt"),   "Prewitt"),
    ]

    fig, axes = plt.subplots(1, 5, figsize=(22, 5))
    fig.suptitle(f"Image: {base_name}", fontsize=9)

    for ax, (img, title) in zip(axes, images):
        ax.imshow(img, cmap="gray")
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def run():
    originals = sorted(glob.glob(os.path.join(EDGE_DIR, "*_original.png")))

    if len(originals) != 42:
        raise ValueError(f"Expected 42 originals in edges/, found {len(originals)}. "
                         f"Run boundary_extractor.py first.")

    print(f"Creating {len(originals)} comparison plots...")

    all_plots = []
    for orig_path in originals:
        base = os.path.basename(orig_path).replace("_original.png", "")
        plot_path = os.path.join(PLOTS_DIR, f"{base}_comparison.png")
        create_comparison_plot(base, plot_path)
        all_plots.append(plot_path)
        print(f"  Saved: {base}_comparison.png")

    random.seed(RANDOM_SEED)
    for i, src in enumerate(random.sample(all_plots, 6), start=1):
        dst = os.path.join(README_DIR, f"readme_plot_{i:02d}.png")
        shutil.copy(src, dst)

    print(f"\n[visual_report.py] Done.")
    print(f"  42 plots saved to output/plots/")
    print(f"  6 random plots saved to readme_plots/")


if __name__ == "__main__":
    run()