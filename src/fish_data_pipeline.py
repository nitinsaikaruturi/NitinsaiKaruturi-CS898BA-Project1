"""
fish_data_pipeline.py
CS 898BA - Homework 3, Part 2
Loads the Fish dataset, creates stratified train/val/test splits,
resizes + normalizes images, and applies data augmentation to the
training set.
"""

import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# Config
# ---------------------------------------------------------
DATA_DIR = os.path.join("hw3_data", "Fish")
OUTPUT_DIR = os.path.join("hw3_output")
PLOTS_DIR = os.path.join("hw3_plots")
IMG_SIZE = (128, 128)
BATCH_SIZE = 32
SEED = 42

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

# ---------------------------------------------------------
# Step 1: Gather filepaths and labels from folder structure
# ---------------------------------------------------------
class_names = sorted(os.listdir(DATA_DIR))
filepaths = []
labels = []

for class_index, class_name in enumerate(class_names):
    class_folder = os.path.join(DATA_DIR, class_name)
    for fname in os.listdir(class_folder):
        filepaths.append(os.path.join(class_folder, fname))
        labels.append(class_index)

filepaths = np.array(filepaths)
labels = np.array(labels)

print(f"Found {len(filepaths)} images across {len(class_names)} classes: {class_names}")

# ---------------------------------------------------------
# Step 2: Stratified split -> 70% train, 15% val, 15% test
# ---------------------------------------------------------
train_paths, temp_paths, train_labels, temp_labels = train_test_split(
    filepaths, labels, test_size=0.30, stratify=labels, random_state=SEED
)
val_paths, test_paths, val_labels, test_labels = train_test_split(
    temp_paths, temp_labels, test_size=0.50, stratify=temp_labels, random_state=SEED
)

print(f"Train: {len(train_paths)} | Val: {len(val_paths)} | Test: {len(test_paths)}")

# ---------------------------------------------------------
# Step 3: Build tf.data pipelines (load, resize, normalize)
# ---------------------------------------------------------
def load_and_preprocess(path, label):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, IMG_SIZE)
    image = image / 255.0  # normalize to [0, 1]
    return image, label

def make_dataset(paths, labels, shuffle=False):
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(paths), seed=SEED)
    ds = ds.map(load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(BATCH_SIZE)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds

train_ds = make_dataset(train_paths, train_labels, shuffle=True)
val_ds = make_dataset(val_paths, val_labels)
test_ds = make_dataset(test_paths, test_labels)

# ---------------------------------------------------------
# Step 4: Data augmentation (applied only to training set)
# ---------------------------------------------------------
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.08),
    tf.keras.layers.RandomBrightness(0.15, value_range=(0.0, 1.0)),
])

def augment(image, label):
    image = data_augmentation(image, training=True)
    return image, label

train_ds_augmented = train_ds.map(augment, num_parallel_calls=tf.data.AUTOTUNE)

# ---------------------------------------------------------
# Step 5: Save a visual sample of augmented training images
# ---------------------------------------------------------
sample_images, sample_labels = next(iter(train_ds_augmented.take(1)))

plt.figure(figsize=(10, 10))
for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(sample_images[i].numpy())
    plt.title(class_names[sample_labels[i].numpy()])
    plt.axis("off")

plt.suptitle("Sample Augmented Training Images")
save_path = os.path.join(PLOTS_DIR, "augmentation_sample.png")
plt.savefig(save_path, bbox_inches="tight")
print(f"Saved augmentation sample grid to {save_path}")

# ---------------------------------------------------------
# Step 6: Report class distribution across splits
# ---------------------------------------------------------
def report_split(name, split_labels):
    counts = {class_names[i]: int(np.sum(split_labels == i)) for i in range(len(class_names))}
    print(f"{name} class distribution: {counts}")

report_split("Train", train_labels)
report_split("Val", val_labels)
report_split("Test", test_labels)

print("Data pipeline complete.")