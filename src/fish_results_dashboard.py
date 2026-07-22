"""
fish_results_dashboard.py
CS 898BA - Homework 3, Part 5
Builds a combined visualization grid: baseline vs tuned training curves,
plus the tuned model's confusion matrix, for the README.
"""

import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

DATA_DIR = os.path.join("hw3_data", "Fish")
OUTPUT_DIR = os.path.join("hw3_output")
PLOTS_DIR = os.path.join("hw3_plots")
IMG_SIZE = (128, 128)
SEED = 42
EPOCHS = 10  # matches the tuned model's training length

# ---------------------------------------------------------
# Step 1: Rebuild data splits (same seed as before)
# ---------------------------------------------------------
class_names = sorted(os.listdir(DATA_DIR))
filepaths, labels = [], []

for class_index, class_name in enumerate(class_names):
    class_folder = os.path.join(DATA_DIR, class_name)
    for fname in os.listdir(class_folder):
        filepaths.append(os.path.join(class_folder, fname))
        labels.append(class_index)

filepaths = np.array(filepaths)
labels = np.array(labels)

train_paths, temp_paths, train_labels, temp_labels = train_test_split(
    filepaths, labels, test_size=0.30, stratify=labels, random_state=SEED
)
val_paths, test_paths, val_labels, test_labels = train_test_split(
    temp_paths, temp_labels, test_size=0.50, stratify=temp_labels, random_state=SEED
)

def load_and_preprocess(path, label):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, IMG_SIZE)
    image = image / 255.0
    return image, label

def make_dataset(paths, labels, batch_size, shuffle=False, augment=False):
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(paths), seed=SEED)
    ds = ds.map(load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    if augment:
        aug = tf.keras.Sequential([
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.08),
            tf.keras.layers.RandomBrightness(0.15, value_range=(0.0, 1.0)),
        ])
        ds = ds.map(lambda x, y: (aug(x, training=True), y), num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds

num_classes = len(class_names)

# ---------------------------------------------------------
# Step 2: Retrain tuned model briefly just to capture its history
# (best_tuned_model.keras has no saved history, so we rebuild
# the identical architecture/config and retrain to get curves)
# ---------------------------------------------------------
train_ds_tuned = make_dataset(train_paths, train_labels, batch_size=64, shuffle=True, augment=True)
val_ds_tuned = make_dataset(val_paths, val_labels, batch_size=64)

tuned_model = tf.keras.Sequential([
    tf.keras.Input(shape=(128, 128, 3)),
    tf.keras.layers.Conv2D(32, (3, 3), activation="relu"),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(128, (3, 3), activation="relu"),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dense(num_classes, activation="softmax"),
])
tuned_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

print("Retraining tuned config to capture history for plotting...")
tuned_history = tuned_model.fit(train_ds_tuned, validation_data=val_ds_tuned, epochs=EPOCHS, verbose=1)

# ---------------------------------------------------------
# Step 3: Build the combined dashboard image
# ---------------------------------------------------------
fig = plt.figure(figsize=(16, 10))

# Baseline curves come from the already-saved plot image
baseline_curves_img = mpimg.imread(os.path.join(PLOTS_DIR, "baseline_training_curves.png"))
ax1 = plt.subplot(2, 1, 1)
ax1.imshow(baseline_curves_img)
ax1.axis("off")
ax1.set_title("Baseline Model: Training/Validation Curves", fontsize=13)

# Tuned model curves (freshly captured)
ax2 = plt.subplot(2, 2, 3)
ax2.plot(tuned_history.history["loss"], label="Train Loss")
ax2.plot(tuned_history.history["val_loss"], label="Val Loss")
ax2.set_title("Tuned Model: Loss Curves")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Loss")
ax2.legend()

ax3 = plt.subplot(2, 2, 4)
ax3.plot(tuned_history.history["accuracy"], label="Train Accuracy")
ax3.plot(tuned_history.history["val_accuracy"], label="Val Accuracy")
ax3.set_title("Tuned Model: Accuracy Curves")
ax3.set_xlabel("Epoch")
ax3.set_ylabel("Accuracy")
ax3.legend()

plt.tight_layout()
dashboard_path = os.path.join(PLOTS_DIR, "results_dashboard.png")
plt.savefig(dashboard_path, bbox_inches="tight", dpi=150)
print(f"Saved combined results dashboard to {dashboard_path}")

print("Dashboard generation complete.")