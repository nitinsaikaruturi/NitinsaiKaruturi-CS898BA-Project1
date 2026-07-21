"""
fish_cnn_baseline.py
CS 898BA - Homework 3, Part 3
Builds and trains a baseline CNN for fish species classification,
then saves the model weights and training curves.
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
EPOCHS = 20

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

# ---------------------------------------------------------
# Step 1: Gather filepaths and labels
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

# ---------------------------------------------------------
# Step 2: Stratified split -> 70% train, 15% val, 15% test
# ---------------------------------------------------------
train_paths, temp_paths, train_labels, temp_labels = train_test_split(
    filepaths, labels, test_size=0.30, stratify=labels, random_state=SEED
)
val_paths, test_paths, val_labels, test_labels = train_test_split(
    temp_paths, temp_labels, test_size=0.50, stratify=temp_labels, random_state=SEED
)

# ---------------------------------------------------------
# Step 3: Build tf.data pipelines
# ---------------------------------------------------------
def load_and_preprocess(path, label):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, IMG_SIZE)
    image = image / 255.0
    return image, label

def make_dataset(paths, labels, shuffle=False, augment=False):
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
    ds = ds.batch(BATCH_SIZE)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds

train_ds = make_dataset(train_paths, train_labels, shuffle=True, augment=True)
val_ds = make_dataset(val_paths, val_labels)
test_ds = make_dataset(test_paths, test_labels)

# ---------------------------------------------------------
# Step 4: Build the baseline CNN
# ---------------------------------------------------------
num_classes = len(class_names)

model = tf.keras.Sequential([
    tf.keras.Input(shape=(128, 128, 3)),

    tf.keras.layers.Conv2D(32, (3, 3), activation="relu"),
    tf.keras.layers.MaxPooling2D((2, 2)),

    tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
    tf.keras.layers.MaxPooling2D((2, 2)),

    tf.keras.layers.Conv2D(128, (3, 3), activation="relu"),
    tf.keras.layers.MaxPooling2D((2, 2)),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dense(num_classes, activation="softmax"),
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

model.summary()

# ---------------------------------------------------------
# Step 5: Train the baseline model
# ---------------------------------------------------------
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
)

# ---------------------------------------------------------
# Step 6: Save the trained model
# ---------------------------------------------------------
model_path = os.path.join(OUTPUT_DIR, "baseline_model.keras")
model.save(model_path)
print(f"Saved baseline model to {model_path}")

# ---------------------------------------------------------
# Step 7: Plot and save training/validation curves
# ---------------------------------------------------------
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Val Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Baseline Model: Loss Curves")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history["accuracy"], label="Train Accuracy")
plt.plot(history.history["val_accuracy"], label="Val Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Baseline Model: Accuracy Curves")
plt.legend()

plt.tight_layout()
curve_path = os.path.join(PLOTS_DIR, "baseline_training_curves.png")
plt.savefig(curve_path, bbox_inches="tight")
print(f"Saved training curves to {curve_path}")

print("Baseline CNN training complete.")