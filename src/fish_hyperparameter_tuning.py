"""
fish_hyperparameter_tuning.py
CS 898BA - Homework 3, Part 4
Grid search over learning rate, batch size, and dropout rate.
Selects and saves the best-performing model based on validation loss.
"""

import os
import json
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------
# Config
# ---------------------------------------------------------
DATA_DIR = os.path.join("hw3_data", "Fish")
OUTPUT_DIR = os.path.join("hw3_output")
IMG_SIZE = (128, 128)
SEED = 42
EPOCHS = 10

LEARNING_RATES = [0.01, 0.001, 0.0001]
BATCH_SIZES = [32, 64]
DROPOUT_RATES = [0.3, 0.5]

os.makedirs(OUTPUT_DIR, exist_ok=True)

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

num_classes = len(class_names)

# ---------------------------------------------------------
# Step 3: Data pipeline builder (batch size varies per run)
# ---------------------------------------------------------
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

# ---------------------------------------------------------
# Step 4: Model builder (dropout varies per run)
# ---------------------------------------------------------
def build_model(dropout_rate, learning_rate):
    model = tf.keras.Sequential([
        tf.keras.Input(shape=(128, 128, 3)),

        tf.keras.layers.Conv2D(32, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D((2, 2)),

        tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D((2, 2)),

        tf.keras.layers.Conv2D(128, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D((2, 2)),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dropout(dropout_rate),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(num_classes, activation="softmax"),
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model

# ---------------------------------------------------------
# Step 5: Grid search loop
# ---------------------------------------------------------
results = []
best_val_loss = float("inf")
best_config = None
best_model = None

run_number = 1
total_runs = len(LEARNING_RATES) * len(BATCH_SIZES) * len(DROPOUT_RATES)

for lr in LEARNING_RATES:
    for batch_size in BATCH_SIZES:
        for dropout in DROPOUT_RATES:
            print(f"\n=== Run {run_number}/{total_runs}: lr={lr}, batch_size={batch_size}, dropout={dropout} ===")

            train_ds = make_dataset(train_paths, train_labels, batch_size, shuffle=True, augment=True)
            val_ds = make_dataset(val_paths, val_labels, batch_size)

            model = build_model(dropout, lr)
            history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, verbose=1)

            final_val_loss = history.history["val_loss"][-1]
            final_val_acc = history.history["val_accuracy"][-1]

            results.append({
                "learning_rate": lr,
                "batch_size": batch_size,
                "dropout": dropout,
                "final_val_loss": final_val_loss,
                "final_val_accuracy": final_val_acc,
            })

            if final_val_loss < best_val_loss:
                best_val_loss = final_val_loss
                best_config = results[-1]
                best_model = model

            run_number += 1

# ---------------------------------------------------------
# Step 6: Save results and best model
# ---------------------------------------------------------
results_path = os.path.join(OUTPUT_DIR, "hyperparameter_search_results.json")
with open(results_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved all run results to {results_path}")

best_model_path = os.path.join(OUTPUT_DIR, "best_tuned_model.keras")
best_model.save(best_model_path)
print(f"Saved best model to {best_model_path}")

print("\n=== BEST CONFIGURATION ===")
print(best_config)
print("\nHyperparameter tuning complete.")