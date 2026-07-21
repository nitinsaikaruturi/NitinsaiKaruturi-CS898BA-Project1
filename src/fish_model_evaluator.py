"""
fish_model_evaluator.py
CS 898BA - Homework 3, Part 5
Evaluates the baseline and tuned models on the held-out test set.
Generates classification reports, a confusion matrix, and a combined
visualization grid for the README.
"""

import os
import json
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------------------------------------
# Config
# ---------------------------------------------------------
DATA_DIR = os.path.join("hw3_data", "Fish")
OUTPUT_DIR = os.path.join("hw3_output")
PLOTS_DIR = os.path.join("hw3_plots")
IMG_SIZE = (128, 128)
BATCH_SIZE = 32
SEED = 42

# ---------------------------------------------------------
# Step 1: Rebuild the exact same test split used before
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

def make_dataset(paths, labels, batch_size):
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    ds = ds.map(load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds

test_ds = make_dataset(test_paths, test_labels, BATCH_SIZE)

# ---------------------------------------------------------
# Step 2: Load both trained models
# ---------------------------------------------------------
baseline_model = tf.keras.models.load_model(os.path.join(OUTPUT_DIR, "baseline_model.keras"))
tuned_model = tf.keras.models.load_model(os.path.join(OUTPUT_DIR, "best_tuned_model.keras"))

# ---------------------------------------------------------
# Step 3: Get predictions for both models
# ---------------------------------------------------------
def get_predictions(model, dataset):
    y_true, y_pred = [], []
    for images, labels_batch in dataset:
        preds = model.predict(images, verbose=0)
        y_pred.extend(np.argmax(preds, axis=1))
        y_true.extend(labels_batch.numpy())
    return np.array(y_true), np.array(y_pred)

print("Evaluating baseline model on test set...")
y_true_base, y_pred_base = get_predictions(baseline_model, test_ds)

print("Evaluating tuned model on test set...")
y_true_tuned, y_pred_tuned = get_predictions(tuned_model, test_ds)

# ---------------------------------------------------------
# Step 4: Classification reports
# ---------------------------------------------------------
report_baseline = classification_report(y_true_base, y_pred_base, target_names=class_names, digits=3)
report_tuned = classification_report(y_true_tuned, y_pred_tuned, target_names=class_names, digits=3)

print("\n=== BASELINE MODEL - Test Set Classification Report ===")
print(report_baseline)

print("\n=== TUNED MODEL - Test Set Classification Report ===")
print(report_tuned)

report_path = os.path.join(OUTPUT_DIR, "classification_reports.txt")
with open(report_path, "w") as f:
    f.write("=== BASELINE MODEL - Test Set Classification Report ===\n")
    f.write(report_baseline)
    f.write("\n\n=== TUNED MODEL - Test Set Classification Report ===\n")
    f.write(report_tuned)
print(f"\nSaved classification reports to {report_path}")

# ---------------------------------------------------------
# Step 5: Confusion matrix for the tuned model
# ---------------------------------------------------------
cm = confusion_matrix(y_true_tuned, y_pred_tuned)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Tuned Model (Test Set)")
plt.tight_layout()
cm_path = os.path.join(PLOTS_DIR, "confusion_matrix_tuned.png")
plt.savefig(cm_path, bbox_inches="tight")
plt.close()
print(f"Saved confusion matrix to {cm_path}")

# ---------------------------------------------------------
# Step 6: Load hyperparameter search results for reference
# ---------------------------------------------------------
with open(os.path.join(OUTPUT_DIR, "hyperparameter_search_results.json")) as f:
    hp_results = json.load(f)

best_run = min(hp_results, key=lambda r: r["final_val_loss"])
print("\n=== Best hyperparameter configuration (for reference) ===")
print(best_run)

print("\nEvaluation complete.")