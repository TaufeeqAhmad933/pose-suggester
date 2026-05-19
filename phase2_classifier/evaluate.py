"""
Phase 2 — Evaluate Scene Classifier
--------------------------------------
Generates:
  - Confusion matrix plot
  - Per-class accuracy report
  - Overall val accuracy

Usage:
    python evaluate.py
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow import keras

MODELS_DIR = Path("models")
IMG_SIZE = (224, 224)


def evaluate():

    print("=" * 60)
    print("   Evaluating Scene Classifier")
    print("=" * 60)

    # ----------------------------
    # Load trained model
    # ----------------------------
    model = keras.models.load_model(
        str(MODELS_DIR / "scene_classifier.keras")
    )

    # ----------------------------
    # Load class names
    # ----------------------------
    with open(MODELS_DIR / "class_indices.json") as f:
        class_index = json.load(f)

    class_names = [
        k for k, v in sorted(class_index.items(), key=lambda x: x[1])
    ]

    print(f"\nClasses: {class_names}")

    # ----------------------------
    # Validation dataset
    # ----------------------------
    flat_dir = Path("flat_data")

    val_ds = keras.utils.image_dataset_from_directory(
        str(flat_dir),
        validation_split=0.2,
        subset="validation",
        seed=42,
        image_size=IMG_SIZE,
        batch_size=32,
        label_mode="int",
        shuffle=True,
    )

    # ----------------------------
    # Get TRUE labels safely
    # ----------------------------
    y_true = np.concatenate(
        [labels.numpy() for _, labels in val_ds],
        axis=0
    )

    # ----------------------------
    # Predictions
    # ----------------------------
    y_pred_probs = model.predict(val_ds)

    y_pred = np.argmax(y_pred_probs, axis=1)

    # ----------------------------
    # Classification report
    # ----------------------------
    print("\n📊 Classification Report:\n")

    print(
        classification_report(
            y_true,
            y_pred,
            labels=np.arange(len(class_names)),
            target_names=class_names,
            zero_division=0,
        )
    )

    # ----------------------------
    # Confusion Matrix
    # ----------------------------
    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=np.arange(len(class_names))
    )

    plt.figure(figsize=(10, 8))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )

    plt.title("Confusion Matrix — Scene Classifier")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")

    plt.xticks(rotation=30, ha="right")

    plt.tight_layout()

    save_path = MODELS_DIR / "confusion_matrix.png"

    plt.savefig(save_path, dpi=150)

    print(f"\n✅ Confusion matrix saved → {save_path}")

    plt.show()

    # ----------------------------
    # Overall Accuracy
    # ----------------------------
    acc = np.mean(y_pred == y_true)

    print(f"\n✅ Overall Validation Accuracy: {acc:.2%}")


if __name__ == "__main__":
    evaluate()