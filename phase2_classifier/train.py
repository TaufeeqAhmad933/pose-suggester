"""
Phase 2 — Scene Classifier Training
--------------------------------------
Fine-tunes MobileNetV2 on the 6-class scene dataset:
  cafe_indoor/full_body, cafe_indoor/waist_up,
  nature/full_body,      nature/waist_up,
  urban_street/full_body, urban_street/waist_up

Output:
  models/scene_classifier.keras    ← full SavedModel
  models/class_indices.json        ← label → index mapping

Usage:
    python train.py
    python train.py --epochs 30 --batch_size 16
"""

import argparse
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import (EarlyStopping, ModelCheckpoint,
                                        ReduceLROnPlateau)

# ── Config ────────────────────────────────────────────────────────────────────
DATA_DIR   = Path("../phase1_data/augmented")
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)

IMG_SIZE    = (224, 224)   # MobileNetV2 native input
NUM_CLASSES = 6
SEED        = 42

tf.random.set_seed(SEED)


def build_class_label(folder_path: Path) -> str:
    """Converts augmented/cafe_indoor/full_body → 'cafe_indoor__full_body'"""
    parts = folder_path.parts
    # last two parts are category/subcategory
    return "__".join(parts[-2:])


def load_dataset(data_dir: Path, batch_size: int, val_split: float = 0.2):
    """
    Loads images from augmented/ using keras image_dataset_from_directory.
    Folder structure must be flat — we create a flat symlink structure.
    """
    # Build a flat directory: flat_data/cafe_indoor__full_body/*.jpg
    flat_dir = Path("flat_data")
    flat_dir.mkdir(exist_ok=True)

    categories    = ["cafe_indoor", "nature", "urban_street"]
    subcategories = ["full_body", "waist_up"]
    class_names   = []

    for cat in categories:
        for sub in subcategories:
            src = data_dir / cat / sub
            label = f"{cat}__{sub}"
            class_names.append(label)
            dst = flat_dir / label
            dst.mkdir(exist_ok=True)

            if src.exists():
                for img in src.iterdir():
                    if img.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                        link = dst / img.name
                        if not link.exists():
                            try:
                                link.symlink_to(img.resolve())
                            except Exception:
                                import shutil
                                shutil.copy(str(img), str(link))

    # Save class index mapping
    class_index = {name: i for i, name in enumerate(sorted(class_names))}
    with open(MODELS_DIR / "class_indices.json", "w") as f:
        json.dump(class_index, f, indent=2)
    print(f"Classes: {class_index}")

    train_ds = keras.utils.image_dataset_from_directory(
        str(flat_dir),
        validation_split=val_split,
        subset="training",
        seed=SEED,
        image_size=IMG_SIZE,
        batch_size=batch_size,
        label_mode="categorical",
    )
    val_ds = keras.utils.image_dataset_from_directory(
        str(flat_dir),
        validation_split=val_split,
        subset="validation",
        seed=SEED,
        image_size=IMG_SIZE,
        batch_size=batch_size,
        label_mode="categorical",
    )

    # Performance optimisation
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds   = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    return train_ds, val_ds


def build_model(num_classes: int, freeze_base: bool = True) -> keras.Model:
    """
    MobileNetV2 base + custom classification head.
    Phase A: train head only (base frozen).
    Phase B: unfreeze top 30 layers for fine-tuning.
    """
    # Preprocessing built into model for portability
    inputs = layers.Input(shape=(*IMG_SIZE, 3))
    x = layers.Rescaling(1.0 / 127.5, offset=-1)(inputs)  # [-1, 1] normalisation

    base = MobileNetV2(
        input_shape=(*IMG_SIZE, 3),
        include_top=False,
        weights="imagenet",
    )
    base.trainable = not freeze_base

    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = keras.Model(inputs, outputs)
    return model, base


def plot_history(history, title: str, save_path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(title, fontsize=14)

    axes[0].plot(history.history["accuracy"],     label="Train")
    axes[0].plot(history.history["val_accuracy"], label="Val")
    axes[0].set_title("Accuracy"); axes[0].legend(); axes[0].grid(True)

    axes[1].plot(history.history["loss"],     label="Train")
    axes[1].plot(history.history["val_loss"], label="Val")
    axes[1].set_title("Loss"); axes[1].legend(); axes[1].grid(True)

    plt.tight_layout()
    plt.savefig(str(save_path), dpi=150)
    plt.close()
    print(f"  Plot saved → {save_path}")


def train(epochs_head: int = 20, epochs_finetune: int = 15, batch_size: int = 32):
    print("=" * 55)
    print("  Phase 2 — Scene Classifier Training")
    print("=" * 55)

    train_ds, val_ds = load_dataset(DATA_DIR, batch_size)

    # ── Phase A: Train head only ──────────────────────────────────────────────
    print(f"\n📚 Phase A: Training classification head ({epochs_head} epochs)")
    model, base = build_model(NUM_CLASSES, freeze_base=True)
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()

    callbacks_a = [
        EarlyStopping(patience=5, restore_best_weights=True, verbose=1),
        ModelCheckpoint(str(MODELS_DIR / "best_head.keras"),
                        save_best_only=True, verbose=1),
        ReduceLROnPlateau(patience=3, factor=0.5, verbose=1),
    ]

    history_a = model.fit(
        train_ds, validation_data=val_ds,
        epochs=epochs_head, callbacks=callbacks_a
    )
    plot_history(history_a, "Phase A — Head Training",
                 MODELS_DIR / "history_phase_a.png")

    # ── Phase B: Fine-tune top layers of base ─────────────────────────────────
    print(f"\n🔧 Phase B: Fine-tuning top 30 base layers ({epochs_finetune} epochs)")
    base.trainable = True
    for layer in base.layers[:-30]:
        layer.trainable = False

    model.compile(
        optimizer=keras.optimizers.Adam(1e-5),  # lower LR for fine-tune
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    callbacks_b = [
        EarlyStopping(patience=7, restore_best_weights=True, verbose=1),
        ModelCheckpoint(str(MODELS_DIR / "scene_classifier.keras"),
                        save_best_only=True, verbose=1),
        ReduceLROnPlateau(patience=3, factor=0.5, verbose=1),
    ]

    history_b = model.fit(
        train_ds, validation_data=val_ds,
        epochs=epochs_finetune, callbacks=callbacks_b,
        initial_epoch=0,
    )
    plot_history(history_b, "Phase B — Fine-tuning",
                 MODELS_DIR / "history_phase_b.png")

    # ── Final save ────────────────────────────────────────────────────────────
    model.save(str(MODELS_DIR / "scene_classifier.keras"))
    print(f"\n✅ Model saved → {MODELS_DIR / 'scene_classifier.keras'}")

    # Final val accuracy
    loss, acc = model.evaluate(val_ds, verbose=0)
    print(f"   Final val accuracy: {acc:.2%}  |  Loss: {loss:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",      type=int, default=20)
    parser.add_argument("--finetune_epochs", type=int, default=15)
    parser.add_argument("--batch_size",  type=int, default=32)
    args = parser.parse_args()

    train(
        epochs_head=args.epochs,
        epochs_finetune=args.finetune_epochs,
        batch_size=args.batch_size,
    )