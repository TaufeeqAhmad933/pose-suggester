"""
Phase 2 — Single Image Prediction
Usage:
    python predict.py --image path/to/photo.jpg
"""

import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras

MODELS_DIR = Path("models")
IMG_SIZE   = (224, 224)


def predict(image_path: str) -> dict:
    model = keras.models.load_model(str(MODELS_DIR / "scene_classifier.keras"))

    with open(MODELS_DIR / "class_indices.json") as f:
        class_index = json.load(f)
    idx_to_class = {v: k for k, v in class_index.items()}

    img = keras.utils.load_img(image_path, target_size=IMG_SIZE)
    img_array = keras.utils.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)   # (1, 224, 224, 3)

    probs = model.predict(img_array, verbose=0)[0]
    top_idx = np.argsort(probs)[::-1][:3]

    results = {
        "top_prediction": idx_to_class[top_idx[0]],
        "confidence": float(probs[top_idx[0]]),
        "top_3": [
            {"label": idx_to_class[i], "confidence": float(probs[i])}
            for i in top_idx
        ]
    }
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    args = parser.parse_args()

    result = predict(args.image)
    print(f"\n🏷️  Scene: {result['top_prediction']}  ({result['confidence']:.1%})")
    print("\n  Top 3:")
    for r in result["top_3"]:
        print(f"    {r['label']:<35} {r['confidence']:.1%}")