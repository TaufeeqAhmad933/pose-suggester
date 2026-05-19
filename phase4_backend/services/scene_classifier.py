"""
services/scene_classifier.py
------------------------------
Wraps the trained MobileNetV2 model from Phase 2.
Returns predicted scene label + confidence.
"""

import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras

IMG_SIZE   = (224, 224)
MODELS_DIR = Path(__file__).parent.parent / "models"


class SceneClassifier:
    def __init__(self):
        model_path = MODELS_DIR / "scene_classifier.keras"
        index_path = MODELS_DIR / "class_indices.json"

        if not model_path.exists():
            raise FileNotFoundError(
                f"Scene classifier model not found at {model_path}.\n"
                "Run phase2_classifier/train.py first."
            )

        self._model = keras.models.load_model(str(model_path))

        with open(index_path) as f:
            class_index = json.load(f)
        # Invert: index → label
        self._idx_to_class = {v: k for k, v in class_index.items()}

    def predict(self, image_bytes: bytes) -> dict:
        """
        Returns:
        {
          "label": "cafe_indoor__full_body",
          "category": "cafe_indoor",
          "subcategory": "full_body",
          "confidence": 0.94,
          "top_3": [{"label": ..., "confidence": ...}, ...]
        }
        """
        import cv2
        nparr = np.frombuffer(image_bytes, np.uint8)
        img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, IMG_SIZE)

        img_array = np.expand_dims(img_resized.astype(np.float32), axis=0)
        probs = self._model.predict(img_array, verbose=0)[0]

        top_indices = np.argsort(probs)[::-1][:3]
        top_label   = self._idx_to_class[top_indices[0]]
        category, subcategory = top_label.split("__")

        return {
            "label": top_label,
            "category": category,
            "subcategory": subcategory,
            "confidence": float(probs[top_indices[0]]),
            "top_3": [
                {
                    "label": self._idx_to_class[i],
                    "confidence": float(probs[i])
                }
                for i in top_indices
            ],
        }


# Singleton
_classifier_instance: SceneClassifier | None = None


def get_scene_classifier() -> SceneClassifier:
    global _classifier_instance
    if _classifier_instance is None:
        _classifier_instance = SceneClassifier()
    return _classifier_instance