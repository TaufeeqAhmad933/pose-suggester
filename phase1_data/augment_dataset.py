"""
Phase 1 - Step 2: Data Augmentation
--------------------------------------
Takes each raw image + its keypoint annotation and produces
augmented copies so each subcategory has ~300 samples.

Augmentations applied:
  - Horizontal flip  (keypoints mirrored correctly)
  - Brightness / contrast jitter
  - HSV hue/saturation shift
  - Gaussian blur  (mild)
  - Random crop + resize
  - Grid distortion  (subtle)

Output mirrors the dataset/ structure under augmented/:
  augmented/
    cafe_indoor/full_body/
      img_001.jpg          (original copy)
      img_001_aug_0.jpg
      img_001_aug_1.jpg
      ...
    ...
  augmented_annotations/   (matching JSONs with updated keypoints)

Usage:
    python augment_dataset.py
"""

import cv2
import json
import os
import random
from pathlib import Path

import albumentations as A
import numpy as np
from tqdm import tqdm

# ── Config ────────────────────────────────────────────────────────────────────
DATASET_DIR      = Path("dataset")
ANNOTATIONS_DIR  = Path("annotations")
AUG_IMG_DIR      = Path("augmented")
AUG_ANN_DIR      = Path("augmented_annotations")

TARGET_PER_SUBCAT = 300   # target images per category/subcategory
SEED              = 42

CATEGORIES    = ["cafe_indoor", "nature", "urban_street"]
SUBCATEGORIES = ["full_body", "waist_up"]

random.seed(SEED)

# ── MediaPipe left↔right landmark swap table for horizontal flip ──────────────
# When we flip horizontally, left and right body parts swap.
FLIP_PAIRS = [
    (1, 4), (2, 5), (3, 6),         # eyes
    (7, 8),                          # ears
    (9, 10),                         # mouth
    (11, 12), (13, 14), (15, 16),   # shoulder, elbow, wrist
    (17, 18), (19, 20), (21, 22),   # pinky, index, thumb
    (23, 24), (25, 26), (27, 28),   # hip, knee, ankle
    (29, 30), (31, 32),             # heel, foot
]

LANDMARK_NAMES = [
    "nose", "left_eye_inner", "left_eye", "left_eye_outer",
    "right_eye_inner", "right_eye", "right_eye_outer",
    "left_ear", "right_ear", "mouth_left", "mouth_right",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_pinky", "right_pinky",
    "left_index", "right_index", "left_thumb", "right_thumb",
    "left_hip", "right_hip", "left_knee", "right_knee",
    "left_ankle", "right_ankle", "left_heel", "right_heel",
    "left_foot_index", "right_foot_index"
]


# ── Augmentation pipeline ─────────────────────────────────────────────────────
def build_pipeline(is_flip: bool = False) -> A.Compose:
    transforms = [
        A.RandomBrightnessContrast(brightness_limit=0.25, contrast_limit=0.25, p=0.7),
        A.HueSaturationValue(hue_shift_limit=15, sat_shift_limit=30, val_shift_limit=15, p=0.6),
        A.GaussianBlur(blur_limit=(3, 5), p=0.3),
        A.RandomResizedCrop(height=512, width=512, scale=(0.85, 1.0), p=0.5),
        A.GridDistortion(num_steps=5, distort_limit=0.1, p=0.3),
    ]
    if is_flip:
        transforms.insert(0, A.HorizontalFlip(p=1.0))

    return A.Compose(transforms)


def flip_keypoints(keypoints: dict) -> dict:
    """Mirror x-coords and swap left/right landmark names."""
    if not keypoints:
        return keypoints

    kp_list = [keypoints.get(name, {}) for name in LANDMARK_NAMES]

    # Mirror x
    for kp in kp_list:
        if kp:
            kp["x"] = round(1.0 - kp["x"], 5)

    # Swap left↔right
    for i, j in FLIP_PAIRS:
        kp_list[i], kp_list[j] = kp_list[j], kp_list[i]

    return {LANDMARK_NAMES[i]: kp_list[i] for i in range(len(LANDMARK_NAMES))}


def augment_image(img: np.ndarray, pipeline: A.Compose) -> np.ndarray:
    result = pipeline(image=img)
    return result["image"]


def save_image(img: np.ndarray, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), img)


def save_annotation(annotation: dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(annotation, f, indent=2)


# ── Main ──────────────────────────────────────────────────────────────────────
def augment_dataset():
    total_generated = 0

    for category in CATEGORIES:
        for subcategory in SUBCATEGORIES:
            src_img_dir = DATASET_DIR / category / subcategory
            src_ann_dir = ANNOTATIONS_DIR / category / subcategory

            if not src_img_dir.exists():
                print(f"[WARN] Missing: {src_img_dir}, skipping.")
                continue

            image_files = sorted([
                f for f in src_img_dir.iterdir()
                if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
            ])

            if not image_files:
                print(f"[WARN] No images in {src_img_dir}")
                continue

            n_originals = len(image_files)
            augs_needed  = max(0, TARGET_PER_SUBCAT - n_originals)
            augs_per_img = (augs_needed // n_originals) + 1

            print(f"\n📂 {category}/{subcategory}  "
                  f"({n_originals} originals → targeting {TARGET_PER_SUBCAT})")

            for img_path in tqdm(image_files, desc="  Augmenting"):
                # --- Copy original ---
                img_bgr = cv2.imread(str(img_path))
                if img_bgr is None:
                    continue

                stem = img_path.stem
                ann_path = src_ann_dir / (stem + ".json")
                annotation = {}
                if ann_path.exists():
                    with open(ann_path) as f:
                        annotation = json.load(f)

                # Save original to augmented dir
                save_image(img_bgr,
                    AUG_IMG_DIR / category / subcategory / img_path.name)
                if annotation:
                    save_annotation(annotation,
                        AUG_ANN_DIR / category / subcategory / (stem + ".json"))

                # --- Generate augmented copies ---
                for aug_idx in range(augs_per_img):
                    use_flip = (aug_idx % 2 == 1)   # every other aug uses flip
                    pipeline = build_pipeline(is_flip=use_flip)
                    aug_img  = augment_image(img_bgr, pipeline)

                    aug_kp = annotation.get("keypoints", {})
                    if use_flip:
                        aug_kp = flip_keypoints(aug_kp)

                    aug_name = f"{stem}_aug_{aug_idx}"
                    save_image(aug_img,
                        AUG_IMG_DIR / category / subcategory / (aug_name + ".jpg"))

                    aug_ann = {
                        **annotation,
                        "image": f"{category}/{subcategory}/{aug_name}.jpg",
                        "augmented": True,
                        "aug_index": aug_idx,
                        "flipped": use_flip,
                        "keypoints": aug_kp
                    }
                    save_annotation(aug_ann,
                        AUG_ANN_DIR / category / subcategory / (aug_name + ".json"))

                    total_generated += 1

    print(f"\n✅ Augmentation complete.  Total new images generated: {total_generated}")
    print(f"   Augmented images → {AUG_IMG_DIR}/")
    print(f"   Augmented annotations → {AUG_ANN_DIR}/")


if __name__ == "__main__":
    augment_dataset()