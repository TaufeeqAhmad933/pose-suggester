"""
Phase 1 - Step 1: Extract Pose Keypoints
-----------------------------------------
Runs MediaPipe Pose Landmarker (Tasks API, v0.10.13+) on every image in
dataset/ and saves a JSON annotation file per image in annotations/.

FIRST-TIME SETUP — download the model file once:
    curl -L "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task" \
         -o pose_landmarker.task

Annotation format:
{
  "image": "cafe_indoor/full_body/img_001.jpg",
  "category": "cafe_indoor",
  "subcategory": "full_body",
  "keypoints": {
      "nose":          {"x": 0.51, "y": 0.12, "visibility": 0.99},
      "left_shoulder": {"x": 0.42, "y": 0.28, "visibility": 0.98},
      ...  (33 landmarks total)
  },
  "person_detected": true
}

Usage:
    python extract_keypoints.py
    python extract_keypoints.py --model path/to/pose_landmarker.task
"""

import argparse
import cv2
import json
import os
from pathlib import Path

import mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision as mp_vision
import numpy as np
from tqdm import tqdm

# ── Config ─────────────────────────────────────────────────────────────────────
DATASET_DIR     = Path("dataset")
ANNOTATIONS_DIR = Path("annotations")
ANNOTATIONS_DIR.mkdir(exist_ok=True)

CATEGORIES    = ["cafe_indoor", "nature", "urban_street"]
SUBCATEGORIES = ["full_body", "waist_up"]
DEFAULT_MODEL = "pose_landmarker.task"

# MediaPipe landmark names (33 total — matches PoseLandmark enum order)
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


def build_landmarker(model_path: str):
    """Create a PoseLandmarker for static image inference."""
    base_options = mp_tasks.BaseOptions(model_asset_path=model_path)
    options = mp_vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=mp_vision.RunningMode.IMAGE,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        output_segmentation_masks=False,
    )
    return mp_vision.PoseLandmarker.create_from_options(options)


def extract_keypoints_from_image(landmarker, image_path: Path) -> dict | None:
    """
    Returns dict of {landmark_name: {x, y, z, visibility}} or None.
    x/y are normalised [0,1]; z is depth relative to hip midpoint.
    """
    img_bgr = cv2.imread(str(image_path))
    if img_bgr is None:
        return None

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)

    result = landmarker.detect(mp_image)
    if not result.pose_landmarks or len(result.pose_landmarks) == 0:
        return None

    # Use first detected pose
    landmarks = result.pose_landmarks[0]

    # Visibility comes from pose_world_landmarks if available
    world = result.pose_world_landmarks[0] if result.pose_world_landmarks else None

    keypoints = {}
    for idx, lm in enumerate(landmarks):
        vis = world[idx].visibility if world else 1.0
        keypoints[LANDMARK_NAMES[idx]] = {
            "x": round(lm.x, 5),
            "y": round(lm.y, 5),
            "z": round(lm.z, 5),
            "visibility": round(float(vis) if vis is not None else 1.0, 5),
        }
    return keypoints


def process_dataset(model_path: str):
    if not Path(model_path).exists():
        print(f"""
ERROR: Model file not found: {model_path}

Download it once with:
  curl -L "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task" \\
       -o pose_landmarker.task
""")
        return

    landmarker = build_landmarker(model_path)
    total, detected, skipped = 0, 0, 0
    failed_images = []

    for category in CATEGORIES:
        for subcategory in SUBCATEGORIES:
            folder = DATASET_DIR / category / subcategory
            if not folder.exists():
                print(f"  [WARN] Folder not found, skipping: {folder}")
                continue

            image_files = sorted([
                f for f in folder.iterdir()
                if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
            ])

            print(f"\n📂 {category}/{subcategory}  ({len(image_files)} images)")

            for img_path in tqdm(image_files, desc="  Extracting"):
                total += 1
                rel_path = str(img_path.relative_to(DATASET_DIR))

                keypoints = extract_keypoints_from_image(landmarker, img_path)
                person_detected = keypoints is not None

                if person_detected:
                    detected += 1
                else:
                    skipped += 1
                    failed_images.append(rel_path)

                annotation = {
                    "image": rel_path,
                    "category": category,
                    "subcategory": subcategory,
                    "person_detected": person_detected,
                    "keypoints": keypoints or {},
                }

                out_dir = ANNOTATIONS_DIR / category / subcategory
                out_dir.mkdir(parents=True, exist_ok=True)
                out_file = out_dir / (img_path.stem + ".json")
                with open(out_file, "w") as f:
                    json.dump(annotation, f, indent=2)

    landmarker.close()

    print("\n" + "="*50)
    print(f"✅ Done!  Total: {total}  |  Detected: {detected}  |  Skipped: {skipped}")
    if failed_images:
        print(f"\n⚠️  No person detected in {len(failed_images)} image(s):")
        for f in failed_images:
            print(f"   - {f}")
        print("\n  → Replace these images or improve photo quality.")
    print("="*50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help="Path to pose_landmarker.task model file")
    args = parser.parse_args()
    process_dataset(args.model)