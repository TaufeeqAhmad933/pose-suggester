"""
Phase 1 - Step 3: Verify Dataset
----------------------------------
Sanity checks the dataset and annotations:
  1. Counts images per subcategory (raw + augmented)
  2. Checks annotation JSON exists for each image
  3. Reports how many images had no person detected
  4. Visualises keypoint skeletons on a random sample of images

Usage:
    python verify_dataset.py              # check counts only
    python verify_dataset.py --visualise  # also show skeleton overlays
"""

import argparse
import json
import random
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

# ── Config ────────────────────────────────────────────────────────────────────
DATASET_DIR     = Path("dataset")
ANNOTATIONS_DIR = Path("annotations")
AUG_IMG_DIR     = Path("augmented")
AUG_ANN_DIR     = Path("augmented_annotations")

CATEGORIES    = ["cafe_indoor", "nature", "urban_street"]
SUBCATEGORIES = ["full_body", "waist_up"]

# Skeleton connections (MediaPipe pairs to draw lines between)
SKELETON_CONNECTIONS = [
    ("left_shoulder",  "right_shoulder"),
    ("left_shoulder",  "left_elbow"),
    ("left_elbow",     "left_wrist"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow",    "right_wrist"),
    ("left_shoulder",  "left_hip"),
    ("right_shoulder", "right_hip"),
    ("left_hip",       "right_hip"),
    ("left_hip",       "left_knee"),
    ("left_knee",      "left_ankle"),
    ("right_hip",      "right_knee"),
    ("right_knee",     "right_ankle"),
    ("nose",           "left_shoulder"),
    ("nose",           "right_shoulder"),
]

JOINT_COLOR = (0, 255, 128)    # green
BONE_COLOR  = (255, 200, 0)    # yellow


def draw_skeleton(img: np.ndarray, keypoints: dict) -> np.ndarray:
    """Draw skeleton joints and bones on a copy of the image."""
    vis = img.copy()
    h, w = vis.shape[:2]

    # Draw bones
    for (a, b) in SKELETON_CONNECTIONS:
        kp_a = keypoints.get(a)
        kp_b = keypoints.get(b)
        if kp_a and kp_b and kp_a.get("visibility", 0) > 0.4 and kp_b.get("visibility", 0) > 0.4:
            pt_a = (int(kp_a["x"] * w), int(kp_a["y"] * h))
            pt_b = (int(kp_b["x"] * w), int(kp_b["y"] * h))
            cv2.line(vis, pt_a, pt_b, BONE_COLOR, 2, cv2.LINE_AA)

    # Draw joints
    for name, kp in keypoints.items():
        if kp.get("visibility", 0) > 0.4:
            pt = (int(kp["x"] * w), int(kp["y"] * h))
            cv2.circle(vis, pt, 5, JOINT_COLOR, -1, cv2.LINE_AA)

    return vis


def count_images(base_dir: Path) -> dict:
    counts = {}
    for category in CATEGORIES:
        for subcategory in SUBCATEGORIES:
            folder = base_dir / category / subcategory
            if folder.exists():
                imgs = [f for f in folder.iterdir()
                        if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}]
                counts[f"{category}/{subcategory}"] = len(imgs)
            else:
                counts[f"{category}/{subcategory}"] = 0
    return counts


def check_annotations(ann_dir: Path, img_dir: Path) -> tuple[int, int, list]:
    """Returns (matched, missing, no_person_list)."""
    matched, missing = 0, 0
    no_person = []

    for category in CATEGORIES:
        for subcategory in SUBCATEGORIES:
            img_folder = img_dir / category / subcategory
            if not img_folder.exists():
                continue
            for img_path in img_folder.iterdir():
                if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}:
                    continue
                ann_path = ann_dir / category / subcategory / (img_path.stem + ".json")
                if ann_path.exists():
                    matched += 1
                    with open(ann_path) as f:
                        ann = json.load(f)
                    if not ann.get("person_detected", True):
                        no_person.append(str(img_path))
                else:
                    missing += 1

    return matched, missing, no_person


def visualise_samples(n: int = 6):
    """Pick n random annotated images and show skeleton overlays."""
    samples = []
    for category in CATEGORIES:
        for subcategory in SUBCATEGORIES:
            ann_folder = ANNOTATIONS_DIR / category / subcategory
            img_folder = DATASET_DIR / category / subcategory
            if not ann_folder.exists():
                continue
            for ann_path in ann_folder.iterdir():
                if ann_path.suffix != ".json":
                    continue
                img_path = img_folder / (ann_path.stem + ".jpg")
                if not img_path.exists():
                    img_path = img_folder / (ann_path.stem + ".png")
                if img_path.exists():
                    samples.append((img_path, ann_path, f"{category}\n{subcategory}"))

    random.shuffle(samples)
    samples = samples[:n]

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle("Sample Keypoint Annotations", fontsize=16, fontweight="bold")

    for ax, (img_path, ann_path, label) in zip(axes.flat, samples):
        img_bgr = cv2.imread(str(img_path))
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        with open(ann_path) as f:
            ann = json.load(f)

        if ann.get("keypoints"):
            img_vis = draw_skeleton(img_rgb, ann["keypoints"])
        else:
            img_vis = img_rgb

        ax.imshow(img_vis)
        ax.set_title(label, fontsize=10)
        ax.axis("off")

    plt.tight_layout()
    plt.savefig("verify_samples.png", dpi=150)
    plt.show()
    print("  Saved → verify_samples.png")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--visualise", action="store_true",
                        help="Show skeleton overlays on random samples")
    args = parser.parse_args()

    print("=" * 55)
    print("  PoseSuggester — Dataset Verification")
    print("=" * 55)

    # Raw image counts
    print("\n📊 Raw Dataset Counts:")
    raw_counts = count_images(DATASET_DIR)
    for k, v in raw_counts.items():
        bar = "█" * (v // 2) + f"  {v}"
        status = "✅" if v >= 40 else "⚠️ "
        print(f"  {status}  {k:<30} {bar}")

    # Augmented counts
    print("\n📊 Augmented Dataset Counts:")
    aug_counts = count_images(AUG_IMG_DIR)
    for k, v in aug_counts.items():
        bar = "█" * (v // 10) + f"  {v}"
        status = "✅" if v >= 250 else ("⚠️ " if v > 0 else "❌")
        print(f"  {status}  {k:<30} {bar}")

    # Annotation coverage
    print("\n📋 Annotation Coverage (raw):")
    matched, missing, no_person = check_annotations(ANNOTATIONS_DIR, DATASET_DIR)
    print(f"  ✅ Annotated   : {matched}")
    print(f"  ❌ Missing JSON: {missing}")
    print(f"  ⚠️  No person   : {len(no_person)}")
    if no_person:
        print("  Images with no person detected:")
        for p in no_person:
            print(f"    - {p}")

    # Recommendations
    print("\n💡 Recommendations:")
    for k, v in raw_counts.items():
        if v < 40:
            print(f"  → {k}: only {v} images — need at least 40–50 before augmenting")
    if missing > 0:
        print(f"  → Run extract_keypoints.py to generate {missing} missing annotations")
    if all(v >= 250 for v in aug_counts.values()):
        print("  → Dataset looks ready for Phase 2 (scene classifier training)! 🎉")

    if args.visualise:
        print("\n🖼️  Generating visualisation...")
        visualise_samples(n=6)

    print("\n" + "=" * 55)


if __name__ == "__main__":
    main()