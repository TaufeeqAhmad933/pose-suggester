"""
Phase 3 — Build Pose Library
-------------------------------
Reads all keypoint annotations from phase1_data/augmented_annotations/
and builds a KNN index per scene subcategory.

For each subcategory, we:
  1. Flatten each pose's keypoints into a feature vector
  2. Normalise vectors (hip-centred + scale-normalised)
  3. Fit a NearestNeighbors model (cosine distance)
  4. Save the index + raw keypoint arrays to pose_library/

Output per subcategory (e.g. cafe_indoor__full_body):
  pose_library/
    cafe_indoor__full_body.joblib   ← NearestNeighbors model
    cafe_indoor__full_body_raw.npy  ← (N, 33, 3) raw keypoint arrays
    cafe_indoor__full_body_meta.json ← source image paths

Usage:
    python build_pose_library.py
"""

import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm

# ── Config ────────────────────────────────────────────────────────────────────
ANN_DIR     = Path("../phase1_data/augmented_annotations")
LIBRARY_DIR = Path("pose_library")
LIBRARY_DIR.mkdir(exist_ok=True)

CATEGORIES    = ["cafe_indoor", "nature", "urban_street"]
SUBCATEGORIES = ["full_body", "waist_up"]

# The 17 'body' landmarks we actually care about for pose comparison
# (drop face details like eye inner/outer, ear sub-points)
BODY_LANDMARKS = [
    "nose",
    "left_shoulder",  "right_shoulder",
    "left_elbow",     "right_elbow",
    "left_wrist",     "right_wrist",
    "left_hip",       "right_hip",
    "left_knee",      "right_knee",
    "left_ankle",     "right_ankle",
    "left_ear",       "right_ear",
    "left_eye",       "right_eye",
]
N_LANDMARKS = len(BODY_LANDMARKS)   # 17


# ── Keypoint → feature vector ─────────────────────────────────────────────────

def keypoints_to_vector(keypoints: dict) -> np.ndarray | None:
    """
    Convert a keypoint dict → normalised flat vector of shape (N_LANDMARKS * 2,).

    Normalisation steps:
      1. Extract (x, y) for each body landmark (skip if visibility < 0.3)
      2. Translate so hip midpoint = origin
      3. Scale so torso height (shoulder mid → hip mid) = 1.0
    Returns None if essential landmarks are missing.
    """
    coords = []
    for name in BODY_LANDMARKS:
        kp = keypoints.get(name, {})
        vis = kp.get("visibility", 0.0)
        if vis < 0.3 or not kp:
            coords.append([np.nan, np.nan])
        else:
            coords.append([kp["x"], kp["y"]])

    coords = np.array(coords, dtype=np.float32)   # (17, 2)

    # Need at least hips and shoulders visible
    hip_indices = [BODY_LANDMARKS.index("left_hip"),
                   BODY_LANDMARKS.index("right_hip")]
    sho_indices = [BODY_LANDMARKS.index("left_shoulder"),
                   BODY_LANDMARKS.index("right_shoulder")]

    hip_coords = coords[hip_indices]
    sho_coords = coords[sho_indices]

    if np.any(np.isnan(hip_coords)) or np.any(np.isnan(sho_coords)):
        return None   # can't normalise without hips/shoulders

    hip_mid = np.nanmean(hip_coords, axis=0)
    sho_mid = np.nanmean(sho_coords, axis=0)

    # Translate
    coords -= hip_mid

    # Scale (torso height)
    torso_height = np.linalg.norm(sho_mid - hip_mid)
    if torso_height < 1e-6:
        return None
    coords /= torso_height

    # Replace NaN with 0 (missing joints set to origin)
    coords = np.nan_to_num(coords, nan=0.0)

    return coords.flatten()   # (34,)


def keypoints_to_raw(keypoints: dict) -> np.ndarray:
    """
    Full (17, 3) array of [x, y, visibility] for storage and frontend rendering.
    """
    raw = []
    for name in BODY_LANDMARKS:
        kp = keypoints.get(name, {})
        raw.append([kp.get("x", 0.0), kp.get("y", 0.0), kp.get("visibility", 0.0)])
    return np.array(raw, dtype=np.float32)


# ── Main ──────────────────────────────────────────────────────────────────────

def build_library():
    print("=" * 55)
    print("  Phase 3 — Building Pose Library")
    print("=" * 55)

    for category in CATEGORIES:
        for subcategory in SUBCATEGORIES:
            ann_folder = ANN_DIR / category / subcategory
            label = f"{category}__{subcategory}"

            if not ann_folder.exists():
                print(f"\n[WARN] Missing annotations: {ann_folder}")
                continue

            ann_files = sorted(ann_folder.glob("*.json"))
            print(f"\n📂 {label}  ({len(ann_files)} annotations)")

            vectors, raw_kps, meta = [], [], []

            for ann_path in tqdm(ann_files, desc="  Processing"):
                with open(ann_path) as f:
                    ann = json.load(f)

                if not ann.get("person_detected", False):
                    continue
                kp = ann.get("keypoints", {})
                if not kp:
                    continue

                vec = keypoints_to_vector(kp)
                if vec is None:
                    continue

                raw = keypoints_to_raw(kp)
                vectors.append(vec)
                raw_kps.append(raw)
                meta.append({
                    "source_image": ann.get("image", ""),
                    "category": category,
                    "subcategory": subcategory,
                    "augmented": ann.get("augmented", False),
                })

            if len(vectors) < 2:
                print(f"  [WARN] Not enough valid poses ({len(vectors)}), skipping.")
                continue

            X = np.array(vectors)       # (N, 34)
            R = np.array(raw_kps)       # (N, 17, 3)

            # Fit KNN (cosine distance works well for normalised pose vectors)
            k = min(10, len(X))
            knn = NearestNeighbors(n_neighbors=k, metric="cosine", algorithm="brute")
            knn.fit(X)

            # Save
            joblib.dump(knn, LIBRARY_DIR / f"{label}.joblib")
            np.save(LIBRARY_DIR / f"{label}_raw.npy", R)
            with open(LIBRARY_DIR / f"{label}_meta.json", "w") as f:
                json.dump(meta, f, indent=2)

            print(f"  ✅ Saved {len(vectors)} poses → pose_library/{label}.*")

    print(f"\n✅ Pose library built → {LIBRARY_DIR}/")


if __name__ == "__main__":
    build_library()