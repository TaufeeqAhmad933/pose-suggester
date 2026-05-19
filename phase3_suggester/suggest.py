"""
Phase 3 — Pose Suggestion Engine
-----------------------------------
Given:
  - scene_label   : e.g. "cafe_indoor__full_body"
  - person_kp     : keypoint dict from MediaPipe (current person)

Returns the top-N suggested poses as lists of keypoint dicts,
ready to be drawn as stick-figure overlays on the frontend.

Can also be used standalone:
    python suggest.py --scene cafe_indoor__full_body
    (returns 3 random good poses from that category when no person kp given)
"""

import argparse
import json
import random
from pathlib import Path

import joblib
import numpy as np

LIBRARY_DIR = Path("pose_library")

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


# ── Vector helpers (must match build_pose_library.py) ────────────────────────

def keypoints_to_vector(keypoints: dict) -> np.ndarray | None:
    coords = []
    for name in BODY_LANDMARKS:
        kp = keypoints.get(name, {})
        if kp.get("visibility", 0.0) < 0.3 or not kp:
            coords.append([np.nan, np.nan])
        else:
            coords.append([kp["x"], kp["y"]])
    coords = np.array(coords, dtype=np.float32)

    hip_idx = [BODY_LANDMARKS.index("left_hip"), BODY_LANDMARKS.index("right_hip")]
    sho_idx = [BODY_LANDMARKS.index("left_shoulder"), BODY_LANDMARKS.index("right_shoulder")]

    hip_coords = coords[hip_idx]
    sho_coords = coords[sho_idx]
    if np.any(np.isnan(hip_coords)) or np.any(np.isnan(sho_coords)):
        return None

    hip_mid = np.nanmean(hip_coords, axis=0)
    sho_mid = np.nanmean(sho_coords, axis=0)
    coords -= hip_mid
    torso = np.linalg.norm(sho_mid - hip_mid)
    if torso < 1e-6:
        return None
    coords /= torso
    return np.nan_to_num(coords, nan=0.0).flatten()


def raw_to_keypoint_list(raw: np.ndarray) -> list[dict]:
    """
    Converts (17, 3) numpy array → list of dicts, one per landmark.
    Shape the frontend expects:
      [{"name": "nose", "x": 0.51, "y": 0.12, "visibility": 0.99}, ...]
    """
    result = []
    for i, name in enumerate(BODY_LANDMARKS):
        result.append({
            "name": name,
            "x": float(raw[i, 0]),
            "y": float(raw[i, 1]),
            "visibility": float(raw[i, 2]),
        })
    return result


# ── Core suggestion logic ─────────────────────────────────────────────────────

class PoseSuggester:
    def __init__(self):
        self._cache: dict = {}   # label → (knn, raw, meta)

    def _load(self, label: str):
        if label in self._cache:
            return self._cache[label]

        knn_path  = LIBRARY_DIR / f"{label}.joblib"
        raw_path  = LIBRARY_DIR / f"{label}_raw.npy"
        meta_path = LIBRARY_DIR / f"{label}_meta.json"

        if not knn_path.exists():
            raise FileNotFoundError(
                f"No pose library for '{label}'. "
                f"Run build_pose_library.py first."
            )

        knn  = joblib.load(knn_path)
        raw  = np.load(raw_path)
        with open(meta_path) as f:
            meta = json.load(f)

        self._cache[label] = (knn, raw, meta)
        return knn, raw, meta

    def suggest(
        self,
        scene_label: str,
        person_keypoints: dict | None = None,
        n_suggestions: int = 3,
        diversity_factor: float = 0.5,
    ) -> list[dict]:
        """
        Returns up to n_suggestions pose dicts.

        Each pose dict:
        {
          "rank": 1,
          "score": 0.87,           ← similarity to a 'good pose' in this scene
          "scene": "cafe_indoor__full_body",
          "source_image": "cafe_indoor/full_body/img_004.jpg",
          "keypoints": [ {"name": ..., "x": ..., "y": ..., "visibility": ...} ]
        }

        Strategy:
          - If person_keypoints given → find nearest good poses in the scene
            (so suggestions are achievable from current position)
          - If no person_keypoints → return a diverse sample of top poses
        """
        knn, raw, meta = self._load(scene_label)
        n_total = len(meta)

        if person_keypoints:
            vec = keypoints_to_vector(person_keypoints)
        else:
            vec = None

        if vec is not None:
            # KNN query: find poses in the library closest to the person's current pose
            k = min(20, n_total)
            distances, indices = knn.kneighbors([vec], n_neighbors=k)
            distances = distances[0]
            indices   = indices[0]

            # Cosine distance → similarity (1 - distance, but we want DIFFERENT
            # from current, yet achievable → moderate distance range)
            # We prefer poses that are "reachable but distinct" (not identical,
            # not wildly different)
            scored = list(zip(distances, indices))

            # Diversity: spread selections across the sorted list
            # to avoid returning 3 nearly-identical poses
            step = max(1, len(scored) // n_suggestions)
            selected = [scored[i * step] for i in range(n_suggestions)
                        if i * step < len(scored)]
        else:
            # No person detected — pick diverse random poses from library
            idxs = random.sample(range(n_total), min(n_suggestions * 4, n_total))
            scored = [(0.0, i) for i in idxs]

            # Pick with maximum spread (first, middle, last third)
            step = max(1, len(scored) // n_suggestions)
            selected = [scored[i * step] for i in range(n_suggestions)
                        if i * step < len(scored)]

        suggestions = []
        for rank, (dist, idx) in enumerate(selected[:n_suggestions], start=1):
            similarity = round(float(1.0 - dist), 4)
            suggestions.append({
                "rank": rank,
                "score": similarity,
                "scene": scene_label,
                "source_image": meta[idx].get("source_image", ""),
                "keypoints": raw_to_keypoint_list(raw[idx]),
            })

        return suggestions


# ── CLI for quick testing ─────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scene",
        default="cafe_indoor__full_body",
        help="Scene label, e.g. cafe_indoor__full_body"
    )
    parser.add_argument("--n", type=int, default=3)
    args = parser.parse_args()

    suggester = PoseSuggester()
    results = suggester.suggest(args.scene, n_suggestions=args.n)

    print(f"\n🎯 Top {len(results)} pose suggestions for '{args.scene}':\n")
    for r in results:
        print(f"  Rank {r['rank']}  |  Score: {r['score']:.3f}")
        print(f"  Source: {r['source_image']}")
        print(f"  Sample keypoints (first 3):")
        for kp in r["keypoints"][:3]:
            print(f"    {kp['name']}: ({kp['x']:.3f}, {kp['y']:.3f})")
        print()