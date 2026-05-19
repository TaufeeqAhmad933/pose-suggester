"""
services/pose_estimator.py
----------------------------
Wraps MediaPipe PoseLandmarker (Tasks API, v0.10.13+).

FIRST-TIME SETUP — download model once into phase4_backend/:
    curl -L "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task" \
         -o pose_landmarker.task
"""

import os
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision as mp_vision
import numpy as np

# Looks for model in backend root, or override via env var
MODEL_PATH = os.environ.get(
    "POSE_MODEL_PATH",
    str(Path(__file__).parent.parent / "pose_landmarker.task")
)

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


class PoseEstimator:
    def __init__(self, model_path: str = MODEL_PATH):
        if not Path(model_path).exists():
            raise FileNotFoundError(
                f"Pose model not found: {model_path}\n"
                "Download with:\n"
                '  curl -L "https://storage.googleapis.com/mediapipe-models/'
                'pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task"'
                " -o pose_landmarker.task"
            )
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
        self._landmarker = mp_vision.PoseLandmarker.create_from_options(options)

    def estimate(self, image_bytes: bytes) -> dict | None:
        nparr = np.frombuffer(image_bytes, np.uint8)
        img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img_bgr is None:
            return None

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
        result = self._landmarker.detect(mp_image)

        if not result.pose_landmarks or len(result.pose_landmarks) == 0:
            return None

        landmarks = result.pose_landmarks[0]
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

    def close(self):
        self._landmarker.close()


_estimator_instance: PoseEstimator | None = None


def get_pose_estimator() -> PoseEstimator:
    global _estimator_instance
    if _estimator_instance is None:
        _estimator_instance = PoseEstimator()
    return _estimator_instance