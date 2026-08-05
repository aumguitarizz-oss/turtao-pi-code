from __future__ import annotations

import contextlib
import logging
import time
from pathlib import Path

import cv2
import numpy as np

from turtao.state import AppState

logger = logging.getLogger(__name__)

try:
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions
    from mediapipe.tasks.python import vision as mp_vision
except ImportError:
    mp = None
    mp_vision = None
    BaseOptions = None
    logger.warning("MediaPipe not available. Pose tracking disabled.")

DEFAULT_MODEL_PATH = "models/pose_landmarker_lite.task"
MAX_POSES = 4


class PoseTracker:
    def __init__(self, state: AppState, model_path: str = DEFAULT_MODEL_PATH) -> None:
        self.state = state
        self._landmarker = None
        if mp is None:
            return
        if not Path(model_path).is_file():
            logger.error("Pose landmarker model not found at %s", model_path)
            return
        try:
            options = mp_vision.PoseLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=model_path),
                running_mode=mp_vision.RunningMode.VIDEO,
                num_poses=MAX_POSES,
                min_pose_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            self._landmarker = mp_vision.PoseLandmarker.create_from_options(options)
            logger.info(
                "MediaPipe multi-person PoseLandmarker initialized (num_poses=%d).",
                MAX_POSES,
            )
        except Exception:
            logger.exception("Failed to initialize MediaPipe PoseLandmarker")

    def process_frame(self, frame: np.ndarray) -> None:
        if self._landmarker is None:
            return

        try:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            timestamp_ms = int(time.time() * 1000)
            result = self._landmarker.detect_for_video(mp_image, timestamp_ms)

            h, w = frame.shape[:2]
            all_landmarks: list[list[tuple[int, int]]] = []
            for pose_landmarks in result.pose_landmarks:
                points = [(int(lm.x * w), int(lm.y * h)) for lm in pose_landmarks]
                all_landmarks.append(points)

            with self.state:
                self.state.pose_landmarks = all_landmarks
        except Exception:
            logger.exception("Error in MediaPipe Pose processing")

    def close(self) -> None:
        if self._landmarker is not None:
            with contextlib.suppress(Exception):
                self._landmarker.close()
