from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import cv2
import face_recognition
import numpy as np

logger = logging.getLogger(__name__)

REQUIRED_POSES = 5
FRAMES_PER_POSE = 8
MIN_QUALITY_FRAMES = 5

MIN_FACE_AREA_RATIO = 0.15
LAPLACIAN_VARIANCE_MIN = 80
BRIGHTNESS_MIN = 40
BRIGHTNESS_MAX = 220

FACE_DATA_DIR = Path("face_data")
EMBEDDINGS_DIR = FACE_DATA_DIR / "embeddings"
PROFILES_PATH = FACE_DATA_DIR / "profiles.json"


class EnrollmentManager:
    def __init__(self, face_data_dir: Path) -> None:
        self._face_data_dir = face_data_dir
        self._embeddings_dir = face_data_dir / "embeddings"
        self._embeddings_dir.mkdir(parents=True, exist_ok=True)
        self._active_name: str | None = None
        self._current_pose = 0
        self._pose_frames: list[np.ndarray] = []
        self._pose_embeddings: list[np.ndarray] = []
        self._all_embeddings: list[list[np.ndarray]] = []

    def start_enrollment(self, name: str) -> dict[str, Any]:
        if self._active_name is not None:
            return {"status": "error", "message": "Enrollment already in progress"}
        self._active_name = name
        self._current_pose = 0
        self._pose_frames = []
        self._pose_embeddings = []
        self._all_embeddings = []
        logger.info("Started enrollment for '%s'", name)
        return {
            "status": "ok",
            "message": f"Enrollment started for {name}",
            "pose": self._current_pose + 1,
            "total_poses": REQUIRED_POSES,
        }

    def capture_pose(self, frame: np.ndarray) -> dict[str, Any]:
        if self._active_name is None:
            return {"status": "error", "message": "No active enrollment"}

        face_locs = face_recognition.face_locations(
            cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), model="hog"
        )
        if not face_locs:
            return {"status": "retry", "message": "No face detected"}

        face_loc = face_locs[0]
        quality_issue = self.check_quality(frame, face_loc)
        if quality_issue is not None:
            return {"status": "retry", "message": f"Quality issue: {quality_issue}"}

        encodings = face_recognition.face_encodings(
            cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), [face_loc]
        )
        if not encodings:
            return {"status": "retry", "message": "Could not compute embedding"}

        self._pose_frames.append(frame)
        self._pose_embeddings.append(encodings[0])

        collected = len(self._pose_frames)

        if collected >= FRAMES_PER_POSE:
            return self._finalize_pose()

        return {
            "status": "capturing",
            "message": f"Captured {collected}/{FRAMES_PER_POSE}",
            "pose": self._current_pose + 1,
            "total_poses": REQUIRED_POSES,
            "captured": collected,
        }

    def _finalize_pose(self) -> dict[str, Any]:
        if len(self._pose_embeddings) < MIN_QUALITY_FRAMES:
            self._pose_frames.clear()
            self._pose_embeddings.clear()
            return {
                "status": "retry",
                "message": f"Only {len(self._pose_embeddings)}/{MIN_QUALITY_FRAMES} quality frames, retaking pose",
                "pose": self._current_pose + 1,
                "total_poses": REQUIRED_POSES,
            }

        mean_embedding = np.mean(self._pose_embeddings, axis=0)
        mean_embedding = mean_embedding / np.linalg.norm(mean_embedding)

        self._all_embeddings.append(self._pose_embeddings.copy())
        save_path = (
            self._embeddings_dir
            / f"{self._active_name}_{self._current_pose:03d}.npy"
        )
        np.save(str(save_path), mean_embedding)
        logger.info("Saved pose %d embedding to %s", self._current_pose + 1, save_path)

        self._pose_frames.clear()
        self._pose_embeddings.clear()
        self._current_pose += 1

        if self._current_pose >= REQUIRED_POSES:
            return self._finish_enrollment()

        return {
            "status": "next_pose",
            "message": f"Pose {self._current_pose}/{REQUIRED_POSES} captured",
            "pose": self._current_pose + 1,
            "total_poses": REQUIRED_POSES,
        }

    def _finish_enrollment(self) -> dict[str, Any]:
        profiles: list[dict[str, Any]] = []
        if PROFILES_PATH.exists():
            try:
                profiles = json.loads(PROFILES_PATH.read_text())
            except (json.JSONDecodeError, OSError) as e:
                logger.warning("Failed to read profiles.json: %s", e)

        entry = {
            "name": self._active_name,
            "embeddings": [
                f"{self._active_name}_{i:03d}.npy" for i in range(REQUIRED_POSES)
            ],
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }

        existing = [p for p in profiles if p.get("name") != self._active_name]
        existing.append(entry)

        try:
            PROFILES_PATH.write_text(json.dumps(existing, indent=2))
            logger.info("Enrollment complete for '%s'", self._active_name)
        except OSError as e:
            logger.error("Failed to write profiles.json: %s", e)
            return {"status": "error", "message": f"Failed to save profile: {e}"}

        name = self._active_name
        self._active_name = None

        return {
            "status": "complete",
            "message": f"Enrollment complete for {name}",
            "name": name,
        }

    def cancel_enrollment(self) -> dict[str, Any]:
        if self._active_name is None:
            return {"status": "error", "message": "No active enrollment"}
        name = self._active_name
        for i in range(self._current_pose):
            p = self._embeddings_dir / f"{name}_{i:03d}.npy"
            try:
                p.unlink(missing_ok=True)
            except OSError as e:
                logger.warning("Failed to delete %s: %s", p, e)
        self._active_name = None
        self._pose_frames.clear()
        self._pose_embeddings.clear()
        self._all_embeddings.clear()
        self._current_pose = 0
        logger.info("Enrollment cancelled for '%s'", name)
        return {"status": "ok", "message": f"Enrollment cancelled for {name}"}

    def get_status(self) -> dict[str, Any]:
        if self._active_name is None:
            return {"status": "idle"}
        return {
            "status": "active",
            "name": self._active_name,
            "pose": self._current_pose + 1,
            "total_poses": REQUIRED_POSES,
            "captured": len(self._pose_frames),
        }

    @staticmethod
    def check_quality(
        frame: np.ndarray, face_loc: tuple[int, int, int, int]
    ) -> str | None:
        top, right, bottom, left = face_loc
        fw = right - left
        fh = bottom - top
        face_area = fw * fh
        frame_area = frame.shape[0] * frame.shape[1]
        if face_area / frame_area < MIN_FACE_AREA_RATIO:
            return "too_far"

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        if lap_var < LAPLACIAN_VARIANCE_MIN:
            return "blurry"

        mean_brightness = np.mean(gray)
        if mean_brightness < BRIGHTNESS_MIN:
            return "too_dark"
        if mean_brightness > BRIGHTNESS_MAX:
            return "too_bright"

        return None
