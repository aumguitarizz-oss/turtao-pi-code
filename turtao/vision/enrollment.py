from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path
from typing import Any

import cv2
import face_recognition
import numpy as np

from turtao.vision.dlib_lock import DLIB_LOCK

logger = logging.getLogger(__name__)

REQUIRED_POSES = 5
FRAMES_PER_POSE = 5          # Option C: capture 5 frames per pose burst
MIN_QUALITY_FRAMES = 3       # need at least 3 good frames out of 5
BURST_DELAY_MS = 80          # ~80ms between burst frames (≈12fps burst)

MIN_FACE_AREA_RATIO = 0.05
LAPLACIAN_VARIANCE_MIN = 30
BRIGHTNESS_MIN = 40
BRIGHTNESS_MAX = 220

FACE_DATA_DIR = Path("face_data")
EMBEDDINGS_DIR = FACE_DATA_DIR / "embeddings"
PROFILES_PATH = FACE_DATA_DIR / "profiles.json"

# Option B: Pose guide messages shown to the user
POSE_GUIDES: list[dict[str, str]] = [
    {
        "action": "Look straight at the camera",
        "tip":    "Keep your face centred and at eye level",
    },
    {
        "action": "Turn your head slightly to the LEFT",
        "tip":    "About 20–30° — don't go too far",
    },
    {
        "action": "Turn your head slightly to the RIGHT",
        "tip":    "About 20–30° — don't go too far",
    },
    {
        "action": "Tilt your head slightly UP",
        "tip":    "Chin up a little, keep eyes on camera",
    },
    {
        "action": "Tilt your head slightly DOWN",
        "tip":    "Chin down a little, keep eyes on camera",
    },
]

# Option B: Quality-issue human-readable guidance
QUALITY_GUIDANCE: dict[str, str] = {
    "too_far":    "Come in closer — your face is too small in the frame",
    "blurry":     "Hold still — the image is blurry",
    "too_dark":   "Move to a brighter area or add more light",
    "too_bright": "Avoid strong light behind or directly on you",
    "no_face":    "No face detected — make sure your face is visible",
    "multi_face": "Please have only one person in frame at a time",
    "embed_fail": "Could not compute embedding — please retry",
    "no_frame":   "No camera frame available — please retry",
}


class EnrollmentManager:
    def __init__(self, face_data_dir: Path, settings: Any = None) -> None:
        # Read live off the shared Settings object (not snapshotted) so
        # flipping the toggle takes effect on the very next capture, no
        # restart needed — the user asked for this specifically to use
        # right before a demo.
        self._settings = settings
        self._face_data_dir = face_data_dir
        self._embeddings_dir = face_data_dir / "embeddings"
        self._embeddings_dir.mkdir(parents=True, exist_ok=True)
        self._active_name: str | None = None
        self._current_pose = 0
        self._pose_frames: list[np.ndarray] = []
        self._pose_embeddings: list[np.ndarray] = []
        self._all_embeddings: list[list[np.ndarray]] = []
        # Option B: last guidance message
        self._last_guidance: str = ""
        self._last_quality_issue: str = ""
        # Option D: track how many outliers were discarded across all poses
        self._total_outliers_discarded: int = 0
        self._last_pose_outliers: int = 0
        self._processing = False
        self._processing_lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start_enrollment(self, name: str) -> dict[str, Any]:
        if self._active_name is not None:
            return {"status": "error", "message": "Enrollment already in progress"}
        self._active_name = name
        self._current_pose = 0
        self._pose_frames = []
        self._pose_embeddings = []
        self._all_embeddings = []
        self._last_guidance = ""
        self._last_quality_issue = ""
        logger.info("Started enrollment for '%s'", name)
        guide = self._current_pose_guide()
        return {
            "status": "ok",
            "message": f"Enrollment started for {name}",
            "pose": self._current_pose + 1,
            "total_poses": REQUIRED_POSES,
            "guidance": guide,
        }

    def capture_pose(self, get_frame) -> dict[str, Any]:
        """
        Burst-capture FRAMES_PER_POSE frames, calling `get_frame()` fresh
        for each one so the burst captures genuinely distinct frames
        rather than five copies of a single snapshot.
        """
        if self._active_name is None:
            return {"status": "error", "message": "No active enrollment"}

        burst_results = self._collect_burst(get_frame)
        good_frames = [r for r in burst_results if r["ok"]]
        bad_frames  = [r for r in burst_results if not r["ok"]]

        logger.info(
            "Burst for pose %d: %d/%d good frames",
            self._current_pose + 1,
            len(good_frames),
            FRAMES_PER_POSE,
        )

        if len(good_frames) < MIN_QUALITY_FRAMES:
            issues = [r.get("issue", "unknown") for r in bad_frames]
            top_issue = max(set(issues), key=issues.count) if issues else "unknown"
            self._last_quality_issue = top_issue
            guidance_msg = QUALITY_GUIDANCE.get(top_issue, top_issue)
            self._last_guidance = guidance_msg
            return {
                "status": "retry",
                "message": f"Only {len(good_frames)}/{MIN_QUALITY_FRAMES} quality frames: {guidance_msg}",
                "guidance": guidance_msg,
                "pose": self._current_pose + 1,
                "total_poses": REQUIRED_POSES,
            }

        for r in good_frames:
            self._pose_frames.append(r["frame"])
            self._pose_embeddings.append(r["embedding"])

        return self._finalize_pose()

    def capture_pose_burst(self, frames: list[np.ndarray]) -> dict[str, Any]:
        """
        Process pre-supplied frames (e.g. uploaded from a phone) through the
        same quality gate and pose-advancement logic as capture_pose, without
        the live-camera burst delay — the frames already represent distinct
        instants captured client-side.
        """
        if self._active_name is None:
            return {"status": "error", "message": "No active enrollment"}

        results = [self._process_single_frame(f) for f in frames]
        good_frames = [r for r in results if r["ok"]]
        bad_frames = [r for r in results if not r["ok"]]

        logger.info(
            "Upload burst for pose %d: %d/%d good frames",
            self._current_pose + 1,
            len(good_frames),
            len(frames),
        )

        if len(good_frames) < MIN_QUALITY_FRAMES:
            issues = [r.get("issue", "unknown") for r in bad_frames]
            top_issue = max(set(issues), key=issues.count) if issues else "unknown"
            self._last_quality_issue = top_issue
            guidance_msg = QUALITY_GUIDANCE.get(top_issue, top_issue)
            self._last_guidance = guidance_msg
            return {
                "status": "retry",
                "message": f"Only {len(good_frames)}/{MIN_QUALITY_FRAMES} quality frames: {guidance_msg}",
                "guidance": guidance_msg,
                "pose": self._current_pose + 1,
                "total_poses": REQUIRED_POSES,
            }

        for r in good_frames:
            self._pose_frames.append(r["frame"])
            self._pose_embeddings.append(r["embedding"])

        return self._finalize_pose()

    @property
    def is_processing(self) -> bool:
        return self._processing

    def try_begin_processing(self) -> bool:
        with self._processing_lock:
            if self._processing:
                return False
            self._processing = True
            return True

    def release_processing(self) -> None:
        self._processing = False

    def start_capture_burst(
        self, frames: list[np.ndarray], on_complete: Any = None
    ) -> None:
        def _run() -> None:
            try:
                result = self.capture_pose_burst(frames)
                if result.get("status") == "complete" and on_complete is not None:
                    on_complete()
            except Exception:
                logger.exception("capture_pose_burst worker failed")
                self._last_quality_issue = "processing_error"
            finally:
                self.release_processing()

        threading.Thread(target=_run, daemon=True).start()

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
        self._last_guidance = ""
        self._last_quality_issue = ""
        logger.info("Enrollment cancelled for '%s'", name)
        return {"status": "ok", "message": f"Enrollment cancelled for {name}"}

    def get_status(self) -> dict[str, Any]:
        if self._active_name is None:
            return {"status": "idle"}
        guide = self._current_pose_guide()
        return {
            "status": "active",
            "name": self._active_name,
            "pose": self._current_pose + 1,
            "total_poses": REQUIRED_POSES,
            "captured": len(self._pose_frames),
            "guidance": guide,
            "quality_issue": self._last_quality_issue,
            # Option D: expose outlier stats for the UI
            "outliers_discarded": self._total_outliers_discarded,
            "last_pose_outliers": self._last_pose_outliers,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _collect_burst(self, get_frame) -> list[dict[str, Any]]:
        """Collect FRAMES_PER_POSE genuinely distinct frames via get_frame(),
        with a short delay between each so they represent different instants."""
        frames_to_process: list[np.ndarray] = []
        for i in range(FRAMES_PER_POSE):
            if i > 0:
                time.sleep(BURST_DELAY_MS / 1000.0)
            frames_to_process.append(get_frame())

        return [self._process_single_frame(f) for f in frames_to_process]

    def _process_single_frame(self, raw_frame: np.ndarray | None) -> dict[str, Any]:
        """Quality-gate + embed one frame directly (no preprocessing)."""
        if raw_frame is None:
            return {"ok": False, "issue": "no_frame", "frame": None, "embedding": None}

        rgb = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2RGB)

        # dlib's detector is not safe to call concurrently with the
        # recognition engine's own dlib calls on another thread — see
        # dlib_lock.py.
        with DLIB_LOCK:
            face_locs = face_recognition.face_locations(rgb, model="hog")

        # Multi-face check
        if not face_locs:
            return {"ok": False, "issue": "no_face", "frame": raw_frame, "embedding": None}
        if len(face_locs) > 1:
            return {"ok": False, "issue": "multi_face", "frame": raw_frame, "embedding": None}

        face_loc = face_locs[0]
        # Settings' strict_face_scan toggle: when off, a merely imperfect
        # scan (too_far/blurry/too_dark/too_bright) is no longer rejected —
        # a face still has to be detected and encodable regardless.
        if self._settings is None or self._settings.strict_face_scan:
            quality_issue = self.check_quality(raw_frame, face_loc)
            if quality_issue is not None:
                return {"ok": False, "issue": quality_issue, "frame": raw_frame, "embedding": None}

        with DLIB_LOCK:
            encodings = face_recognition.face_encodings(rgb, [face_loc])
        if not encodings:
            return {"ok": False, "issue": "embed_fail", "frame": raw_frame, "embedding": None}

        return {"ok": True, "issue": None, "frame": raw_frame, "embedding": encodings[0]}

    def _finalize_pose(self) -> dict[str, Any]:
        if len(self._pose_embeddings) < MIN_QUALITY_FRAMES:
            self._pose_frames.clear()
            self._pose_embeddings.clear()
            guidance_msg = QUALITY_GUIDANCE.get(
                self._last_quality_issue, "Please retry this pose"
            )
            return {
                "status": "retry",
                "message": f"Only {len(self._pose_embeddings)}/{MIN_QUALITY_FRAMES} quality frames, retaking pose",
                "guidance": guidance_msg,
                "pose": self._current_pose + 1,
                "total_poses": REQUIRED_POSES,
            }

        # Average the good-frame embeddings for this pose — raw, unnormalised,
        # matching the raw dlib encodings the recognition engine compares
        # against at match time (shellalert's proven train.py does the same:
        # "average embeddings for robustness across angles/lighting").
        centroid = np.mean(self._pose_embeddings, axis=0)

        self._all_embeddings.append(self._pose_embeddings.copy())
        save_path = (
            self._embeddings_dir
            / f"{self._active_name}_{self._current_pose:03d}.npy"
        )
        np.save(str(save_path), centroid)
        logger.info(
            "Saved pose %d embedding (mean of %d frames) to %s",
            self._current_pose + 1,
            len(self._pose_embeddings),
            save_path,
        )

        self._pose_frames.clear()
        self._pose_embeddings.clear()
        self._current_pose += 1

        if self._current_pose >= REQUIRED_POSES:
            return self._finish_enrollment()

        self._last_quality_issue = ""
        guide = self._current_pose_guide()
        return {
            "status": "next_pose",
            "message": f"Pose {self._current_pose}/{REQUIRED_POSES} captured",
            "pose": self._current_pose + 1,
            "total_poses": REQUIRED_POSES,
            "guidance": guide,
            "outliers_discarded": 0,
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
        total_outliers = self._total_outliers_discarded
        self._active_name = None
        self._last_guidance = ""
        self._last_quality_issue = ""
        self._total_outliers_discarded = 0
        self._last_pose_outliers = 0

        logger.info(
            "Enrollment complete for '%s' — total outlier frames discarded by DBSCAN: %d",
            name,
            total_outliers,
        )
        return {
            "status": "complete",
            "message": f"Enrollment complete for {name}",
            "name": name,
            "total_outliers_discarded": total_outliers,
        }

    def _current_pose_guide(self) -> dict[str, str]:
        """Option B: Return the guidance dict for the current pose."""
        idx = min(self._current_pose, len(POSE_GUIDES) - 1)
        return POSE_GUIDES[idx]

    # ------------------------------------------------------------------
    # Static quality gate
    # ------------------------------------------------------------------

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
