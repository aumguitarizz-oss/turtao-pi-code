from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import cv2
import face_recognition
import numpy as np

from turtao.state import AppState, ThreatLabel

logger = logging.getLogger(__name__)

UNKNOWN_SAVE_INTERVAL = 2.0
TTS_DEBOUNCE_INTERVAL = 5.0
RESIZE_SCALE = 0.5
KP_PAN = 0.18
KP_TILT = 0.22
DEADZONE = 30
PAN_MIN = 10
PAN_MAX = 170
TILT_MIN = 10
TILT_MAX = 170


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))


def _calculate_aim(
    l: int, t: int, r: int, b: int, frame_w: int = 640, frame_h: int = 480
) -> tuple[int, int]:
    cx = (l + r) / 2
    cy = (t + b) / 2
    dx = cx - frame_w / 2
    dy = cy - frame_h / 2
    pan = 90
    tilt = 90
    if abs(dx) > DEADZONE:
        pan = int(90 + dx * KP_PAN)
    if abs(dy) > DEADZONE:
        tilt = int(90 + dy * KP_TILT)
    pan = max(PAN_MIN, min(PAN_MAX, pan))
    tilt = max(TILT_MIN, min(TILT_MAX, tilt))
    return pan, tilt


class FaceRecognitionEngine:
    def __init__(self, state: AppState, tolerance: float = 0.52) -> None:
        self._state = state
        self._tolerance = tolerance
        self._known_embeddings: list[np.ndarray] = []
        self._known_names: list[str] = []
        self._last_unknown_save = 0.0
        self._last_tts_threat = 0.0

    def load_embeddings(self, profile_dir: str) -> None:
        path = Path(profile_dir)
        if not path.is_dir():
            logger.warning("Embedding directory not found: %s", profile_dir)
            return
        self._known_embeddings.clear()
        self._known_names.clear()
        for npy_file in sorted(path.iterdir()):
            if npy_file.suffix != ".npy":
                continue
            name = npy_file.stem
            emb = np.load(str(npy_file))
            self._known_embeddings.append(emb)
            self._known_names.append(name)
        logger.info(
            "Loaded %d embeddings from %s", len(self._known_embeddings), profile_dir
        )

    def process_frame(self, frame: np.ndarray) -> None:
        small = cv2.resize(frame, (0, 0), fx=RESIZE_SCALE, fy=RESIZE_SCALE)
        rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_small, model="hog")
        face_encodings = face_recognition.face_encodings(rgb_small, face_locations)

        with self._state:
            if not face_locations:
                self._state.threat_label = ThreatLabel.IDLE
                return

            best_match = ThreatLabel.THREAT
            best_confidence = 0.0
            best_box = None

            for i, encoding in enumerate(face_encodings):
                top, right, bottom, left = face_locations[i]
                # Scale back to original resolution
                left = int(left / RESIZE_SCALE)
                top = int(top / RESIZE_SCALE)
                right = int(right / RESIZE_SCALE)
                bottom = int(bottom / RESIZE_SCALE)

                if not self._known_embeddings:
                    best_match = ThreatLabel.THREAT
                    best_box = (left, top, right, bottom)
                    break

                sims = [
                    cosine_similarity(encoding, known)
                    for known in self._known_embeddings
                ]
                max_sim = max(sims) if sims else 0.0

                if max_sim >= self._tolerance:
                    best_match = ThreatLabel.SAFE
                    best_confidence = max_sim
                    best_box = (left, top, right, bottom)
                    break
                else:
                    if max_sim > best_confidence:
                        best_confidence = max_sim
                        best_box = (left, top, right, bottom)

            if best_match == ThreatLabel.THREAT:
                now = time.time()
                if now - self._last_unknown_save >= UNKNOWN_SAVE_INTERVAL:
                    self._save_unknown(frame, best_box)
                    self._last_unknown_save = now
                if now - self._last_tts_threat >= TTS_DEBOUNCE_INTERVAL:
                    logger.info("TTS trigger: unknown face detected")
                    self._last_tts_threat = now
                self._update_threat_state(best_box, best_confidence)

            self._state.threat_label = best_match

            if best_match == ThreatLabel.THREAT and best_box is not None:
                pan, tilt = _calculate_aim(*best_box)
                self._state.pan = pan
                self._state.tilt = tilt

    def _update_threat_state(
        self, box: tuple[int, int, int, int] | None, confidence: float
    ) -> None:
        face_crop = None
        if box is not None:
            l, t, r, b = box
        self._state.threat_state.active = True
        self._state.threat_state.confidence = confidence
        self._state.threat_state.timestamp = time.time()

    @staticmethod
    def _save_unknown(frame: np.ndarray, box: tuple[int, int, int, int] | None) -> None:
        try:
            unknown_dir = Path("face_data/unknowns")
            unknown_dir.mkdir(parents=True, exist_ok=True)
            ts = time.strftime("%Y%m%d_%H%M%S")
            if box is not None:
                l, t, r, b = box
                crop = frame[t:b, l:r]
                if crop.size > 0:
                    cv2.imwrite(str(unknown_dir / f"unknown_{ts}.jpg"), crop)
                    logger.info("Saved unknown face crop to unknowns/%s.jpg", ts)
        except (OSError, cv2.error) as e:
            logger.error("Failed to save unknown face: %s", e)


def face_recognition_loop(state: AppState, engine: FaceRecognitionEngine) -> None:
    """Daemon thread target for continuous face recognition."""
    while not state.stop_event.is_set():
        frame: np.ndarray | None = None
        with state:
            if state.frame_queue:
                frame = state.frame_queue[-1]
        if frame is not None:
            engine.process_frame(frame)
        else:
            with state:
                state.threat_label = ThreatLabel.IDLE
        time.sleep(0.03)
