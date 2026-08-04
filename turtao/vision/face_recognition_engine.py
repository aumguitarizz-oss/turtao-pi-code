from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import face_recognition
import numpy as np

from turtao.state import AppState, ThreatLabel
from turtao.vision.dlib_lock import DLIB_LOCK

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


@dataclass
class FaceSummary:
    name: str
    pose_count: int


@dataclass
class UnknownFaceSummary:
    id: str
    first_seen: str
    cluster_count: int


class FaceRecognitionEngine:
    def __init__(self, state: AppState, tolerance: float = 0.6) -> None:
        self._state = state
        self._tolerance = tolerance
        self._embeddings_dir: Path = Path(".")
        self._known_embeddings: list[np.ndarray] = []
        self._known_names: list[str] = []
        self._unknown_embeddings: list[np.ndarray] = []
        self._unknown_names: list[str] = []
        self._last_unknown_save = 0.0
        self._last_tts_threat = 0.0
        self._frames_since_seen = 0

    def load_embeddings(self, profile_dir: str) -> None:
        path = Path(profile_dir)
        self._embeddings_dir = path
        if not path.is_dir():
            logger.warning("Embedding directory not found: %s", profile_dir)
            return
        self._known_embeddings.clear()
        self._known_names.clear()
        for npy_file in sorted(path.iterdir()):
            if npy_file.suffix != ".npy":
                continue
            name = re.sub(r"_\d{3}$", "", npy_file.stem)
            emb = np.load(str(npy_file))
            self._known_embeddings.append(emb)
            self._known_names.append(name)
        logger.info(
            "Loaded %d embeddings from %s", len(self._known_embeddings), profile_dir
        )

    def list_faces(self) -> list[FaceSummary]:
        counts: dict[str, int] = {}
        for npy_file in sorted(self._embeddings_dir.glob("*.npy")):
            name = re.sub(r"_\d{3}$", "", npy_file.stem)
            counts[name] = counts.get(name, 0) + 1
        return [FaceSummary(name=n, pose_count=c) for n, c in counts.items()]

    def face_exists(self, name: str) -> bool:
        return any(f.name == name for f in self.list_faces())

    def get_thumb(self, name: str) -> bytes | None:
        thumb_path = self._embeddings_dir.parent / "thumbs" / f"{name}.jpg"
        if not thumb_path.is_file():
            return None
        return thumb_path.read_bytes()

    def delete_face(self, name: str) -> None:
        matches = [
            f for f in self._embeddings_dir.glob("*.npy")
            if re.sub(r"_\d{3}$", "", f.stem) == name
        ]
        if not matches:
            raise ValueError(f"Face '{name}' not found")
        for m in matches:
            m.unlink()
        self._remove_profile(name)
        self.load_embeddings(str(self._embeddings_dir))

    def _profiles_path(self) -> Path:
        return self._embeddings_dir.parent / "profiles.json"

    def _read_profiles(self) -> list[dict]:
        path = self._profiles_path()
        if not path.is_file():
            return []
        try:
            return json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            return []

    def _write_profiles(self, profiles: list[dict]) -> None:
        self._profiles_path().write_text(json.dumps(profiles, indent=2))

    def _upsert_profile(self, name: str, embeddings: list[str]) -> None:
        profiles = [p for p in self._read_profiles() if p.get("name") != name]
        profiles.append({
            "name": name,
            "embeddings": embeddings,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        })
        self._write_profiles(profiles)

    def _remove_profile(self, name: str) -> None:
        profiles = [p for p in self._read_profiles() if p.get("name") != name]
        self._write_profiles(profiles)

    def _unknowns_dir(self) -> Path:
        return self._embeddings_dir.parent / "unknowns"

    def list_unknowns(self) -> list[UnknownFaceSummary]:
        unknown_dir = self._unknowns_dir()
        if not unknown_dir.is_dir():
            return []
        out = []
        for jpg in sorted(unknown_dir.glob("unknown_*.jpg")):
            ts_str = jpg.stem.removeprefix("unknown_")
            first_seen = time.strftime(
                "%Y-%m-%dT%H:%M:%S", time.strptime(ts_str, "%Y%m%d_%H%M%S")
            )
            out.append(UnknownFaceSummary(id=jpg.stem, first_seen=first_seen, cluster_count=1))
        return out

    def get_unknown_thumb(self, id: str) -> bytes | None:
        path = self._unknowns_dir() / f"{id}.jpg"
        if not path.is_file():
            return None
        return path.read_bytes()

    def delete_unknown(self, id: str) -> None:
        path = self._unknowns_dir() / f"{id}.jpg"
        if not path.is_file():
            raise ValueError(f"Unknown '{id}' not found")
        path.unlink()

    def promote_unknown(self, face_id: str, name: str) -> None:
        src = self._unknowns_dir() / f"{face_id}.jpg"
        if not src.is_file():
            raise ValueError(f"Unknown '{face_id}' not found")
        if self.face_exists(name):
            raise ValueError(f"Face '{name}' already exists")
        frame = cv2.imread(str(src))
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        with DLIB_LOCK:
            encodings = face_recognition.face_encodings(rgb)
        if not encodings:
            raise ValueError("No face detected in unknown crop")
        np.save(str(self._embeddings_dir / f"{name}_000.npy"), encodings[0])
        src.unlink()
        self._upsert_profile(name, [f"{name}_000.npy"])
        self.load_embeddings(str(self._embeddings_dir))

    def process_frame(self, frame: np.ndarray) -> None:
        # Direct pipeline (shellalert-style): resize, detect, encode, compare.
        # No preprocessing and no temporal smoothing — both were shown to
        # hurt more than help (CPU starvation on Pi hardware, added latency)
        # versus the simple, proven approach.
        small = cv2.resize(frame, (0, 0), fx=RESIZE_SCALE, fy=RESIZE_SCALE)
        rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        # dlib's detector is not safe to call concurrently with enrollment's
        # own dlib calls on another thread — see dlib_lock.py.
        with DLIB_LOCK:
            face_locations = face_recognition.face_locations(rgb_small, model="hog")
            face_encodings = face_recognition.face_encodings(rgb_small, face_locations)

        if not face_locations:
            with self._state:
                self._frames_since_seen += 1
                if self._frames_since_seen > 5:  # persistence limit
                    self._state.threat_label = ThreatLabel.IDLE
                    self._state.threat_state.box = None
                    self._state.threat_state.landmarks = []
                    self._state.threat_state.name = ""
                    self._state.threat_state.active = False
            return

        match_label = ThreatLabel.THREAT
        name = "Unknown"
        box: tuple[int, int, int, int] | None = None
        confidence = 0.0

        for i, encoding in enumerate(face_encodings):
            top, right, bottom, left = face_locations[i]
            # Scale back to original resolution
            left = int(left / RESIZE_SCALE)
            top = int(top / RESIZE_SCALE)
            right = int(right / RESIZE_SCALE)
            bottom = int(bottom / RESIZE_SCALE)

            # Check against known enrolled profiles
            if self._known_embeddings:
                distances = [
                    np.linalg.norm(encoding - known)
                    for known in self._known_embeddings
                ]
                min_idx = int(np.argmin(distances))
                min_dist = distances[min_idx]

                if min_dist <= self._tolerance:
                    match_label = ThreatLabel.SAFE
                    confidence = 1.0 - min_dist
                    box = (left, top, right, bottom)
                    # _known_names already holds pose-suffix-stripped names
                    # (see load_embeddings).
                    name = self._known_names[min_idx]
                    break

            # Check session unknowns
            if self._unknown_embeddings:
                u_distances = [
                    np.linalg.norm(encoding - u_emb)
                    for u_emb in self._unknown_embeddings
                ]
                min_u_idx = int(np.argmin(u_distances))
                min_u_dist = u_distances[min_u_idx]

                if min_u_dist <= self._tolerance:
                    match_label = ThreatLabel.THREAT
                    confidence = 1.0 - min_u_dist
                    box = (left, top, right, bottom)
                    name = self._unknown_names[min_u_idx]
                    break

            # Completely new unknown face
            new_id = len(self._unknown_embeddings) + 1
            new_name = f"Unknown {new_id}"
            self._unknown_embeddings.append(encoding)
            self._unknown_names.append(new_name)

            match_label = ThreatLabel.THREAT
            confidence = 0.0
            box = (left, top, right, bottom)
            name = new_name
            break

        if match_label == ThreatLabel.THREAT:
            now = time.time()
            if now - self._last_unknown_save >= UNKNOWN_SAVE_INTERVAL:
                self._save_unknown(frame, box)
                self._last_unknown_save = now
            if now - self._last_tts_threat >= TTS_DEBOUNCE_INTERVAL:
                logger.info("TTS trigger: unknown face detected")
                self._last_tts_threat = now

        with self._state:
            self._frames_since_seen = 0
            self._update_threat_state(box, confidence, [], name)
            self._state.threat_label = match_label

            if match_label == ThreatLabel.THREAT and box is not None:
                pan, tilt = _calculate_aim(*box)
                self._state.pan = pan
                self._state.tilt = tilt

    def _update_threat_state(
        self,
        box: tuple[int, int, int, int] | None,
        confidence: float,
        landmarks: list[tuple[int, int]],
        name: str,
    ) -> None:
        self._state.threat_state.active = True
        self._state.threat_state.confidence = confidence
        self._state.threat_state.timestamp = time.time()
        self._state.threat_state.box = box
        self._state.threat_state.landmarks = landmarks
        self._state.threat_state.name = name

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
