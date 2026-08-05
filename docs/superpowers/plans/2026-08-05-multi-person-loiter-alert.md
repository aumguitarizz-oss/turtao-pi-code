# Multi-Person Detection + Loiter/Alert Workflow Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make face recognition and MediaPipe pose tracking handle multiple simultaneous people, add a loiter/alert workflow that records (0.5s) and alerts the app (2s) when a tracked person's face stays unresolved, and fix the delete-face GUI bug.

**Architecture:** `FaceRecognitionEngine.process_frame` gains a per-face loop producing a `list[FaceDetection]` on `ThreatState.faces`, keeping the existing single-value fields as a "worst case" summary for backward compatibility. `PoseTracker` moves from MediaPipe's legacy single-person `Pose` API to the Tasks API's multi-person `PoseLandmarker`. A new pure-logic `LoiterMonitor` correlates YOLO's per-person tracker IDs against per-face and per-pose data via a new 9th daemon thread in `core.py`, writing into the already-existing (but previously unfed) `state.events` — which the phone app already polls, so no app-repo changes are needed.

**Tech Stack:** Python 3.11+, `face_recognition` (dlib), MediaPipe Tasks API (`PoseLandmarker`), OpenCV, pytest with `unittest.mock`.

## Global Constraints

- Record threshold: **0.5s** of a tracked person having no overlapping face detection.
- Alert threshold: **2.0s** of the same, unresolved.
- Presence-confirmed gate: requires **both** a YOLO person box **and** at least one MediaPipe pose landmark set present that frame (per user's explicit ask to reduce false positives).
- Person-absence grace period before dropping a tracker's timer state: **1.0s**.
- Face region heuristic for correlating a person's bbox to a face: **top third** of the person's bounding box — reuse the exact fraction already used in `turtao/vision/person_tracker.py::_check_phone_spoof` (`face_bottom = py1 + (py2 - py1) // 3`).
- `num_poses=4` cap for MediaPipe `PoseLandmarker`.
- Recorded loiter crops MUST use the existing filename convention `unknown_<%Y%m%d_%H%M%S>.jpg` in `face_data/unknowns/` so they appear in the current GUI Unknowns tab / `/api/faces/unknowns` / promote-to-known flow with zero new code there.
- Alert `Event` fields: `id`, `type="unidentified_person"`, `message`, `at` (ISO `%Y-%m-%dT%H:%M:%S`) — matches the `Event` dataclass already in `turtao/state.py` and the phone app's `AppEvent` model exactly; no app-repo changes.
- No changes to the `turtaoapp` repo in this plan.
- No changes to `_calculate_aim`/servo targeting beyond aiming at the existing "summary" (worst-case) face.
- All 230 existing tests must stay green after every task.
- GUI code (`turtao/gui/`) has no existing unit test coverage in this repo (confirmed: no test imports tkinter or any `turtao.gui.*` module) — do not introduce a new GUI testing pattern; verify GUI-facing tasks manually as specified in each task.
- `mediapipe` is currently commented out in `requirements.txt` and not installed in this dev environment — pose-tracker tests must mock the `mediapipe` module the same way existing tests mock `face_recognition`/`cv2` (patch `turtao.vision.pose_tracker.mp`, `.mp_vision`, `.BaseOptions` directly), since it cannot be imported for real here.

---

### Task 1: `FaceDetection` dataclass + `ThreatState.faces` field

**Files:**
- Modify: `turtao/state.py`
- Test: `tests/unit/test_state.py` (new file)

**Interfaces:**
- Consumes: `ThreatLabel` (existing enum in this file).
- Produces: `FaceDetection` dataclass (`box: tuple[int,int,int,int]`, `name: str`, `label: ThreatLabel`, `confidence: float`) and `ThreatState.faces: list[FaceDetection]` (default empty list) — used by Task 2 (engine), Task 3 (LoiterMonitor), Task 8 (GUI).

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_state.py`:

```python
from turtao.state import FaceDetection, ThreatState, ThreatLabel


class TestFaceDetection:
    def test_construction(self):
        fd = FaceDetection(box=(1, 2, 3, 4), name="alice", label=ThreatLabel.SAFE, confidence=0.8)
        assert fd.box == (1, 2, 3, 4)
        assert fd.name == "alice"
        assert fd.label == ThreatLabel.SAFE
        assert fd.confidence == 0.8


class TestThreatStateFaces:
    def test_defaults_to_empty_list(self):
        ts = ThreatState()
        assert ts.faces == []

    def test_faces_is_independent_per_instance(self):
        ts1 = ThreatState()
        ts2 = ThreatState()
        ts1.faces.append(FaceDetection(box=(0, 0, 1, 1), name="x", label=ThreatLabel.THREAT, confidence=0.0))
        assert ts2.faces == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `venv/bin/python3 -m pytest tests/unit/test_state.py -v`
Expected: FAIL — `ImportError: cannot import name 'FaceDetection' from 'turtao.state'`

- [ ] **Step 3: Write minimal implementation**

In `turtao/state.py`, add the `FaceDetection` dataclass immediately before `ThreatState`, and add the `faces` field to `ThreatState`:

```python
@dataclass
class FaceDetection:
    box: tuple[int, int, int, int]
    name: str
    label: ThreatLabel
    confidence: float


@dataclass
class ThreatState:
    active: bool = False
    face_crop: bytes | None = None
    confidence: float = 0.0
    timestamp: float | None = None
    box: tuple[int, int, int, int] | None = None
    landmarks: list[tuple[int, int]] = field(default_factory=list)
    name: str = ""
    faces: list[FaceDetection] = field(default_factory=list)
```

(Only the `faces: list[FaceDetection] = field(default_factory=list)` line and the new `FaceDetection` class are additions — every other line already exists in `turtao/state.py` verbatim; do not reorder or duplicate the existing fields.)

- [ ] **Step 4: Run test to verify it passes**

Run: `venv/bin/python3 -m pytest tests/unit/test_state.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Run full suite to confirm no regressions**

Run: `venv/bin/python3 -m pytest -q`
Expected: 233 passed (230 existing + 3 new), 1 skipped

- [ ] **Step 6: Commit**

```bash
git add turtao/state.py tests/unit/test_state.py
git commit -m "feat: add FaceDetection dataclass and ThreatState.faces field"
```

---

### Task 2: Multi-face recognition in `FaceRecognitionEngine.process_frame`

**Files:**
- Modify: `turtao/vision/face_recognition_engine.py:199-313` (the `process_frame` and `_update_threat_state` methods)
- Test: `tests/unit/test_face_recognition_engine.py`

**Interfaces:**
- Consumes: `FaceDetection`, `ThreatState.faces` (Task 1).
- Produces: `process_frame` now populates `state.threat_state.faces` with one `FaceDetection` per detected face this frame, in addition to the existing summary fields (`box`, `name`, `confidence` = the worst-case face: any THREAT beats any SAFE, first found among ties). `state.threat_label` = THREAT if any face is THREAT, else SAFE if any face is SAFE, else unchanged (no faces this frame is handled by the existing early-return branch). Later tasks (3, 8) read `state.threat_state.faces`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/test_face_recognition_engine.py` (new test class, same file, same mocking conventions already used above it):

```python
class TestMultiFaceRecognition:
    @patch("turtao.vision.face_recognition_engine.face_recognition")
    @patch("turtao.vision.face_recognition_engine.cv2")
    def test_two_known_faces_both_appear_in_faces_list(
        self, mock_cv2, mock_fr, app_state: AppState
    ):
        mock_fr.face_locations.return_value = [(100, 200, 300, 50), (100, 500, 300, 350)]
        alice_encoding = np.array([1.0, 0.0, 0.0, 0.0])
        bob_encoding = np.array([0.0, 1.0, 0.0, 0.0])
        mock_fr.face_encodings.return_value = [alice_encoding, bob_encoding]
        mock_cv2.resize.return_value = np.zeros((240, 320, 3), dtype=np.uint8)
        mock_cv2.cvtColor.return_value = np.zeros((240, 320, 3), dtype=np.uint8)

        engine = FaceRecognitionEngine(app_state, tolerance=0.1)
        engine._known_embeddings = [alice_encoding, bob_encoding]
        engine._known_names = ["alice", "bob"]
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        engine.process_frame(frame)

        faces = app_state.threat_state.faces
        assert len(faces) == 2
        names = {f.name for f in faces}
        assert names == {"alice", "bob"}
        assert all(f.label == ThreatLabel.SAFE for f in faces)
        assert app_state.threat_label == ThreatLabel.SAFE

    @patch("turtao.vision.face_recognition_engine.face_recognition")
    @patch("turtao.vision.face_recognition_engine.cv2")
    def test_one_known_one_unknown_summary_is_threat(
        self, mock_cv2, mock_fr, app_state: AppState
    ):
        mock_fr.face_locations.return_value = [(100, 200, 300, 50), (100, 500, 300, 350)]
        alice_encoding = np.array([1.0, 0.0, 0.0, 0.0])
        stranger_encoding = np.array([0.0, 0.0, 1.0, 0.0])
        mock_fr.face_encodings.return_value = [alice_encoding, stranger_encoding]
        mock_cv2.resize.return_value = np.zeros((240, 320, 3), dtype=np.uint8)
        mock_cv2.cvtColor.return_value = np.zeros((240, 320, 3), dtype=np.uint8)

        engine = FaceRecognitionEngine(app_state, tolerance=0.1)
        engine._known_embeddings = [alice_encoding]
        engine._known_names = ["alice"]
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        engine.process_frame(frame)

        faces = app_state.threat_state.faces
        assert len(faces) == 2
        labels = {f.name: f.label for f in faces}
        assert labels["alice"] == ThreatLabel.SAFE
        assert any(v == ThreatLabel.THREAT for k, v in labels.items() if k != "alice")
        # Global summary must be THREAT (any-unresolved-face wins)
        assert app_state.threat_label == ThreatLabel.THREAT
        assert app_state.threat_state.name != "alice"

    @patch("turtao.vision.face_recognition_engine.face_recognition")
    @patch("turtao.vision.face_recognition_engine.cv2")
    def test_no_faces_clears_faces_list(
        self, mock_cv2, mock_fr, app_state: AppState
    ):
        mock_fr.face_locations.return_value = []
        mock_fr.face_encodings.return_value = []
        mock_cv2.resize.return_value = np.zeros((240, 320, 3), dtype=np.uint8)
        mock_cv2.cvtColor.return_value = np.zeros((240, 320, 3), dtype=np.uint8)

        engine = FaceRecognitionEngine(app_state)
        engine._frames_since_seen = 10  # force the IDLE-reset branch immediately
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        engine.process_frame(frame)

        assert app_state.threat_state.faces == []
```

Add `ThreatLabel` to the existing import line at the top of the test file if not already present (it already is — `from turtao.state import AppState, ThreatLabel`).

- [ ] **Step 2: Run tests to verify they fail**

Run: `venv/bin/python3 -m pytest tests/unit/test_face_recognition_engine.py::TestMultiFaceRecognition -v`
Expected: FAIL — `assert len(faces) == 2` fails with `len(faces) == 0` (or `AssertionError` on the second face never being processed), because `process_frame` currently `break`s after the first face.

- [ ] **Step 3: Rewrite `process_frame` and `_update_threat_state`**

Replace lines 199-313 of `turtao/vision/face_recognition_engine.py` (the entire `process_frame` method through the end of `_update_threat_state`) with:

```python
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
                    self._state.threat_state.faces = []
            return

        faces: list[FaceDetection] = []

        for i, encoding in enumerate(face_encodings):
            top, right, bottom, left = face_locations[i]
            # Scale back to original resolution
            left = int(left / RESIZE_SCALE)
            top = int(top / RESIZE_SCALE)
            right = int(right / RESIZE_SCALE)
            bottom = int(bottom / RESIZE_SCALE)
            box = (left, top, right, bottom)

            # Check against known enrolled profiles
            if self._known_embeddings:
                distances = [
                    np.linalg.norm(encoding - known)
                    for known in self._known_embeddings
                ]
                min_idx = int(np.argmin(distances))
                min_dist = distances[min_idx]

                if min_dist <= self._tolerance:
                    # _known_names already holds pose-suffix-stripped names
                    # (see load_embeddings).
                    faces.append(FaceDetection(
                        box=box, name=self._known_names[min_idx],
                        label=ThreatLabel.SAFE, confidence=1.0 - min_dist,
                    ))
                    continue

            # Check session unknowns
            if self._unknown_embeddings:
                u_distances = [
                    np.linalg.norm(encoding - u_emb)
                    for u_emb in self._unknown_embeddings
                ]
                min_u_idx = int(np.argmin(u_distances))
                min_u_dist = u_distances[min_u_idx]

                if min_u_dist <= self._tolerance:
                    faces.append(FaceDetection(
                        box=box, name=self._unknown_names[min_u_idx],
                        label=ThreatLabel.THREAT, confidence=1.0 - min_u_dist,
                    ))
                    continue

            # Completely new unknown face
            new_id = len(self._unknown_embeddings) + 1
            new_name = f"Unknown {new_id}"
            self._unknown_embeddings.append(encoding)
            self._unknown_names.append(new_name)
            faces.append(FaceDetection(
                box=box, name=new_name, label=ThreatLabel.THREAT, confidence=0.0,
            ))

        # Summary = worst-case face this frame: any THREAT beats any SAFE,
        # first-found among ties. There's always >=1 face here since we
        # returned early above when face_locations was empty.
        threat_faces = [f for f in faces if f.label == ThreatLabel.THREAT]
        summary = threat_faces[0] if threat_faces else faces[0]
        match_label = summary.label

        if threat_faces:
            now = time.time()
            if now - self._last_unknown_save >= UNKNOWN_SAVE_INTERVAL:
                self._save_unknown(frame, threat_faces[0].box)
                self._last_unknown_save = now
            if now - self._last_tts_threat >= TTS_DEBOUNCE_INTERVAL:
                logger.info("TTS trigger: unknown face detected")
                self._last_tts_threat = now

        with self._state:
            self._frames_since_seen = 0
            self._update_threat_state(summary.box, summary.confidence, [], summary.name, faces)
            self._state.threat_label = match_label

            if match_label == ThreatLabel.THREAT:
                pan, tilt = _calculate_aim(*summary.box)
                self._state.pan = pan
                self._state.tilt = tilt

    def _update_threat_state(
        self,
        box: tuple[int, int, int, int] | None,
        confidence: float,
        landmarks: list[tuple[int, int]],
        name: str,
        faces: list[FaceDetection],
    ) -> None:
        self._state.threat_state.active = True
        self._state.threat_state.confidence = confidence
        self._state.threat_state.timestamp = time.time()
        self._state.threat_state.box = box
        self._state.threat_state.landmarks = landmarks
        self._state.threat_state.name = name
        self._state.threat_state.faces = faces
```

Add `FaceDetection` to the import line near the top of the file:

```python
from turtao.state import AppState, FaceDetection, ThreatLabel
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `venv/bin/python3 -m pytest tests/unit/test_face_recognition_engine.py -v`
Expected: PASS (all tests in the file, including the 3 new ones and every pre-existing one)

- [ ] **Step 5: Run full suite**

Run: `venv/bin/python3 -m pytest -q`
Expected: all green, no regressions

- [ ] **Step 6: Lint**

Run: `venv/bin/ruff check turtao/vision/face_recognition_engine.py`
Expected: no new findings beyond the two pre-existing `E741` (`l` variable name) findings already present before this plan.

- [ ] **Step 7: Commit**

```bash
git add turtao/vision/face_recognition_engine.py tests/unit/test_face_recognition_engine.py
git commit -m "feat: recognize multiple faces per frame instead of stopping at the first"
```

---

### Task 3: `LoiterMonitor` — pure timing/correlation logic

**Files:**
- Create: `turtao/vision/loiter_monitor.py`
- Test: `tests/unit/test_loiter_monitor.py`

**Interfaces:**
- Consumes: `FaceDetection` (Task 1), person dicts shaped like `person_tracker.py`'s output (`{"bbox": (x1,y1,x2,y2), "tracker_id": int, "class_name": str, ...}`).
- Produces: `LoiterMonitor` class with `update(persons, pose_present, faces, frame, now, record_crop, emit_alert) -> None`. `record_crop: Callable[[np.ndarray, tuple[int,int,int,int]], None]` and `emit_alert: Callable[[str], None]` are injected callbacks — Task 4 wires these to real frame-saving and `state.events` appending. Keeping them as callbacks (rather than baking I/O into this class) is what makes this class testable with plain synthetic data and no mocking of cv2/filesystem/state locking.

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_loiter_monitor.py`:

```python
import numpy as np
import pytest
from turtao.state import FaceDetection, ThreatLabel
from turtao.vision.loiter_monitor import LoiterMonitor


def _person(tracker_id=1, bbox=(100, 100, 200, 300), class_name="Person"):
    return {"bbox": bbox, "tracker_id": tracker_id, "class_name": class_name, "confidence": 0.9}


def _face_in(bbox, name="alice"):
    x1, y1, x2, y2 = bbox
    face_top = y1
    face_bottom = y1 + (y2 - y1) // 3
    cx = (x1 + x2) // 2
    cy = (face_top + face_bottom) // 2
    return FaceDetection(box=(cx - 10, cy - 10, cx + 10, cy + 10), name=name, label=ThreatLabel.SAFE, confidence=0.9)


class TestLoiterMonitor:
    def test_no_alert_while_pose_not_present(self):
        monitor = LoiterMonitor()
        recorded, alerted = [], []
        monitor.update(
            persons=[_person()], pose_present=False, faces=[],
            frame=np.zeros((480, 640, 3), dtype=np.uint8), now=100.0,
            record_crop=recorded.append, emit_alert=alerted.append,
        )
        assert recorded == []
        assert alerted == []

    def test_records_after_half_second_missing_face(self):
        monitor = LoiterMonitor()
        recorded, alerted = [], []
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        monitor.update([_person()], True, [], frame, now=100.0,
                        record_crop=lambda f, b: recorded.append(b), emit_alert=alerted.append)
        assert recorded == []  # not yet at threshold

        monitor.update([_person()], True, [], frame, now=100.5,
                        record_crop=lambda f, b: recorded.append(b), emit_alert=alerted.append)
        assert recorded == [(100, 100, 200, 300)]
        assert alerted == []  # record fires before alert

    def test_alerts_after_two_seconds_missing_face(self):
        monitor = LoiterMonitor()
        recorded, alerted = [], []
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        monitor.update([_person()], True, [], frame, now=100.0,
                        record_crop=lambda f, b: recorded.append(b), emit_alert=alerted.append)
        monitor.update([_person()], True, [], frame, now=102.0,
                        record_crop=lambda f, b: recorded.append(b), emit_alert=alerted.append)

        assert len(recorded) == 1  # recorded once, not every cycle
        assert len(alerted) == 1
        assert "tracker #1" in alerted[0]

    def test_face_resolving_resets_timer(self):
        monitor = LoiterMonitor()
        recorded, alerted = [], []
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        person = _person()

        monitor.update([person], True, [], frame, now=100.0,
                        record_crop=lambda f, b: recorded.append(b), emit_alert=alerted.append)
        # Face resolves before the 0.5s record threshold
        monitor.update([person], True, [_face_in(person["bbox"])], frame, now=100.2,
                        record_crop=lambda f, b: recorded.append(b), emit_alert=alerted.append)
        # Now missing again for another 0.6s — should re-trigger record, not
        # be treated as continuously missing since 100.0
        monitor.update([person], True, [], frame, now=100.8,
                        record_crop=lambda f, b: recorded.append(b), emit_alert=alerted.append)

        assert recorded == [person["bbox"]]  # fired once, from the second missing episode
        assert alerted == []  # never reached 2s of *continuous* missing

    def test_person_leaving_frame_clears_state_after_grace_period(self):
        monitor = LoiterMonitor()
        recorded, alerted = [], []
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        person = _person()

        monitor.update([person], True, [], frame, now=100.0,
                        record_crop=lambda f, b: recorded.append(b), emit_alert=alerted.append)
        # Person gone; > 1.0s grace period passes
        monitor.update([], True, [], frame, now=101.5,
                        record_crop=lambda f, b: recorded.append(b), emit_alert=alerted.append)
        # Same tracker_id reappears — must behave like a fresh episode, not
        # inherit the old first_missing_at
        monitor.update([person], True, [], frame, now=200.0,
                        record_crop=lambda f, b: recorded.append(b), emit_alert=alerted.append)
        assert len(recorded) == 1  # only from the very first episode so far
        monitor.update([person], True, [], frame, now=200.5,
                        record_crop=lambda f, b: recorded.append(b), emit_alert=alerted.append)
        assert len(recorded) == 2  # second episode's own 0.5s record fired

    def test_non_person_class_ignored(self):
        monitor = LoiterMonitor()
        recorded, alerted = [], []
        monitor.update(
            persons=[_person(class_name="Cell Phone")], pose_present=True, faces=[],
            frame=np.zeros((480, 640, 3), dtype=np.uint8), now=100.5,
            record_crop=recorded.append, emit_alert=alerted.append,
        )
        assert recorded == []
        assert alerted == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `venv/bin/python3 -m pytest tests/unit/test_loiter_monitor.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'turtao.vision.loiter_monitor'`

- [ ] **Step 3: Implement `LoiterMonitor`**

Create `turtao/vision/loiter_monitor.py`:

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from turtao.state import FaceDetection

FACE_MISSING_RECORD_THRESHOLD = 0.5   # seconds
FACE_MISSING_ALERT_THRESHOLD = 2.0    # seconds
PERSON_ABSENT_GRACE_PERIOD = 1.0      # seconds


@dataclass
class _PersonTimer:
    first_missing_at: float | None = None
    last_seen_at: float = 0.0
    recorded: bool = False
    alerted: bool = False


class LoiterMonitor:
    """Tracks, per YOLO tracker_id, how long a person's face has gone
    unresolved, firing a one-shot record callback at 0.5s and a one-shot
    alert callback at 2.0s. Pure logic — no I/O, no locking, no threads —
    so it's driven by (and testable with) plain synthetic snapshots and an
    explicit `now` rather than the real clock."""

    def __init__(self) -> None:
        self._timers: dict[int, _PersonTimer] = {}

    def update(
        self,
        persons: list[dict[str, Any]],
        pose_present: bool,
        faces: list[FaceDetection],
        frame: np.ndarray | None,
        now: float,
        record_crop: Callable[[np.ndarray, tuple[int, int, int, int]], None],
        emit_alert: Callable[[str], None],
    ) -> None:
        seen_ids: set[int] = set()

        for person in persons:
            if person.get("class_name") != "Person":
                continue
            tracker_id = person.get("tracker_id", -1)
            if tracker_id < 0:
                continue
            bbox = person["bbox"]

            seen_ids.add(tracker_id)
            timer = self._timers.setdefault(tracker_id, _PersonTimer())
            timer.last_seen_at = now

            if not pose_present:
                continue

            if self._face_overlaps(bbox, faces):
                timer.first_missing_at = None
                timer.recorded = False
                timer.alerted = False
                continue

            if timer.first_missing_at is None:
                timer.first_missing_at = now
            missing_for = now - timer.first_missing_at

            if missing_for >= FACE_MISSING_RECORD_THRESHOLD and not timer.recorded:
                if frame is not None:
                    record_crop(frame, bbox)
                timer.recorded = True

            if missing_for >= FACE_MISSING_ALERT_THRESHOLD and not timer.alerted:
                emit_alert(
                    f"Unrecognized person (tracker #{tracker_id}) unresolved "
                    f"for {FACE_MISSING_ALERT_THRESHOLD:.0f}s+"
                )
                timer.alerted = True

        stale = [
            tid for tid, t in self._timers.items()
            if tid not in seen_ids and now - t.last_seen_at > PERSON_ABSENT_GRACE_PERIOD
        ]
        for tid in stale:
            del self._timers[tid]

    @staticmethod
    def _face_overlaps(
        bbox: tuple[int, int, int, int], faces: list[FaceDetection]
    ) -> bool:
        px1, py1, px2, py2 = bbox
        face_top = py1
        face_bottom = py1 + (py2 - py1) // 3
        for f in faces:
            fx1, fy1, fx2, fy2 = f.box
            fcx = (fx1 + fx2) / 2
            fcy = (fy1 + fy2) / 2
            if px1 <= fcx <= px2 and face_top <= fcy <= face_bottom:
                return True
        return False
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `venv/bin/python3 -m pytest tests/unit/test_loiter_monitor.py -v`
Expected: PASS (7 tests)

- [ ] **Step 5: Run full suite**

Run: `venv/bin/python3 -m pytest -q`
Expected: all green

- [ ] **Step 6: Lint**

Run: `venv/bin/ruff check turtao/vision/loiter_monitor.py`
Expected: clean

- [ ] **Step 7: Commit**

```bash
git add turtao/vision/loiter_monitor.py tests/unit/test_loiter_monitor.py
git commit -m "feat: add LoiterMonitor — per-person missing-face timing logic"
```

---

### Task 4: Wire `LoiterMonitor` into `TurtaoCore` as a 9th daemon thread

**Files:**
- Modify: `turtao/core.py`
- Test: `tests/unit/test_core_loiter_wrapper.py` (new file)

**Interfaces:**
- Consumes: `LoiterMonitor.update(...)` (Task 3), `state.latest_persons`, `state.pose_landmarks` (still the *old* flat-list shape until Task 6 — treat `pose_present = bool(state.pose_landmarks)`, which works for both the old and new shapes since both are falsy-when-empty lists), `state.threat_state.faces` (Task 2), `state.events`, `state.event_counter` (both already exist in `AppState`), `Event` dataclass (already exists in `turtao/state.py`).
- Produces: `TurtaoCore._loiter_wrapper` thread method, `TurtaoCore._record_loiter_crop`, `TurtaoCore._emit_loiter_alert` — no other module calls these; this task is self-contained glue.

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_core_loiter_wrapper.py`:

```python
import time
from pathlib import Path

import numpy as np
import pytest

from turtao.config import AppConfig, Settings
from turtao.core import TurtaoCore
from turtao.hardware.mocks import MockCamera, MockSerialLink
from turtao.state import AppState, Event, FaceDetection, ThreatLabel


@pytest.fixture
def core(tmp_path):
    state = AppState()
    c = TurtaoCore(
        config=AppConfig(),
        settings=Settings(),
        state=state,
        serial_link=MockSerialLink(),
        camera=MockCamera(),
        face_data_dir=tmp_path / "face_data",
    )
    return c


class TestRecordLoiterCrop:
    def test_saves_crop_to_unknowns_dir(self, core, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        frame = np.full((480, 640, 3), 200, dtype=np.uint8)
        core._record_loiter_crop(frame, (10, 10, 100, 100))

        saved = list(Path("face_data/unknowns").glob("unknown_*.jpg"))
        assert len(saved) == 1


class TestEmitLoiterAlert:
    def test_appends_event_to_state(self, core):
        core._emit_loiter_alert("Unrecognized person (tracker #7) unresolved for 2s+")

        assert len(core.state.events) == 1
        evt = core.state.events[0]
        assert evt.type == "unidentified_person"
        assert "tracker #7" in evt.message
        assert evt.id  # non-empty
        assert evt.at  # non-empty


class TestLoiterWrapperIntegration:
    def test_one_loop_iteration_drives_monitor(self, core):
        """Directly exercises the per-iteration body the thread runs, via
        the extracted _loiter_tick helper, rather than starting a real
        thread (which would be slow/flaky to assert against)."""
        with core.state:
            core.state.latest_persons = [
                {"bbox": (100, 100, 200, 300), "tracker_id": 1, "class_name": "Person", "confidence": 0.9}
            ]
            core.state.pose_landmarks = [[(150, 150)]]  # non-empty => pose_present
            core.state.threat_state.faces = []  # no resolved face
            core.state.latest_frame = np.zeros((480, 640, 3), dtype=np.uint8)

        core._loiter_tick(now=1000.0)
        core._loiter_tick(now=1002.5)  # past both 0.5s and 2.0s thresholds

        assert len(core.state.events) == 1
        assert "tracker #1" in core.state.events[0].message
```

- [ ] **Step 2: Run test to verify it fails**

Run: `venv/bin/python3 -m pytest tests/unit/test_core_loiter_wrapper.py -v`
Expected: FAIL — `AttributeError: 'TurtaoCore' object has no attribute '_record_loiter_crop'`

- [ ] **Step 3: Implement the wiring**

In `turtao/core.py`, add the import (alongside the existing `turtao.vision.*` imports):

```python
from turtao.vision.loiter_monitor import LoiterMonitor
```

In `TurtaoCore.__init__`, immediately after the `self.pose_tracker = PoseTracker(state)` line, add:

```python
        self.loiter_monitor = LoiterMonitor()
```

In `TurtaoCore.start()`, add a new entry to `threads_config` (after `"_face_recognition_loop"` is a natural place, but anywhere in the list works — order doesn't matter, each spawns its own thread):

```python
            ("_loiter_wrapper", self._loiter_wrapper, ()),
```

Add these new methods to `TurtaoCore` (place them near `_tracker_wrapper`, in the "Thread wrappers" section):

```python
    def _loiter_wrapper(self) -> None:
        """Poll person/face/pose state and drive the loiter monitor."""
        while not self.state.stop_event.is_set():
            self._loiter_tick(time.time())
            time.sleep(0.1)

    def _loiter_tick(self, now: float) -> None:
        with self.state:
            persons = list(self.state.latest_persons)
            pose_present = bool(self.state.pose_landmarks)
            faces = list(self.state.threat_state.faces)
            frame = self.state.latest_frame

        self.loiter_monitor.update(
            persons=persons,
            pose_present=pose_present,
            faces=faces,
            frame=frame,
            now=now,
            record_crop=self._record_loiter_crop,
            emit_alert=self._emit_loiter_alert,
        )

    def _record_loiter_crop(
        self, frame: Any, bbox: tuple[int, int, int, int]
    ) -> None:
        import cv2

        try:
            x1, y1, x2, y2 = bbox
            crop = frame[max(0, y1):y2, max(0, x1):x2]
            if crop.size == 0:
                return
            unknown_dir = Path("face_data/unknowns")
            unknown_dir.mkdir(parents=True, exist_ok=True)
            ts = time.strftime("%Y%m%d_%H%M%S")
            cv2.imwrite(str(unknown_dir / f"unknown_{ts}.jpg"), crop)
            logger.info("Loiter: saved unrecognized person crop to unknowns/%s.jpg", ts)
        except Exception:
            logger.exception("Loiter: failed to save crop")

    def _emit_loiter_alert(self, message: str) -> None:
        with self.state:
            self.state.event_counter += 1
            self.state.events.append(Event(
                id=f"evt_{self.state.event_counter}",
                type="unidentified_person",
                message=message,
                at=time.strftime("%Y-%m-%dT%H:%M:%S"),
            ))
        logger.warning("Loiter alert: %s", message)
```

Add `Event` to the existing state import line in `turtao/core.py`:

```python
from turtao.state import AppState, Event, Mode, ThreatLabel
```

`time` and `Path` are already imported at the top of `core.py`; `Any` is already imported from `typing`.

- [ ] **Step 4: Run test to verify it passes**

Run: `venv/bin/python3 -m pytest tests/unit/test_core_loiter_wrapper.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Run full suite**

Run: `venv/bin/python3 -m pytest -q`
Expected: all green, thread count in the `TurtaoCore started with %d threads` log line is now 9 (verify by grepping a manual run's log, not a hard test assertion — thread count isn't asserted anywhere in the existing suite)

- [ ] **Step 6: Lint**

Run: `venv/bin/ruff check turtao/core.py`
Expected: no new findings

- [ ] **Step 7: Commit**

```bash
git add turtao/core.py tests/unit/test_core_loiter_wrapper.py
git commit -m "feat: wire LoiterMonitor into TurtaoCore as a 9th daemon thread"
```

---

### Task 5: Fix the delete-face GUI bug

**Files:**
- Modify: `turtao/gui/tab_faces.py:125-135`

**Interfaces:**
- Consumes: `FaceRecognitionEngine.delete_face(name: str) -> None` (already exists, already tested in `tests/unit/test_face_recognition_engine.py::TestDeleteFace`, already raises `ValueError` if the name doesn't exist).
- Produces: nothing new consumed elsewhere — this is a leaf fix.

No new automated test: `turtao/gui/` has zero existing unit test coverage in this repo (mypy explicitly excludes it, no test file imports tkinter), and the method being called (`FaceRecognitionEngine.delete_face`) already has full coverage proving it works correctly. Introducing a new Tkinter test harness for one call site would be a new, unestablished pattern out of proportion to the fix. Verify manually per Step 3 below instead.

- [ ] **Step 1: Confirm the target behavior already exists and is tested**

Run: `venv/bin/python3 -m pytest tests/unit/test_face_recognition_engine.py::TestDeleteFace -v`
Expected: PASS (3 tests: `test_removes_all_poses`, `test_raises_value_error_when_absent`, `test_does_not_delete_other_face_with_prefix_name`, `test_removes_profile_entry` — confirms `delete_face` correctly removes `.npy` files, updates `profiles.json`, and reloads in-memory embeddings)

- [ ] **Step 2: Replace the hand-rolled deletion**

In `turtao/gui/tab_faces.py`, replace the entire `_delete_face` method:

```python
    def _delete_face(self, name: str) -> None:
        try:
            profiles_path = Path("face_data/profiles.json")
            if not profiles_path.exists():
                return
            profiles = json.loads(profiles_path.read_text())
            profiles = [p for p in profiles if p.get("name") != name]
            profiles_path.write_text(json.dumps(profiles, indent=2))
        except (json.JSONDecodeError, OSError):
            pass
```

with:

```python
    def _delete_face(self, name: str) -> None:
        try:
            self.core.face_engine.delete_face(name)
        except ValueError:
            pass
```

The `json` and `Path` imports at the top of the file may now be unused if nothing else in `tab_faces.py` uses them — check with:

Run: `grep -n "json\.\|Path(" turtao/gui/tab_faces.py`

If `json.` and `Path(` no longer appear anywhere else in the file (they're still used by `refresh()` and `_load_thumb()` for reading `profiles.json` and constructing paths — expect them to still be needed), leave the imports; otherwise remove the now-unused one(s).

- [ ] **Step 3: Manual verification (no automated GUI test exists in this repo)**

On a machine with a display (or the Pi with `python main.py --gui`):
1. Enroll a face (5 poses) via the GUI's Enroll tab.
2. Switch to the Faces tab, confirm the enrolled name appears.
3. Enable GUARD mode and confirm the recognition tab labels you as SAFE with your name.
4. Click Delete on your face in the Faces tab.
5. Confirm: the Faces tab card disappears, `face_data/embeddings/<name>_*.npy` files are gone from disk, and — without restarting the app — the recognition tab now labels you as an unrecognized THREAT instead of SAFE (this last part is the actual regression check: before this fix, step 5 would still show SAFE because the running engine's in-memory embeddings were never reloaded).

- [ ] **Step 4: Lint**

Run: `venv/bin/ruff check turtao/gui/tab_faces.py`
Expected: no new findings

- [ ] **Step 5: Commit**

```bash
git add turtao/gui/tab_faces.py
git commit -m "fix: GUI delete-face button now actually deletes the embedding and reloads the engine"
```

---

### Task 6: Multi-person MediaPipe pose via the Tasks API

**Files:**
- Modify: `requirements.txt` (uncomment mediapipe)
- Modify: `turtao/vision/pose_tracker.py` (full rewrite)
- Modify: `turtao/state.py` (change `pose_landmarks` type comment/shape — see below)
- Test: `tests/unit/test_pose_tracker.py` (new file)

**Interfaces:**
- Consumes: nothing new from earlier tasks.
- Produces: `PoseTracker.process_frame(frame)` now writes `state.pose_landmarks` as `list[list[tuple[int,int]]]` (one inner list per detected person) instead of a flat `list[tuple[int,int]]`. Task 4's `pose_present = bool(state.pose_landmarks)` check already works correctly against this new shape unchanged (an empty list of lists is still falsy). Task 7 (GUI) and Task 8 (install.sh model download) depend on this.

- [ ] **Step 1: Uncomment the dependency**

In `requirements.txt`, change:

```
# mediapipe>=0.10
```

to:

```
mediapipe>=0.10
```

- [ ] **Step 2: Write the failing tests**

Create `tests/unit/test_pose_tracker.py`:

```python
from unittest.mock import MagicMock, patch

import numpy as np

from turtao.state import AppState
from turtao.vision.pose_tracker import PoseTracker


def _make_landmark(x, y):
    lm = MagicMock()
    lm.x = x
    lm.y = y
    return lm


class TestPoseTrackerMultiPerson:
    @patch("turtao.vision.pose_tracker.mp")
    @patch("turtao.vision.pose_tracker.mp_vision")
    @patch("turtao.vision.pose_tracker.BaseOptions")
    @patch("turtao.vision.pose_tracker.cv2")
    def test_two_poses_produce_two_landmark_lists(
        self, mock_cv2, mock_base_options, mock_mp_vision, mock_mp, app_state: AppState, tmp_path
    ):
        model_path = tmp_path / "pose_landmarker_lite.task"
        model_path.write_bytes(b"fake")

        mock_cv2.cvtColor.return_value = np.zeros((480, 640, 3), dtype=np.uint8)

        landmarker = MagicMock()
        mock_mp_vision.PoseLandmarker.create_from_options.return_value = landmarker

        person_a = [_make_landmark(0.1, 0.2)] * 33
        person_b = [_make_landmark(0.6, 0.5)] * 33
        result = MagicMock()
        result.pose_landmarks = [person_a, person_b]
        landmarker.detect_for_video.return_value = result

        tracker = PoseTracker(app_state, model_path=str(model_path))
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        tracker.process_frame(frame)

        assert len(app_state.pose_landmarks) == 2
        assert len(app_state.pose_landmarks[0]) == 33
        assert len(app_state.pose_landmarks[1]) == 33

    @patch("turtao.vision.pose_tracker.mp")
    @patch("turtao.vision.pose_tracker.mp_vision")
    @patch("turtao.vision.pose_tracker.BaseOptions")
    def test_missing_model_file_disables_tracker_gracefully(
        self, mock_base_options, mock_mp_vision, mock_mp, app_state: AppState
    ):
        tracker = PoseTracker(app_state, model_path="/nonexistent/model.task")
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        tracker.process_frame(frame)  # must not raise
        assert app_state.pose_landmarks == []

    def test_missing_mediapipe_import_disables_tracker_gracefully(self, app_state: AppState):
        with patch("turtao.vision.pose_tracker.mp", None):
            tracker = PoseTracker(app_state)
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            tracker.process_frame(frame)  # must not raise
            assert app_state.pose_landmarks == []
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `venv/bin/python3 -m pytest tests/unit/test_pose_tracker.py -v`
Expected: FAIL — `AttributeError: <module 'turtao.vision.pose_tracker'> does not have the attribute 'mp_vision'` (the current module has no such name) or a `TypeError` on `PoseTracker(app_state, model_path=...)` since the current constructor doesn't accept `model_path`.

- [ ] **Step 4: Rewrite `pose_tracker.py`**

Replace the entire contents of `turtao/vision/pose_tracker.py`:

```python
from __future__ import annotations

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
            try:
                self._landmarker.close()
            except Exception:
                pass
```

- [ ] **Step 5: Update `state.py`'s type comment for `pose_landmarks`**

In `turtao/state.py`, the `AppState.__init__` line:

```python
        self.pose_landmarks: list[tuple[int, int]] = []
```

becomes:

```python
        self.pose_landmarks: list[list[tuple[int, int]]] = []  # one list per detected person
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `venv/bin/python3 -m pytest tests/unit/test_pose_tracker.py -v`
Expected: PASS (3 tests)

- [ ] **Step 7: Run full suite**

Run: `venv/bin/python3 -m pytest -q`
Expected: all green (this will surface any other code still assuming the old flat `pose_landmarks` shape — fix forward in this task if so; the only other reader is `tab_recognition.py`, addressed in Task 7)

- [ ] **Step 8: Lint**

Run: `venv/bin/ruff check turtao/vision/pose_tracker.py turtao/state.py`
Expected: clean

- [ ] **Step 9: Commit**

```bash
git add requirements.txt turtao/vision/pose_tracker.py turtao/state.py tests/unit/test_pose_tracker.py
git commit -m "feat: multi-person MediaPipe pose via the Tasks API PoseLandmarker"
```

---

### Task 7: Decouple pose computation from the skeleton-overlay toggle; draw multiple faces/skeletons in the GUI

**Files:**
- Modify: `turtao/core.py:118-143` (`_tracker_wrapper`)
- Modify: `turtao/gui/tab_recognition.py`

**Interfaces:**
- Consumes: `state.pose_landmarks: list[list[tuple[int,int]]]` (Task 6), `state.threat_state.faces: list[FaceDetection]` (Task 2).
- Produces: nothing consumed elsewhere — this is the last consumer in the chain.

- [ ] **Step 1: Decouple pose computation from the display toggle in `core.py`**

In `turtao/core.py`'s `_tracker_wrapper`, find:

```python
                if show_mp:
                    self.pose_tracker.process_frame(frame)
                else:
                    with self.state:
                        self.state.pose_landmarks = []
```

Replace with:

```python
                self.pose_tracker.process_frame(frame)
```

(Pose now always runs while `active` is true — i.e. whenever `mode != Mode.IDLE` — matching YOLO's own condition on the same branch. `show_mp` continues to exist as a field and continues to gate *drawing* the skeleton in the GUI, handled in Step 2 below; it no longer gates whether pose is computed.)

- [ ] **Step 2: Run the full suite to check for fallout**

Run: `venv/bin/python3 -m pytest -q`
Expected: all green. (No existing test asserts on the `show_mp`-gates-computation behavior being removed — this is a behavior change with no test coverage on the `core.py` side, consistent with the rest of `_tracker_wrapper` being untested at the wrapper-loop level; Task 4's precedent of testing extracted per-tick logic doesn't apply here since there's no meaningful pure logic to extract from a two-line change.)

- [ ] **Step 3: Update `tab_recognition.py` to draw multiple faces and multiple skeletons**

In `turtao/gui/tab_recognition.py::_show_frame`, find this block:

```python
            if threat_box is not None:
                l, t, r, b = threat_box
                with self.core.state:
                    name = getattr(self.core.state.threat_state, "name", "")
                
                color = (0, 255, 0) if threat_label.value == "SAFE" else (255, 0, 0)
                cv2.rectangle(rgb, (l, t), (r, b), color, 2)
                
                label_text = name if name else threat_label.value
                cv2.putText(rgb, label_text, (l, max(0, t - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
```

Replace it with:

```python
            with self.core.state:
                faces = list(self.core.state.threat_state.faces)

            for face in faces:
                fl, ft, fr, fb = face.box
                color = (0, 255, 0) if face.label.value == "SAFE" else (255, 0, 0)
                cv2.rectangle(rgb, (fl, ft), (fr, fb), color, 2)
                cv2.putText(
                    rgb, face.name, (fl, max(0, ft - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2,
                )
```

This makes the earlier `threat_box` local (fetched a few lines above at `threat_box = self.core.state.threat_state.box`) unused for drawing — check with:

Run: `grep -n "threat_box" turtao/gui/tab_recognition.py`

If `threat_box` is no longer referenced anywhere else in the method after this change, remove its assignment line too (`threat_box = self.core.state.threat_state.box`) and the now-unused `threat_label` variable only if it's *also* unused elsewhere in the method — check first, since `threat_label` may still legitimately be used elsewhere in `_show_frame` for other purposes; do not remove it if so.

Next, find the pose-skeleton drawing block (the `if show_mp:` section) which currently indexes `pose_lms` as a flat list of points:

```python
            if show_mp:
                # Read 33 dynamic landmarks from state
                with self.core.state:
                    pose_lms = getattr(self.core.state, "pose_landmarks", [])
                
                # Draw MediaPipe Pose connections
                if pose_lms and len(pose_lms) >= 33:
                    pose_connections = [
                        # Torso
                        (11, 12), (11, 23), (12, 24), (23, 24),
                        # Left Arm
                        (11, 13), (13, 15),
                        # Right Arm
                        (12, 14), (14, 16),
                        # Left Leg
                        (23, 25), (25, 27),
                        # Right Leg
                        (24, 26), (26, 28),
                        # Shoulders to head connection
                        (0, 11), (0, 12)
                    ]
                    for start_idx, end_idx in pose_connections:
                        if start_idx < len(pose_lms) and end_idx < len(pose_lms):
                            pt1 = pose_lms[start_idx]
                            pt2 = pose_lms[end_idx]
                            cv2.line(rgb, pt1, pt2, (0, 255, 255), 2)
                    
                    # Only draw joint dots on core nodes (nose, shoulders, elbows, wrists, hips, knees, ankles)
                    core_joints = {0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28}
                    for idx, pt in enumerate(pose_lms):
                        if idx in core_joints:
                            cv2.circle(rgb, pt, 4, (0, 0, 255), -1)
```

Replace it with (same drawing logic, now looped over each detected person's landmark list instead of assuming one flat list):

```python
            if show_mp:
                with self.core.state:
                    all_pose_lms = list(getattr(self.core.state, "pose_landmarks", []))

                pose_connections = [
                    # Torso
                    (11, 12), (11, 23), (12, 24), (23, 24),
                    # Left Arm
                    (11, 13), (13, 15),
                    # Right Arm
                    (12, 14), (14, 16),
                    # Left Leg
                    (23, 25), (25, 27),
                    # Right Leg
                    (24, 26), (26, 28),
                    # Shoulders to head connection
                    (0, 11), (0, 12)
                ]
                core_joints = {0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28}

                for pose_lms in all_pose_lms:
                    if not pose_lms or len(pose_lms) < 33:
                        continue
                    for start_idx, end_idx in pose_connections:
                        if start_idx < len(pose_lms) and end_idx < len(pose_lms):
                            pt1 = pose_lms[start_idx]
                            pt2 = pose_lms[end_idx]
                            cv2.line(rgb, pt1, pt2, (0, 255, 255), 2)
                    for idx, pt in enumerate(pose_lms):
                        if idx in core_joints:
                            cv2.circle(rgb, pt, 4, (0, 0, 255), -1)
```

- [ ] **Step 4: Manual verification (no automated GUI test exists in this repo)**

On the Pi with `python main.py --gui`, with mediapipe installed (Task 6) and two people in frame (one enrolled, one not):
1. Enable GUARD mode and the MediaPipe Pose checkbox.
2. Confirm both people get their own colored face box + name/label (green+name for the enrolled person, red+"Unknown N" for the other).
3. Confirm both people get their own skeleton overlay drawn simultaneously.

- [ ] **Step 5: Lint**

Run: `venv/bin/ruff check turtao/core.py turtao/gui/tab_recognition.py`
Expected: no new findings

- [ ] **Step 6: Commit**

```bash
git add turtao/core.py turtao/gui/tab_recognition.py
git commit -m "feat: draw all detected faces and pose skeletons, not just the first"
```

---

### Task 8: Vendor the pose landmarker model in `install.sh`

**Files:**
- Modify: `install.sh`

**Interfaces:**
- Consumes: nothing.
- Produces: `models/pose_landmarker_lite.task` on disk, matching `DEFAULT_MODEL_PATH` in `turtao/vision/pose_tracker.py` (Task 6).

- [ ] **Step 1: Add the download block**

In `install.sh`, immediately after the existing YOLO model download block (the `if [ ! -f "$MODELS_DIR/yolov8n.onnx" ]; then ... fi` block), add:

```bash
if [ ! -f "$MODELS_DIR/pose_landmarker_lite.task" ]; then
    echo ">>> Downloading MediaPipe pose landmarker model..."
    wget -q "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task" \
        -O "$MODELS_DIR/pose_landmarker_lite.task"
fi
```

- [ ] **Step 2: Manual verification**

Run: `bash -n install.sh`
Expected: no syntax errors (dry-run parse check; do not actually re-run the full installer here since it does system-level `apt-get`/`usermod` operations — confirm the new block's syntax only)

- [ ] **Step 3: Commit**

```bash
git add install.sh
git commit -m "chore: vendor the MediaPipe pose landmarker model in install.sh"
```

---

## Self-Review Notes

- **Spec coverage:** §1 (multi-face) → Task 2. §2 (multi-pose) → Tasks 6-7. §3 (loiter workflow) → Tasks 3-4. §4 (delete-face fix) → Task 5. Model asset / CPU-load risks from the spec's Risks section → Tasks 6 and 8, with the FPS checkpoint called out as manual on-Pi verification (cannot be measured in this dev environment, which has neither a camera nor `mediapipe` installed).
- **Type consistency checked:** `FaceDetection` (Task 1) is used identically in Tasks 2, 3, 4, 7. `LoiterMonitor.update`'s signature (Task 3) matches its only call site in `TurtaoCore._loiter_tick` (Task 4) exactly. `pose_landmarks`'s new `list[list[tuple[int,int]]]` shape (Task 6) matches how Task 4's `pose_present` check and Task 7's drawing loop both consume it.
- **No placeholders:** every step has literal code or an exact command; the two "manual verification" steps (Tasks 5 and 7) are explicit numbered scripts with expected observations, not "test it manually" placeholders — this matches the constraint that GUI code has no automated coverage in this repo.
