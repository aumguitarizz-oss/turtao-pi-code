# Multi-Person Detection + Loiter/Alert Workflow — Design

**Date:** 2026-08-05
**Branch:** fix/face-detection-rebuild (continues on this branch)
**Status:** Approved by user, pending implementation plan

## Problem

Three related gaps found while testing the rebuilt face pipeline:

1. **Single-target detection.** `FaceRecognitionEngine.process_frame` breaks
   after the first face it finds; `PoseTracker` uses MediaPipe's legacy
   `mp.solutions.pose.Pose`, which only ever tracks one body. Both need to
   handle multiple people simultaneously in frame.
2. **No loiter/alert workflow.** There's no logic that reacts to "a person
   is present but no face has been resolved for them" — the feature the
   user wants: record after 0.5s missing, alert the app after 2s missing.
3. **Delete-face bug.** `turtao/gui/tab_faces.py:_delete_face` only rewrites
   `profiles.json`; it never deletes the `.npy` embedding or reloads the
   running engine's in-memory `_known_embeddings`, so deleted faces keep
   matching. Root-caused via grep — the GUI reimplemented deletion instead
   of calling `FaceRecognitionEngine.delete_face`, which already does this
   correctly and is already used by the API's DELETE endpoint.

Also discovered: `mediapipe` is commented out in `requirements.txt`
(`# mediapipe>=0.10`) — it was never installed, which is the literal reason
"mediapipe isn't working." Separately from the multi-person rewrite below,
it needs to actually be installed.

## Non-goals

- No changes to the phone app repo (`turtaoapp`). Confirmed during design:
  the app's enrollment path already shares the Pi's `capture_pose_burst`
  code exactly (same 5-frame burst, same server-side logic already fixed
  in the prior session), and the app's `AppEvent` model (`{id, type,
  message, at}`) already matches `/api/events`'s response shape and is
  already polled by an existing `events_provider.dart` / `event_log.dart`.
  The alert pipe exists end-to-end and is just never fed — no app changes
  needed to receive alerts.
- App UI will not be updated to show multiple simultaneous face boxes.
  Confirmed with user: Pi-side (GUI + alerting) correctness is the goal for
  this pass; the app keeps its current single-summary threat display.
- No change to `_calculate_aim`/servo targeting logic beyond picking a
  reasonable single target among multiple faces (see below) — multi-person
  *aiming* is out of scope.

## Design

### 1. Multi-person face recognition

`turtao/vision/face_recognition_engine.py::process_frame`: remove the
`break` after the first face. Loop over every `(location, encoding)` pair,
computing each face's match independently (known / session-unknown / new
unknown), exactly as today's single-face logic already does per-face — the
existing logic is already written per-encoding, it's just truncated early.

**Data model change** (`turtao/state.py`):

```python
@dataclass
class FaceDetection:
    box: tuple[int, int, int, int]
    name: str
    label: ThreatLabel   # SAFE or THREAT for this specific face
    confidence: float

@dataclass
class ThreatState:
    active: bool = False
    face_crop: bytes | None = None
    confidence: float = 0.0       # summary: worst-case face's confidence
    timestamp: float | None = None
    box: tuple[int, int, int, int] | None = None   # summary: worst-case face's box
    landmarks: list[tuple[int, int]] = field(default_factory=list)
    name: str = ""                 # summary: worst-case face's name
    faces: list[FaceDetection] = field(default_factory=list)  # NEW: all faces this frame
```

The existing top-level `box`/`name`/`confidence` fields stay and keep
meaning "the most severe face this frame" (THREAT beats SAFE; among ties,
first found) — this is what `ws_status.py` already serializes for the app,
so the app's contract is unchanged. `faces` is additive.

**Global `state.threat_label`:** THREAT if any face this frame is THREAT,
else SAFE if any face is SAFE, else IDLE if no faces at all — same
priority the single-face code already implied, generalized.

**Servo aim:** keep today's behavior — aim at the summary/worst-case
face's box. Not a regression (today only ever has one face to aim at
anyway).

**`tab_recognition.py::_show_frame`:** loop over `threat_state.faces` and
draw one rectangle+label per entry, instead of the single `threat_box`.

### 2. Multi-person MediaPipe pose

Replace the legacy single-person `mp.solutions.pose.Pose` with the Tasks
API's `PoseLandmarker` (`mediapipe.tasks.python.vision`), constructed with
`num_poses=4` (reasonable cap for a home robot's field of view) and
`RunningMode.VIDEO` or `LIVE_STREAM`. This needs a `.task` model file
(e.g. `pose_landmarker_lite.task`) downloaded once into `models/`, the same
way `models/yolov8n.onnx` is already vendored.

**Data model change:** `state.pose_landmarks` becomes
`list[list[tuple[int, int]]]` — one landmark list per detected person,
instead of a single flat list. `tab_recognition.py`'s skeleton-drawing loop
updates to draw one skeleton per entry.

**Always-on computation:** currently pose is only computed when the
cosmetic skeleton-overlay checkbox (`show_mediapipe`, default off) is on.
Decouple: pose runs whenever tracking is active (`mode != IDLE`, same
condition YOLO already uses) regardless of the display toggle, so the
loiter workflow (below) always has fresh pose data. The checkbox continues
to control only whether the skeleton is *drawn*.

**Verification caveat:** `mediapipe` isn't installed in the dev
environment used for this session (confirmed via `pip show` /
`ModuleNotFoundError`), so this cannot be exercised end-to-end locally.
Install it (uncomment `requirements.txt`, `pip install`) and validate the
Tasks API shape on the Pi as part of implementation.

### 3. Loiter / alert workflow

New `turtao/vision/loiter_monitor.py::LoiterMonitor`, driven by a new
`_loiter_wrapper` daemon thread in `core.py` (9th thread), polling every
~0.1s. Reads `state.latest_persons` (YOLO/ByteTrack, has stable
`tracker_id`), `state.pose_landmarks` (now per-person), and
`state.threat_state.faces` (now per-face).

Per-person correlation (now genuinely per-`tracker_id`, since both face and
pose are multi-person as of §1/§2 — this removes the "crowd false-trigger"
limitation flagged during design):

- For each tracked person: find the nearest pose skeleton (by bbox/centroid
  proximity) and check whether any `FaceDetection` box overlaps that
  person's face region (top third of bbox — the same heuristic
  `person_tracker.py::_check_phone_spoof` already uses for phone/face
  proximity).
- **Presence confirmed** = a nearby pose skeleton exists for this person
  (YOLO ∧ MediaPipe agree, per user's explicit ask — reduces false
  triggers from YOLO-only misdetections).
- No overlapping face for **≥0.5s** while presence is confirmed → *record*:
  save a crop of the person's bbox from `state.latest_frame` to
  `face_data/unknowns/unknown_<timestamp>.jpg` — deliberately reusing the
  exact existing filename convention so it appears in the current GUI
  Unknowns tab / `/api/faces/unknowns` / promote-to-known flow with zero
  new code there.
- Still no face at **≥2.0s** → *alert*: append one
  `Event(type="unidentified_person", message=f"Unrecognized person (tracker
  #{tracker_id}) unresolved for 2s+", at=<ISO ts>)` to `state.events` (via
  `AppState.event_counter` for the id, same pattern the dataclass already
  implies). The app's existing event log picks it up on its next
  `/api/events` poll — no app changes.
- State resets per `tracker_id` when: a face resolves for them again, or
  they drop out of `latest_persons` for a grace period (~1s, ByteTrack IDs
  are monotonic and not reused, so a plain drop is safe).
- Record/alert each fire **once** per "missing" episode (not repeatedly
  every poll cycle) via per-`tracker_id` `recorded`/`alerted` flags,
  cleared on reset.

### 4. Delete-face bug fix

`turtao/gui/tab_faces.py::_delete_face`: replace the hand-rolled
`profiles.json` rewrite with `self.core.face_engine.delete_face(name)`
(wrapped in the same try/except ValueError pattern used elsewhere for
"face not found"). This is the exact method the API's DELETE endpoint
already calls correctly — deletes the `.npy` file(s), updates
`profiles.json`, and reloads `_known_embeddings`/`_known_names` so the
running recognition loop stops matching the deleted face immediately.

## Testing

- Unit: `FaceRecognitionEngine.process_frame` with 2-3 mocked faces per
  frame (mix of known/unknown) — assert `threat_state.faces` has the right
  count/labels and the summary fields reflect the worst-case face.
- Unit: `LoiterMonitor` as a pure function of synthetic state snapshots +
  a fake clock — assert record fires once at 0.5s, alert fires once at
  2.0s, both reset when a face resolves or the person disappears. This is
  the highest-value test since the timing logic is the newest, most
  bug-prone piece.
- Unit: `tab_faces.py` delete flow — assert `.npy` file removed and
  `face_exists()` returns False afterward (regression test for the bug).
- Existing 230-test suite must stay green throughout.
- On-Pi manual verification (can't be done in this dev environment):
  mediapipe actually installs and `PoseLandmarker` produces multi-person
  landmarks; two people in frame simultaneously — one enrolled, one
  not — both get correct independent face boxes; loitering unrecognized
  person triggers a recorded unknown crop at 0.5s and an app-visible event
  at 2s; deleting a face via the GUI stops it being recognized on the very
  next frame.

## Risks

- **CPU load** — MediaPipe now always runs during GUARD/PATROL (not just
  when the overlay is toggled on), and multi-pose (`num_poses=4`) costs
  more than single-pose. Combined with YOLO + face recognition already
  running, this is the biggest risk to real-time performance on Pi
  hardware. Mitigate by keeping `model_complexity=0`/lite model, and
  treat on-Pi FPS measurement as a hard checkpoint before calling this
  done — if it's too slow, the fallback is capping `num_poses` lower or
  reducing loiter-monitor poll frequency, not reverting to single-person.
- **New model asset** — the `.task` pose model file needs to be vendored
  (like `yolov8n.onnx`) or downloaded by `install.sh`; needs a decision on
  which lite/full variant and where it's fetched from.
- **`mediapipe` dependency itself** — commented out in `requirements.txt`
  for reasons not documented in this repo's history. Uncommenting and
  installing it needs to actually succeed on Raspberry Pi ARM hardware,
  which isn't guaranteed for every mediapipe release — verify wheel
  availability for the Pi's Python/OS combination before relying on it.
