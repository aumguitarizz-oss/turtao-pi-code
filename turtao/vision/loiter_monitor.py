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
    last_bbox: tuple[int, int, int, int] | None = None
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
            timer.last_bbox = bbox

            if not pose_present:
                continue

            if self._face_overlaps(bbox, faces):
                timer.first_missing_at = now
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

        # Check recording/alert for timers of persons not in current frame
        for tid, timer in self._timers.items():
            if tid in seen_ids or timer.first_missing_at is None or not pose_present:
                continue
            missing_for = now - timer.first_missing_at

            if missing_for >= FACE_MISSING_RECORD_THRESHOLD and not timer.recorded:
                if frame is not None and timer.last_bbox is not None:
                    record_crop(frame, timer.last_bbox)
                timer.recorded = True

            if missing_for >= FACE_MISSING_ALERT_THRESHOLD and not timer.alerted:
                emit_alert(
                    f"Unrecognized person (tracker #{tid}) unresolved "
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
