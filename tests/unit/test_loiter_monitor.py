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
