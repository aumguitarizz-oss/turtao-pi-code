from pathlib import Path

import numpy as np
import pytest

from turtao.config import AppConfig, Settings
from turtao.core import TurtaoCore
from turtao.hardware.mocks import MockCamera, MockSerialLink
from turtao.state import AppState, Mode, ThreatLabel


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
                {
                    "bbox": (100, 100, 200, 300),
                    "tracker_id": 1,
                    "class_name": "Person",
                    "confidence": 0.9,
                }
            ]
            core.state.pose_landmarks = [[(150, 150)]]  # non-empty => pose_present
            core.state.threat_state.faces = []  # no resolved face
            core.state.latest_frame = np.zeros((480, 640, 3), dtype=np.uint8)

        core._loiter_tick(now=1000.0)
        core._loiter_tick(now=1002.5)  # past both 0.5s and 2.0s thresholds

        assert len(core.state.events) == 1
        assert "tracker #1" in core.state.events[0].message


class TestFaceRecognitionWrapperFrameGap:
    def test_transient_frame_gap_does_not_reset_threat_label(self, core, monkeypatch):
        """Regression: a momentary empty frame_queue (camera/producer hiccup,
        not a real "nothing resolved" signal) must not instantly flip
        threat_label to IDLE while threat_state.box/name from the last
        resolved detection are still populated — that desync made an
        unresolved THREAT render as a green/safe box in the app, since the
        label (not the box) is what the app colors by. Only Mode.IDLE or
        process_frame()'s own persistence-limit clear should reset label and
        box/name together."""
        with core.state:
            core.state.mode = Mode.PATROL
            core.state.frame_queue.clear()  # no frame available this tick
            core.state.threat_label = ThreatLabel.THREAT
            core.state.threat_state.box = (10, 10, 50, 50)
            core.state.threat_state.name = "Unknown 1"

        monkeypatch.setattr(
            "turtao.core.time.sleep", lambda *_: core.state.stop_event.set()
        )

        core._face_recognition_wrapper()

        assert core.state.threat_label == ThreatLabel.THREAT
        assert core.state.threat_state.box == (10, 10, 50, 50)
        assert core.state.threat_state.name == "Unknown 1"
