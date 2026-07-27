import numpy as np
import pytest
from unittest.mock import MagicMock, patch
from turtao.vision.person_tracker import PersonTracker


class TestPersonTracker:
    def test_inactive_returns_last_persons(self):
        tracker = PersonTracker.__new__(PersonTracker)
        tracker._session = MagicMock()
        tracker._tracker = MagicMock()
        tracker._active = False
        tracker._frame_count = 0
        tracker._last_persons = [{"bbox": (0, 0, 10, 10), "confidence": 0.9, "tracker_id": 1}]

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = tracker.process_frame(frame)
        assert result == tracker._last_persons

    def test_no_session_returns_last_persons(self):
        tracker = PersonTracker.__new__(PersonTracker)
        tracker._session = None
        tracker._tracker = MagicMock()
        tracker._active = True
        tracker._frame_count = 0
        tracker._last_persons = []

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = tracker.process_frame(frame)
        assert result == []

    def test_init_failure_sets_session_to_none(self, caplog):
        import logging
        caplog.set_level(logging.ERROR)
        with patch(
            "turtao.vision.person_tracker.onnxruntime.InferenceSession",
            side_effect=RuntimeError("model not found"),
        ):
            tracker = PersonTracker(model_path="/nonexistent/model.onnx")
            assert tracker._session is None
            assert "Failed to load ONNX model" in caplog.text

    def test_set_active_toggles(self):
        tracker = PersonTracker.__new__(PersonTracker)
        tracker._active = False
        tracker.set_active(True)
        assert tracker._active is True
        tracker.set_active(False)
        assert tracker._active is False

    def test_skips_frames_when_not_on_interval(self):
        tracker = PersonTracker.__new__(PersonTracker)
        tracker._session = MagicMock()
        tracker._tracker = MagicMock()
        tracker._active = True
        tracker._frame_count = 1
        tracker._last_persons = []

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        tracker.process_frame(frame)
        assert tracker._frame_count == 2

    @patch("turtao.vision.person_tracker.PersonTracker._infer")
    def test_correct_frame_count_increment(self, mock_infer):
        tracker = PersonTracker.__new__(PersonTracker)
        tracker._session = MagicMock()
        tracker._tracker = MagicMock()
        tracker._tracker.update_with_detections.return_value = []
        mock_infer.return_value = MagicMock()
        tracker._active = True
        tracker._frame_count = 0
        tracker._last_persons = []

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        for i in range(5):
            tracker.process_frame(frame)
        assert tracker._frame_count == 5
