import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest
from unittest.mock import patch
from turtao.vision.enrollment import EnrollmentManager


@pytest.fixture
def manager(tmp_path):
    return EnrollmentManager(tmp_path)


class TestStrictFaceScanToggle:
    """settings.strict_face_scan: when off, a merely low-quality scan
    (blurry/too_dark/etc) is no longer rejected — but a face still has to
    be detected and encodable regardless."""

    @patch("turtao.vision.enrollment.face_recognition")
    @patch("turtao.vision.enrollment.cv2")
    def test_blurry_frame_rejected_when_strict(self, mock_cv2, mock_fr, tmp_path):
        mock_fr.face_locations.return_value = [(100, 300, 300, 100)]
        mock_fr.face_encodings.return_value = [np.zeros(128)]
        mock_cv2.cvtColor.return_value = np.zeros((480, 640, 3), dtype=np.uint8)

        manager = EnrollmentManager(tmp_path, SimpleNamespace(strict_face_scan=True))
        with patch.object(manager, "check_quality", return_value="blurry"):
            result = manager._process_single_frame(np.zeros((480, 640, 3), dtype=np.uint8))

        assert result["ok"] is False
        assert result["issue"] == "blurry"

    @patch("turtao.vision.enrollment.face_recognition")
    @patch("turtao.vision.enrollment.cv2")
    def test_blurry_frame_accepted_when_not_strict(self, mock_cv2, mock_fr, tmp_path):
        mock_fr.face_locations.return_value = [(100, 300, 300, 100)]
        mock_fr.face_encodings.return_value = [np.zeros(128)]
        mock_cv2.cvtColor.return_value = np.zeros((480, 640, 3), dtype=np.uint8)

        manager = EnrollmentManager(tmp_path, SimpleNamespace(strict_face_scan=False))
        with patch.object(manager, "check_quality", return_value="blurry"):
            result = manager._process_single_frame(np.zeros((480, 640, 3), dtype=np.uint8))

        assert result["ok"] is True

    @patch("turtao.vision.enrollment.face_recognition")
    @patch("turtao.vision.enrollment.cv2")
    def test_no_face_still_rejected_when_not_strict(self, mock_cv2, mock_fr, tmp_path):
        # Disabling strict mode skips the *quality* gate only — it must not
        # bypass the more fundamental "was a face even detected" check.
        mock_fr.face_locations.return_value = []
        mock_cv2.cvtColor.return_value = np.zeros((480, 640, 3), dtype=np.uint8)

        manager = EnrollmentManager(tmp_path, SimpleNamespace(strict_face_scan=False))
        result = manager._process_single_frame(np.zeros((480, 640, 3), dtype=np.uint8))

        assert result["ok"] is False
        assert result["issue"] == "no_face"


class TestCollectBurstSamplesDistinctFrames:
    def test_burst_calls_get_frame_five_times(self, manager):
        manager.start_enrollment("alice")
        call_count = {"n": 0}

        def get_frame():
            call_count["n"] += 1
            frame = np.zeros((100, 100, 3), dtype=np.uint8)
            frame[:, :, 0] = call_count["n"]  # distinct content per call
            return frame

        with patch.object(manager, "_process_single_frame") as mock_process:
            mock_process.side_effect = lambda f: {
                "ok": True, "issue": None, "frame": f, "embedding": np.zeros(128)
            }
            manager.capture_pose(get_frame)

        assert call_count["n"] == 5
        seen_frames = [c.args[0] for c in mock_process.call_args_list]
        first_pixel_values = {f[0, 0, 0] for f in seen_frames}
        assert len(first_pixel_values) == 5


class TestCapturePoseBurst:
    def test_processes_supplied_frames_without_sleeping(self, manager):
        manager.start_enrollment("alice")
        frames = [np.zeros((100, 100, 3), dtype=np.uint8) for _ in range(5)]

        with patch.object(manager, "_process_single_frame") as mock_process:
            mock_process.side_effect = lambda f: {
                "ok": True, "issue": None, "frame": f, "embedding": np.zeros(128)
            }
            start = time.monotonic()
            result = manager.capture_pose_burst(frames)
            elapsed = time.monotonic() - start

        assert mock_process.call_count == 5
        assert elapsed < 0.05  # no BURST_DELAY_MS sleeps between supplied frames
        assert result["status"] in ("next_pose", "complete")

    def test_returns_error_when_no_active_enrollment(self, manager):
        frames = [np.zeros((10, 10, 3), dtype=np.uint8)]
        result = manager.capture_pose_burst(frames)
        assert result["status"] == "error"


class TestStartCaptureBurst:
    def test_sets_is_processing_during_and_clears_after(self, manager):
        manager.start_enrollment("bob")
        frames = [np.zeros((100, 100, 3), dtype=np.uint8) for _ in range(5)]

        release = threading.Event()

        def slow_process(f):
            release.wait(timeout=2)
            return {"ok": True, "issue": None, "frame": f, "embedding": np.zeros(128)}

        with patch.object(manager, "_process_single_frame", side_effect=slow_process):
            assert manager.is_processing is False
            manager.try_begin_processing()
            manager.start_capture_burst(frames)
            time.sleep(0.05)
            assert manager.is_processing is True
            release.set()
            for _ in range(40):
                if not manager.is_processing:
                    break
                time.sleep(0.05)
            assert manager.is_processing is False

    def test_calls_on_complete_when_enrollment_finishes(self, manager):
        manager.start_enrollment("carol")
        frames = [np.zeros((100, 100, 3), dtype=np.uint8) for _ in range(5)]
        with patch.object(manager, "_process_single_frame") as mock_process:
            mock_process.side_effect = lambda f: {
                "ok": True, "issue": None, "frame": f, "embedding": np.zeros(128)
            }
            for _ in range(4):
                manager.capture_pose_burst(frames)
            completed = threading.Event()
            manager.start_capture_burst(frames, on_complete=completed.set)
            assert completed.wait(timeout=2)

    def test_does_not_call_on_complete_when_pose_advances_only(self, manager):
        manager.start_enrollment("dave")
        frames = [np.zeros((100, 100, 3), dtype=np.uint8) for _ in range(5)]
        completed = threading.Event()

        with patch.object(manager, "_process_single_frame") as mock_process:
            mock_process.side_effect = lambda f: {
                "ok": True, "issue": None, "frame": f, "embedding": np.zeros(128)
            }
            manager.start_capture_burst(frames, on_complete=completed.set)
            for _ in range(40):
                if not manager.is_processing:
                    break
                time.sleep(0.05)
        assert not completed.is_set()

    def test_worker_exception_clears_processing_and_sets_quality_issue(self, manager):
        manager.start_enrollment("erin")
        frames = [np.zeros((100, 100, 3), dtype=np.uint8) for _ in range(5)]

        with patch.object(manager, "_process_single_frame", side_effect=RuntimeError("boom")):
            manager.try_begin_processing()
            manager.start_capture_burst(frames)
            for _ in range(40):
                if not manager.is_processing:
                    break
                time.sleep(0.05)
        assert manager.is_processing is False
        status = manager.get_status()
        assert status.get("quality_issue")


class TestTryBeginProcessing:
    def test_returns_true_once_then_false_until_released(self, manager):
        assert manager.try_begin_processing() is True
        assert manager.try_begin_processing() is False
        manager.release_processing()
        assert manager.try_begin_processing() is True
