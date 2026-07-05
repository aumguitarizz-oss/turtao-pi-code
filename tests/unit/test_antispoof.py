import numpy as np
import pytest
from unittest.mock import MagicMock, patch
from turtao.vision.antispoof import AntiSpoofDetector, OVERLAP_THRESHOLD


@pytest.fixture
def detector():
    return AntiSpoofDetector()


def _make_hand_landmarks(x_min, y_min, x_max, y_max, num_landmarks=21):
    mock_landmarks = MagicMock()
    xs = np.linspace(x_min, x_max, num_landmarks)
    ys = np.linspace(y_min, y_max, num_landmarks)
    landmarks = []
    for x, y in zip(xs, ys):
        lm = MagicMock()
        lm.x = x
        lm.y = y
        landmarks.append(lm)
    mock_landmarks.landmark = landmarks
    return mock_landmarks


class TestAntiSpoofDetector:
    def test_detector_init_failure_returns_false_on_check(self, caplog):
        import logging
        caplog.set_level(logging.ERROR)
        with patch(
            "turtao.vision.antispoof.mp.solutions.hands.Hands",
            side_effect=RuntimeError("no mediapipe"),
        ):
            d = AntiSpoofDetector()
            assert d._hands is None
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            result = d.check_occlusion(frame, (0, 0, 100, 100))
            assert result is False

    def test_no_hands_returns_false(self, detector):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = detector.check_occlusion(frame, (0, 0, 100, 100))
        assert result is False

    def test_hand_overlap_above_threshold_returns_true(self, detector):
        face_bbox = (100, 100, 300, 300)
        face_area = (300 - 100) * (300 - 100)
        overlap_ratio = OVERLAP_THRESHOLD + 0.1
        overlap_px = int(np.sqrt(face_area * overlap_ratio))
        hand_center = 200
        half_side = overlap_px // 2
        hx_min = (hand_center - half_side) / 640
        hy_min = (hand_center - half_side) / 480
        hx_max = (hand_center + half_side) / 640
        hy_max = (hand_center + half_side) / 480
        hand_lm = _make_hand_landmarks(hx_min, hy_min, hx_max, hy_max)
        mock_results = MagicMock()
        mock_results.multi_hand_landmarks = [hand_lm]

        with patch.object(
            detector._hands, "process", return_value=mock_results
        ):
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            result = detector.check_occlusion(frame, face_bbox)
            assert result is True

    def test_hand_overlap_below_threshold_returns_false(self, detector):
        face_bbox = (100, 100, 300, 300)
        face_area = (300 - 100) * (300 - 100)
        overlap_ratio = OVERLAP_THRESHOLD - 0.1
        overlap_px = int(np.sqrt(face_area * overlap_ratio))
        hand_center = 200
        half_side = overlap_px // 4
        hx_min = (hand_center - half_side) / 640
        hy_min = (hand_center - half_side) / 480
        hx_max = (hand_center + half_side) / 640
        hy_max = (hand_center + half_side) / 480
        hand_lm = _make_hand_landmarks(hx_min, hy_min, hx_max, hy_max)
        mock_results = MagicMock()
        mock_results.multi_hand_landmarks = [hand_lm]

        with patch.object(
            detector._hands, "process", return_value=mock_results
        ):
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            result = detector.check_occlusion(frame, face_bbox)
            assert result is False

    def test_hand_outside_face_bbox_returns_false(self, detector):
        hand_lm = _make_hand_landmarks(0.0, 0.0, 0.05, 0.05)
        mock_results = MagicMock()
        mock_results.multi_hand_landmarks = [hand_lm]

        with patch.object(
            detector._hands, "process", return_value=mock_results
        ):
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            face_bbox = (200, 200, 400, 400)
            result = detector.check_occlusion(frame, face_bbox)
            assert result is False

    def test_release_sets_hands_to_none(self, detector):
        detector.release()
        assert detector._hands is None

    def test_release_when_already_none_does_not_crash(self):
        d = AntiSpoofDetector()
        d._hands = None
        d.release()
        assert d._hands is None
