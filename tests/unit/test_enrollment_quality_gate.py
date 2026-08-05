import numpy as np
import pytest
import cv2
from turtao.vision.enrollment import EnrollmentManager, MIN_FACE_AREA_RATIO


class TestCheckQuality:
    @pytest.fixture
    def manager(self, tmp_path):
        return EnrollmentManager(tmp_path)

    def _make_frame(self, height=480, width=640, brightness=128):
        frame = np.full((height, width, 3), brightness, dtype=np.int16)
        noise = np.random.randint(-30, 30, (height, width, 3), dtype=np.int16)
        frame = np.clip(frame + noise, 0, 255).astype(np.uint8)
        return frame

    def test_too_far_face_too_small(self, manager):
        frame = self._make_frame(480, 640)
        top, right, bottom, left = 200, 210, 210, 200
        result = manager.check_quality(frame, (top, right, bottom, left))
        assert result == "too_far"

    def test_blurry_low_laplacian_variance(self, manager):
        frame = np.zeros((480, 640, 3), dtype=np.uint8) # no noise = blurry
        top, right, bottom, left = 100, 300, 300, 100
        result = manager.check_quality(frame, (top, right, bottom, left))
        assert result == "blurry"

    def test_too_dark(self, manager):
        frame = self._make_frame(brightness=10)
        top, right, bottom, left = 100, 500, 400, 140
        result = manager.check_quality(frame, (top, right, bottom, left))
        assert result == "too_dark"

    def test_too_bright(self, manager):
        frame = self._make_frame(brightness=250)
        top, right, bottom, left = 100, 500, 400, 140
        result = manager.check_quality(frame, (top, right, bottom, left))
        assert result == "too_bright"

    def test_passes_all_checks_returns_none(self, manager):
        frame = self._make_frame(brightness=128)
        top, right, bottom, left = 100, 540, 380, 100
        result = manager.check_quality(frame, (top, right, bottom, left))
        assert result is None

    def test_edge_face_area_ratio(self, manager):
        h, w = 480, 640
        frame = self._make_frame(h, w, brightness=128)
        min_area = h * w * MIN_FACE_AREA_RATIO
        side = int(np.sqrt(min_area))
        top, bottom = 100, 100 + side
        left, right = 100, 100 + side
        result = manager.check_quality(frame, (top, right, bottom, left))
        assert result is not None

    def test_large_face_at_boundary(self, manager):
        h, w = 480, 640
        frame = self._make_frame(h, w, brightness=128)
        large_margin = 50
        top, right, bottom, left = (
            large_margin,
            w - large_margin,
            h - large_margin,
            large_margin,
        )
        result = manager.check_quality(frame, (top, right, bottom, left))
        assert result is None
