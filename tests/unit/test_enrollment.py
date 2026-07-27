import numpy as np
import pytest
from unittest.mock import patch
from turtao.vision.enrollment import EnrollmentManager


@pytest.fixture
def manager(tmp_path):
    return EnrollmentManager(tmp_path)


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
