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
