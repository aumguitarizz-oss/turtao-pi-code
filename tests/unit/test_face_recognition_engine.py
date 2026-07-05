import numpy as np
import pytest
from unittest.mock import patch, MagicMock
from turtao.vision.face_recognition_engine import (
    cosine_similarity,
    FaceRecognitionEngine,
    RESIZE_SCALE,
)
from turtao.state import AppState, ThreatLabel


class TestCosineSimilarity:
    def test_identical_vectors_returns_one(self):
        a = np.array([1.0, 2.0, 3.0])
        assert cosine_similarity(a, a) == pytest.approx(1.0)

    def test_orthogonal_vectors_returns_zero(self):
        a = np.array([1.0, 0.0, 0.0])
        b = np.array([0.0, 1.0, 0.0])
        assert cosine_similarity(a, b) == pytest.approx(0.0, abs=1e-10)

    def test_opposite_vectors_returns_negative_one(self):
        a = np.array([1.0, 0.0, 0.0])
        b = np.array([-1.0, 0.0, 0.0])
        assert cosine_similarity(a, b) == pytest.approx(-1.0)

    def test_partial_similarity_between_zero_and_one(self):
        a = np.array([1.0, 0.0])
        b = np.array([0.707, 0.707])
        sim = cosine_similarity(a, b)
        assert 0.0 < sim < 1.0

    def test_same_person_above_tolerance(self):
        tolerance = 0.52
        known = np.array([0.5, 0.5, 0.5, 0.5])
        test = np.array([0.5, 0.5, 0.5, 0.5])
        sim = cosine_similarity(known, test)
        assert sim >= tolerance

    def test_different_person_below_tolerance(self):
        tolerance = 0.52
        known = np.array([1.0, 0.0, 0.0, 0.0])
        test = np.array([0.0, 1.0, 0.0, 0.0])
        sim = cosine_similarity(known, test)
        assert sim < tolerance

    def test_zero_vector_does_not_crash(self):
        a = np.array([0.0, 0.0, 0.0])
        b = np.array([1.0, 0.0, 0.0])
        sim = cosine_similarity(a, b)
        assert isinstance(sim, float)


class TestFaceRecognitionEngine:
    @patch("turtao.vision.face_recognition_engine.face_recognition")
    @patch("turtao.vision.face_recognition_engine.cv2")
    def test_no_faces_sets_threat_idle(
        self, mock_cv2, mock_fr, app_state: AppState
    ):
        mock_fr.face_locations.return_value = []
        mock_fr.face_encodings.return_value = []
        mock_cv2.resize.return_value = np.zeros((240, 320, 3), dtype=np.uint8)
        mock_cv2.cvtColor.return_value = np.zeros((240, 320, 3), dtype=np.uint8)

        engine = FaceRecognitionEngine(app_state)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        engine.process_frame(frame)
        assert app_state.threat_label == ThreatLabel.IDLE

    @patch("turtao.vision.face_recognition_engine.face_recognition")
    @patch("turtao.vision.face_recognition_engine.cv2")
    def test_known_face_sets_threat_safe(
        self, mock_cv2, mock_fr, app_state: AppState
    ):
        mock_fr.face_locations.return_value = [(100, 200, 300, 50)]
        encoding = np.array([0.5, 0.5, 0.5, 0.5])
        mock_fr.face_encodings.return_value = [encoding]
        mock_cv2.resize.return_value = np.zeros((240, 320, 3), dtype=np.uint8)
        mock_cv2.cvtColor.return_value = np.zeros((240, 320, 3), dtype=np.uint8)

        engine = FaceRecognitionEngine(app_state, tolerance=0.52)
        engine._known_embeddings = [encoding]
        engine._known_names = ["alice"]
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        engine.process_frame(frame)
        assert app_state.threat_label == ThreatLabel.SAFE

    @patch("turtao.vision.face_recognition_engine.face_recognition")
    @patch("turtao.vision.face_recognition_engine.cv2")
    def test_unknown_face_sets_threat_threat(
        self, mock_cv2, mock_fr, app_state: AppState
    ):
        mock_fr.face_locations.return_value = [(100, 200, 300, 50)]
        known_encoding = np.array([1.0, 0.0, 0.0, 0.0])
        test_encoding = np.array([0.0, 1.0, 0.0, 0.0])
        mock_fr.face_encodings.return_value = [test_encoding]
        mock_cv2.resize.return_value = np.zeros((240, 320, 3), dtype=np.uint8)
        mock_cv2.cvtColor.return_value = np.zeros((240, 320, 3), dtype=np.uint8)

        engine = FaceRecognitionEngine(app_state, tolerance=0.52)
        engine._known_embeddings = [known_encoding]
        engine._known_names = ["alice"]
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        engine.process_frame(frame)
        assert app_state.threat_label == ThreatLabel.THREAT

    @patch("turtao.vision.face_recognition_engine.face_recognition")
    @patch("turtao.vision.face_recognition_engine.cv2")
    def test_no_known_embeddings_treats_as_threat(
        self, mock_cv2, mock_fr, app_state: AppState
    ):
        mock_fr.face_locations.return_value = [(100, 200, 300, 50)]
        test_encoding = np.array([0.5, 0.5, 0.5, 0.5])
        mock_fr.face_encodings.return_value = [test_encoding]
        mock_cv2.resize.return_value = np.zeros((240, 320, 3), dtype=np.uint8)
        mock_cv2.cvtColor.return_value = np.zeros((240, 320, 3), dtype=np.uint8)

        engine = FaceRecognitionEngine(app_state)
        engine._known_embeddings = []
        engine._known_names = []
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        engine.process_frame(frame)
        assert app_state.threat_label == ThreatLabel.THREAT

    def test_load_embeddings_missing_directory_logs_warning(
        self, app_state: AppState, caplog
    ):
        import logging
        caplog.set_level(logging.WARNING)
        engine = FaceRecognitionEngine(app_state)
        engine.load_embeddings("/nonexistent/path")
        assert "Embedding directory not found" in caplog.text
        assert engine._known_embeddings == []
