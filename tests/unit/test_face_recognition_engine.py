import numpy as np
import pytest
from unittest.mock import patch, MagicMock
from turtao.vision.face_recognition_engine import (
    cosine_similarity,
    FaceRecognitionEngine,
    RESIZE_SCALE,
)
from turtao.state import AppState, ThreatLabel


@pytest.fixture
def engine(tmp_path):
    face_data_dir = tmp_path / "face_data"
    embeddings_dir = face_data_dir / "embeddings"
    embeddings_dir.mkdir(parents=True)
    eng = FaceRecognitionEngine(AppState(), tolerance=0.6)
    eng.load_embeddings(str(embeddings_dir))
    return eng


def _write_embedding(engine, name, pose):
    path = engine._embeddings_dir / f"{name}_{pose:03d}.npy"
    np.save(str(path), np.zeros(128))


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


class TestListFaces:
    def test_empty_when_no_embeddings(self, engine):
        assert engine.list_faces() == []

    def test_groups_multiple_poses_under_one_name(self, engine):
        _write_embedding(engine, "alice", 0)
        _write_embedding(engine, "alice", 1)
        _write_embedding(engine, "alice", 2)
        faces = engine.list_faces()
        assert len(faces) == 1
        assert faces[0].name == "alice"
        assert faces[0].pose_count == 3

    def test_strips_pose_suffix_not_full_stem(self, engine):
        _write_embedding(engine, "bob_smith", 0)
        faces = engine.list_faces()
        assert faces[0].name == "bob_smith"  # only the trailing _NNN is stripped


class TestFaceExists:
    def test_true_when_present(self, engine):
        _write_embedding(engine, "carol", 0)
        assert engine.face_exists("carol") is True

    def test_false_when_absent(self, engine):
        assert engine.face_exists("nobody") is False


class TestDeleteFace:
    def test_removes_all_poses(self, engine):
        _write_embedding(engine, "dave", 0)
        _write_embedding(engine, "dave", 1)
        engine.delete_face("dave")
        assert engine.face_exists("dave") is False
        assert list(engine._embeddings_dir.glob("dave_*.npy")) == []

    def test_raises_value_error_when_absent(self, engine):
        with pytest.raises(ValueError):
            engine.delete_face("nobody")

    def test_does_not_delete_other_face_with_prefix_name(self, engine):
        _write_embedding(engine, "bob", 0)
        _write_embedding(engine, "bob_smith", 0)
        engine.delete_face("bob")
        assert engine.face_exists("bob") is False
        assert engine.face_exists("bob_smith") is True

    def test_removes_profile_entry(self, engine):
        _write_embedding(engine, "frank", 0)
        engine._write_profiles([{"name": "frank", "embeddings": ["frank_000.npy"], "created_at": "2026-01-01T00:00:00"}])
        engine.delete_face("frank")
        profiles = engine._read_profiles()
        assert all(p.get("name") != "frank" for p in profiles)


class TestUnknowns:
    def test_list_unknowns_empty(self, engine):
        assert engine.list_unknowns() == []

    def test_list_unknowns_reads_directory(self, engine):
        unknown_dir = engine._embeddings_dir.parent / "unknowns"
        unknown_dir.mkdir(parents=True)
        (unknown_dir / "unknown_20260101_120000.jpg").write_bytes(b"fake")
        unknowns = engine.list_unknowns()
        assert len(unknowns) == 1
        assert unknowns[0].id == "unknown_20260101_120000"
        assert unknowns[0].first_seen == "2026-01-01T12:00:00"
        assert unknowns[0].cluster_count == 1

    def test_get_unknown_thumb_returns_bytes(self, engine):
        unknown_dir = engine._embeddings_dir.parent / "unknowns"
        unknown_dir.mkdir(parents=True)
        (unknown_dir / "unknown_20260101_120000.jpg").write_bytes(b"fake_jpeg")
        assert engine.get_unknown_thumb("unknown_20260101_120000") == b"fake_jpeg"

    def test_get_unknown_thumb_none_when_absent(self, engine):
        assert engine.get_unknown_thumb("nope") is None

    def test_delete_unknown_removes_file(self, engine):
        unknown_dir = engine._embeddings_dir.parent / "unknowns"
        unknown_dir.mkdir(parents=True)
        (unknown_dir / "unknown_20260101_120000.jpg").write_bytes(b"fake")
        engine.delete_unknown("unknown_20260101_120000")
        assert engine.get_unknown_thumb("unknown_20260101_120000") is None

    def test_delete_unknown_raises_when_absent(self, engine):
        with pytest.raises(ValueError):
            engine.delete_unknown("nope")

    @patch("turtao.vision.face_recognition_engine.face_recognition")
    @patch("turtao.vision.face_recognition_engine.cv2")
    def test_promote_unknown_moves_to_faces(self, mock_cv2, mock_fr, engine):
        unknown_dir = engine._embeddings_dir.parent / "unknowns"
        unknown_dir.mkdir(parents=True)
        (unknown_dir / "unknown_20260101_120000.jpg").write_bytes(b"fake")
        mock_cv2.imread.return_value = np.zeros((100, 100, 3), dtype=np.uint8)
        mock_cv2.cvtColor.return_value = np.zeros((100, 100, 3), dtype=np.uint8)
        mock_fr.face_encodings.return_value = [np.zeros(128)]
        engine.promote_unknown("unknown_20260101_120000", "erin")
        assert engine.face_exists("erin") is True
        assert engine.list_unknowns() == []

    def test_promote_unknown_raises_when_absent(self, engine):
        with pytest.raises(ValueError):
            engine.promote_unknown("nope", "erin")

    def test_promote_unknown_raises_when_no_face_detected(self, engine):
        unknown_dir = engine._embeddings_dir.parent / "unknowns"
        unknown_dir.mkdir(parents=True)
        (unknown_dir / "unknown_20260101_120000.jpg").write_bytes(b"fake")
        with patch("turtao.vision.face_recognition_engine.cv2") as mock_cv2, \
             patch("turtao.vision.face_recognition_engine.face_recognition") as mock_fr:
            mock_cv2.imread.return_value = np.zeros((100, 100, 3), dtype=np.uint8)
            mock_cv2.cvtColor.return_value = np.zeros((100, 100, 3), dtype=np.uint8)
            mock_fr.face_encodings.return_value = []
            with pytest.raises(ValueError):
                engine.promote_unknown("unknown_20260101_120000", "erin")
        # Crop must survive the failed attempt — not silently deleted.
        assert (unknown_dir / "unknown_20260101_120000.jpg").is_file()

    def test_promote_unknown_raises_when_name_already_exists(self, engine):
        _write_embedding(engine, "erin", 0)
        unknown_dir = engine._embeddings_dir.parent / "unknowns"
        unknown_dir.mkdir(parents=True)
        (unknown_dir / "unknown_20260101_120000.jpg").write_bytes(b"fake")
        with pytest.raises(ValueError):
            engine.promote_unknown("unknown_20260101_120000", "erin")

    def test_promote_unknown_writes_profile_entry(self, engine):
        unknown_dir = engine._embeddings_dir.parent / "unknowns"
        unknown_dir.mkdir(parents=True)
        (unknown_dir / "unknown_20260101_120000.jpg").write_bytes(b"fake")
        with patch("turtao.vision.face_recognition_engine.cv2") as mock_cv2, \
             patch("turtao.vision.face_recognition_engine.face_recognition") as mock_fr:
            mock_cv2.imread.return_value = np.zeros((100, 100, 3), dtype=np.uint8)
            mock_cv2.cvtColor.return_value = np.zeros((100, 100, 3), dtype=np.uint8)
            mock_fr.face_encodings.return_value = [np.zeros(128)]
            engine.promote_unknown("unknown_20260101_120000", "grace")
        profiles = engine._read_profiles()
        assert any(p.get("name") == "grace" for p in profiles)


class TestMultiFaceRecognition:
    @patch("turtao.vision.face_recognition_engine.face_recognition")
    @patch("turtao.vision.face_recognition_engine.cv2")
    def test_two_known_faces_both_appear_in_faces_list(
        self, mock_cv2, mock_fr, app_state: AppState
    ):
        mock_fr.face_locations.return_value = [(100, 200, 300, 50), (100, 500, 300, 350)]
        alice_encoding = np.array([1.0, 0.0, 0.0, 0.0])
        bob_encoding = np.array([0.0, 1.0, 0.0, 0.0])
        mock_fr.face_encodings.return_value = [alice_encoding, bob_encoding]
        mock_cv2.resize.return_value = np.zeros((240, 320, 3), dtype=np.uint8)
        mock_cv2.cvtColor.return_value = np.zeros((240, 320, 3), dtype=np.uint8)

        engine = FaceRecognitionEngine(app_state, tolerance=0.1)
        engine._known_embeddings = [alice_encoding, bob_encoding]
        engine._known_names = ["alice", "bob"]
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        engine.process_frame(frame)

        faces = app_state.threat_state.faces
        assert len(faces) == 2
        names = {f.name for f in faces}
        assert names == {"alice", "bob"}
        assert all(f.label == ThreatLabel.SAFE for f in faces)
        assert app_state.threat_label == ThreatLabel.SAFE

    @patch("turtao.vision.face_recognition_engine.face_recognition")
    @patch("turtao.vision.face_recognition_engine.cv2")
    def test_one_known_one_unknown_summary_is_threat(
        self, mock_cv2, mock_fr, app_state: AppState
    ):
        mock_fr.face_locations.return_value = [(100, 200, 300, 50), (100, 500, 300, 350)]
        alice_encoding = np.array([1.0, 0.0, 0.0, 0.0])
        stranger_encoding = np.array([0.0, 0.0, 1.0, 0.0])
        mock_fr.face_encodings.return_value = [alice_encoding, stranger_encoding]
        mock_cv2.resize.return_value = np.zeros((240, 320, 3), dtype=np.uint8)
        mock_cv2.cvtColor.return_value = np.zeros((240, 320, 3), dtype=np.uint8)

        engine = FaceRecognitionEngine(app_state, tolerance=0.1)
        engine._known_embeddings = [alice_encoding]
        engine._known_names = ["alice"]
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        engine.process_frame(frame)

        faces = app_state.threat_state.faces
        assert len(faces) == 2
        labels = {f.name: f.label for f in faces}
        assert labels["alice"] == ThreatLabel.SAFE
        assert any(v == ThreatLabel.THREAT for k, v in labels.items() if k != "alice")
        # Global summary must be THREAT (any-unresolved-face wins)
        assert app_state.threat_label == ThreatLabel.THREAT
        assert app_state.threat_state.name != "alice"

    @patch("turtao.vision.face_recognition_engine.face_recognition")
    @patch("turtao.vision.face_recognition_engine.cv2")
    def test_no_faces_clears_faces_list(
        self, mock_cv2, mock_fr, app_state: AppState
    ):
        mock_fr.face_locations.return_value = []
        mock_fr.face_encodings.return_value = []
        mock_cv2.resize.return_value = np.zeros((240, 320, 3), dtype=np.uint8)
        mock_cv2.cvtColor.return_value = np.zeros((240, 320, 3), dtype=np.uint8)

        engine = FaceRecognitionEngine(app_state)
        engine._frames_since_seen = 10  # force the IDLE-reset branch immediately
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        engine.process_frame(frame)

        assert app_state.threat_state.faces == []
