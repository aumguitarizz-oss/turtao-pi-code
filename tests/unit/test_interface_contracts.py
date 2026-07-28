"""
Asserts the real classes expose every attribute/method the Flask routes
call on them. This is the test that would have caught the entire class of
bug where routes were written against test-only mock shapes that had
silently diverged from the real classes (see design spec §1).
"""
from turtao.vision.face_recognition_engine import FaceRecognitionEngine
from turtao.vision.enrollment import EnrollmentManager
from turtao.vision.camera import Camera
from turtao.hardware.interfaces import CameraInterface
from turtao.audio.tts import TTSManager


REQUIRED_FACE_ENGINE_METHODS = [
    "list_faces", "list_unknowns", "get_thumb", "get_unknown_thumb",
    "face_exists", "promote_unknown", "delete_face", "delete_unknown",
    "load_embeddings", "process_frame",
]

REQUIRED_ENROLLMENT_METHODS = [
    "start_enrollment", "capture_pose", "cancel_enrollment", "get_status",
    "is_processing", "start_capture_burst", "try_begin_processing",
    "release_processing", "capture_pose_burst",
]

REQUIRED_CAMERA_METHODS = ["read", "release", "get_latest_frame"]

REQUIRED_TTS_METHODS = ["speak", "play_file"]


class TestFaceRecognitionEngineInterface:
    def test_has_all_methods_routes_call(self):
        for method in REQUIRED_FACE_ENGINE_METHODS:
            assert hasattr(FaceRecognitionEngine, method), (
                f"FaceRecognitionEngine is missing '{method}', "
                f"which turtao/api/routes_faces.py calls"
            )


class TestEnrollmentManagerInterface:
    def test_has_all_methods_routes_call(self):
        for method in REQUIRED_ENROLLMENT_METHODS:
            assert hasattr(EnrollmentManager, method), (
                f"EnrollmentManager is missing '{method}', "
                f"which turtao/api/routes_faces.py calls"
            )


class TestCameraInterface:
    def test_has_all_methods_routes_call(self):
        for method in REQUIRED_CAMERA_METHODS:
            assert hasattr(Camera, method), (
                f"Camera is missing '{method}', "
                f"which turtao/api/routes_camera.py calls"
            )

    def test_camera_interface_declares_get_latest_frame(self):
        assert hasattr(CameraInterface, "get_latest_frame")


class TestTTSManagerInterface:
    def test_has_all_methods_routes_call(self):
        for method in REQUIRED_TTS_METHODS:
            assert hasattr(TTSManager, method), (
                f"TTSManager is missing '{method}', which turtao/api/routes_audio.py calls"
            )
