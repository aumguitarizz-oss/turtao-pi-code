import copy
import pytest
from unittest.mock import MagicMock
from flask import Flask
from flask_cors import CORS
from flask_sock import Sock
from turtao.hardware.mocks import MockSerialLink
from turtao.state import Mode, ThreatState, BatteryData, SensorData


class MockState:
    """Test double for AppState. Mirrors the real nested shape (state.py)
    by composing the actual dataclasses, so route/broadcaster code that
    reads state.threat_state / state.battery / state.sensor_data works
    identically against this mock and the real AppState.
    """

    def __init__(self):
        self.mode = Mode.IDLE
        self.connected = False
        self.threat_state = ThreatState()
        self.battery = BatteryData()
        self.sensor_data = SensorData()
        self.sensor_data.ble_devices = [
            {"id": "dev1", "name": "JBL Go 3", "rssi": -45, "owner": True},
            {"id": "dev2", "name": "Pixel 7", "rssi": -60, "owner": False},
        ]
        self.heading = 0
        self.latency_ms = 0
        self.speed = 1.0
        self.event_log = []
        self.map_grid = []
        self.map_trail = []
        self.latest_frame = "fake_frame"

    def acquire(self):
        pass

    def release(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


class MockSettings:
    def __init__(self):
        self.hostname = ""
        self.ble_proximity_enabled = True
        self.phone_registration = ""
        self.tts_event_toggles = {"threat": True, "low_battery": True, "intruder": True}
        self.intercom_volume = 0.7
        self.face_tolerance = 0.52
        self.anti_spoof_enabled = True
        self.speed = 0.8
        self.safe_mode = False
        self.auto_flashbang = False
        self.stealth_mode = False
        self.notifications = {
            "threat": True, "gas_danger": True, "low_battery": True,
            "tamper": True, "connection_lost": True,
        }

    def asdict(self):
        return {
            "hostname": self.hostname,
            "ble_proximity_enabled": self.ble_proximity_enabled,
            "phone_registration": self.phone_registration,
            "tts_event_toggles": self.tts_event_toggles,
            "intercom_volume": self.intercom_volume,
            "face_tolerance": self.face_tolerance,
            "anti_spoof_enabled": self.anti_spoof_enabled,
            "speed": self.speed,
            "safe_mode": self.safe_mode,
            "auto_flashbang": self.auto_flashbang,
            "stealth_mode": self.stealth_mode,
            "notifications": self.notifications,
        }

    def save(self):
        pass

    def shallow_merge(self, partial):
        new = copy.deepcopy(self)
        for key, value in partial.items():
            current = getattr(new, key, None)
            if isinstance(current, dict) and isinstance(value, dict):
                current.update(value)
            else:
                setattr(new, key, value)
        return new


class MockEnrollment:
    def __init__(self):
        self._active_name = None
        self._pose_index = 0
        self._poses_total = 5
        self._quality_issue = ""
        self._fail_next_capture = False

    def start_enrollment(self, name):
        self._active_name = name
        self._pose_index = 0
        self._quality_issue = ""
        return {"status": "ok", "pose": 1, "total_poses": self._poses_total}

    def capture_pose(self, frame):
        if self._active_name is None:
            return {"status": "error", "message": "No active enrollment"}
        if self._fail_next_capture:
            self._fail_next_capture = False
            self._quality_issue = "blurry"
            return {"status": "retry", "guidance": "blurry", "pose": self._pose_index + 1, "total_poses": self._poses_total}
        self._quality_issue = ""
        self._pose_index += 1
        if self._pose_index >= self._poses_total:
            name = self._active_name
            self._active_name = None
            return {"status": "complete", "name": name}
        return {"status": "next_pose", "pose": self._pose_index + 1, "total_poses": self._poses_total}

    def cancel_enrollment(self):
        self._active_name = None
        self._pose_index = 0
        return {"status": "ok"}

    def get_status(self):
        if self._active_name is None:
            return {"status": "idle"}
        return {
            "status": "active",
            "pose": self._pose_index + 1,
            "total_poses": self._poses_total,
            "quality_issue": self._quality_issue,
        }


class MockFaceEngine:
    def __init__(self):
        self._faces = {}
        self._unknowns = {}

    def list_faces(self):
        from types import SimpleNamespace
        return [
            SimpleNamespace(name=k, pose_count=v)
            for k, v in self._faces.items()
        ]

    def list_unknowns(self):
        from types import SimpleNamespace
        return [
            SimpleNamespace(id=k, first_seen=v["first_seen"], cluster_count=v["cluster_count"])
            for k, v in self._unknowns.items()
        ]

    def face_exists(self, name):
        return name in self._faces

    def get_thumb(self, name):
        return None if name not in self._faces else b"fake_jpeg"

    def get_unknown_thumb(self, id):
        return None if id not in self._unknowns else b"fake_jpeg"

    def promote_unknown(self, face_id, name):
        if face_id not in self._unknowns:
            raise ValueError(f"Unknown face {face_id} not found")
        del self._unknowns[face_id]
        self._faces[name] = 5

    def delete_face(self, name):
        if name not in self._faces:
            raise ValueError(f"Face {name} not found")
        del self._faces[name]

    def delete_unknown(self, id):
        if id not in self._unknowns:
            raise ValueError(f"Unknown {id} not found")
        del self._unknowns[id]


class MockBTManager:
    def get_devices(self):
        from types import SimpleNamespace
        return [
            SimpleNamespace(id="dev1", name="JBL Go 3", rssi=-45, owner=True),
            SimpleNamespace(id="dev2", name="Pixel 7", rssi=-60, owner=False),
        ]


class MockTTS:
    def speak(self, text):
        pass


@pytest.fixture
def mock_state():
    return MockState()


@pytest.fixture
def mock_settings():
    return MockSettings()


@pytest.fixture
def mock_serial():
    return MockSerialLink()


@pytest.fixture
def mock_enrollment():
    return MockEnrollment()


@pytest.fixture
def mock_face_engine():
    return MockFaceEngine()


@pytest.fixture
def mock_bt_manager():
    return MockBTManager()


@pytest.fixture
def mock_tts():
    return MockTTS()


@pytest.fixture(autouse=True)
def _clear_route_deps():
    import turtao.api.routes_alert
    import turtao.api.routes_audio
    import turtao.api.routes_camera
    import turtao.api.routes_control
    import turtao.api.routes_environment
    import turtao.api.routes_faces
    import turtao.api.routes_mode
    import turtao.api.routes_settings
    import turtao.api.routes_ble
    import turtao.api.routes_misc

    for mod in [
        turtao.api.routes_alert,
        turtao.api.routes_audio,
        turtao.api.routes_camera,
        turtao.api.routes_control,
        turtao.api.routes_environment,
        turtao.api.routes_faces,
        turtao.api.routes_mode,
        turtao.api.routes_settings,
        turtao.api.routes_ble,
        turtao.api.routes_misc,
    ]:
        mod._deps.clear()
    yield


@pytest.fixture
def app(mock_state, mock_settings, mock_serial, mock_enrollment, mock_face_engine,
        mock_bt_manager, mock_tts):
    app = Flask(__name__)
    CORS(app)
    app.config["TESTING"] = True

    from turtao.api.routes_alert import alert_bp, inject_deps as inject_alert
    from turtao.api.routes_audio import audio_bp, inject_deps as inject_audio
    from turtao.api.routes_camera import camera_bp, inject_deps as inject_camera
    from turtao.api.routes_control import control_bp, inject_deps as inject_control
    from turtao.api.routes_environment import environment_bp, inject_deps as inject_env
    from turtao.api.routes_faces import faces_bp, inject_deps as inject_faces
    from turtao.api.routes_mode import mode_bp, inject_deps as inject_mode
    from turtao.api.routes_settings import settings_bp, inject_deps as inject_settings
    from turtao.api.routes_ble import ble_bp, inject_deps as inject_ble
    from turtao.api.routes_misc import misc_bp, inject_deps as inject_misc

    inject_alert(state=mock_state)
    inject_audio(tts=mock_tts)
    inject_camera(camera=MagicMock())
    inject_control(state=mock_state, serial=mock_serial, settings=mock_settings)
    inject_env(state=mock_state)
    inject_faces(face_engine=mock_face_engine, enrollment=mock_enrollment, config=MagicMock(), state=mock_state)
    inject_mode(state=mock_state, serial=mock_serial)
    inject_settings(settings=mock_settings, tts=mock_tts)
    inject_ble(state=mock_state, bt_manager=mock_bt_manager, settings=mock_settings, serial=mock_serial)
    inject_misc(state=mock_state, serial=mock_serial, settings=mock_settings)

    app.register_blueprint(alert_bp)
    app.register_blueprint(audio_bp)
    app.register_blueprint(camera_bp)
    app.register_blueprint(control_bp)
    app.register_blueprint(environment_bp)
    app.register_blueprint(faces_bp)
    app.register_blueprint(mode_bp)
    app.register_blueprint(settings_bp)
    app.register_blueprint(ble_bp)
    app.register_blueprint(misc_bp)

    from turtao.api.errors import register_error_handlers
    register_error_handlers(app)

    app.testing = True
    return app


@pytest.fixture
def client(app):
    return app.test_client()
