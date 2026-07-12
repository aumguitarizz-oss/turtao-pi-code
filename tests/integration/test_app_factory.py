from unittest.mock import MagicMock

from turtao.api.app_factory import create_app
from turtao.config import AppConfig, Settings
from turtao.state import AppState


class TestCreateApp:
    def test_create_app_succeeds_with_mocks(self):
        app = create_app(
            state=AppState(),
            settings=Settings(),
            config=AppConfig(),
            serial_link=MagicMock(),
            camera=MagicMock(),
            face_engine=MagicMock(),
            enrollment=MagicMock(),
            tts=MagicMock(),
            bt_manager=MagicMock(),
            tracker=MagicMock(),
            antispoof=MagicMock(),
        )

        assert app is not None

        client = app.test_client()
        resp = client.get("/api/health")
        assert resp.status_code == 200
        assert resp.get_json()["status"] == "ok"
