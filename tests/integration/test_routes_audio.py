import io
import pytest
from unittest.mock import MagicMock


class TestAudioRoutes:
    def test_play_audio_with_file(self, client):
        from turtao.api import routes_audio
        tts = MagicMock()
        routes_audio.inject_deps(tts=tts)
        data = {"audio": (io.BytesIO(b"fake m4a bytes"), "clip.m4a")}
        resp = client.post("/api/audio/play", data=data, content_type="multipart/form-data")
        assert resp.status_code == 200
        assert resp.get_json() == {"ok": True}
        tts.play_file.assert_called_once()

    def test_play_audio_missing_file_returns_400(self, client):
        resp = client.post("/api/audio/play", data={}, content_type="multipart/form-data")
        assert resp.status_code == 400
