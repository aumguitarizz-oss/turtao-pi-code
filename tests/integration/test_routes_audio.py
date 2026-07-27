import io
import os
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

    def test_route_does_not_delete_temp_file_itself(self, client):
        """The route hands the temp file off to TTSManager.play_file and must
        not delete it itself — cleanup is owned by _execute_play_file, which
        runs (and deletes the file) only after aplay actually finishes
        playing it. With `tts` mocked, play_file never runs and never
        deletes anything, so if the route returned with the file still on
        disk, that proves the route itself performed no cleanup.
        """
        from turtao.api import routes_audio
        tts = MagicMock()
        routes_audio.inject_deps(tts=tts)
        data = {"audio": (io.BytesIO(b"fake m4a bytes"), "clip.m4a")}
        resp = client.post("/api/audio/play", data=data, content_type="multipart/form-data")
        assert resp.status_code == 200

        tmp_path = tts.play_file.call_args.args[0]
        assert os.path.exists(tmp_path)

        os.unlink(tmp_path)

    def test_play_audio_with_default_mock_tts(self, client):
        """Uses the default `mock_tts` fixture (no MagicMock override) to
        prove the route works against the actual MockTTS test double, which
        must implement play_file (see turtao/audio/tts.py TTSManager).
        """
        data = {"audio": (io.BytesIO(b"fake m4a bytes"), "clip.m4a")}
        resp = client.post("/api/audio/play", data=data, content_type="multipart/form-data")
        assert resp.status_code == 200
