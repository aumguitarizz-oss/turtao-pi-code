from pathlib import Path

from turtao.audio.tts import TTSManager


class TestExecutePlayFile:
    def test_execute_play_file_waits_before_deleting(self, tmp_path, monkeypatch):
        call_order = []
        test_file = tmp_path / "clip.m4a"
        test_file.write_bytes(b"fake audio")

        class FakeProc:
            returncode = 0

            def wait(self):
                call_order.append("wait")

        def fake_popen(*args, **kwargs):
            call_order.append("popen")
            return FakeProc()

        original_unlink = Path.unlink

        def tracking_unlink(self, *args, **kwargs):
            call_order.append("unlink")
            return original_unlink(self, *args, **kwargs)

        monkeypatch.setattr("subprocess.Popen", fake_popen)
        monkeypatch.setattr(Path, "unlink", tracking_unlink)

        tts = TTSManager(piper_dir=tmp_path)
        tts._execute_play_file(str(test_file))

        assert call_order == ["popen", "wait", "unlink"]
