from pathlib import Path

from turtao.audio.tts import TTSManager


class TestExecutePlayFile:
    def test_execute_play_file_waits_before_deleting(self, tmp_path, monkeypatch):
        call_order = []
        test_file = tmp_path / "clip.m4a"
        test_file.write_bytes(b"fake audio")

        class FakeProc:
            returncode = 0

            def poll(self):
                return 0  # not running -- nothing to preempt

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


class TestStartExclusive:
    """Regression: play_file()/speak() used to spawn a fully independent
    thread + aplay process per call with no coordination at all -- a burst
    of soundboard taps piled up concurrent processes fighting over the
    same Bluetooth sink, which drained out later in arbitrary order (long
    delay, then an earlier tap's sound plays after the user's moved on).
    _start_exclusive must kill whatever's still running before starting
    the next one, so at most one process ever plays and the latest
    request always wins immediately."""

    def test_kills_still_running_process_before_starting_the_next(self, monkeypatch):
        events = []

        class FakeProc:
            def __init__(self, args):
                self.args = args

            def poll(self):
                return None  # still running

            def terminate(self):
                events.append(f"terminate:{self.args[-1]}")

            def wait(self, timeout=None):
                events.append(f"wait:{self.args[-1]}")

        def fake_popen(args, **kwargs):
            events.append(f"popen:{args[-1]}")
            return FakeProc(args)

        monkeypatch.setattr("subprocess.Popen", fake_popen)

        tts = TTSManager(piper_dir="/tmp")
        tts._start_exclusive(["aplay", "first.wav"])
        tts._start_exclusive(["aplay", "second.wav"])

        assert events == [
            "popen:first.wav",
            "terminate:first.wav",
            "wait:first.wav",
            "popen:second.wav",
        ]

    def test_does_not_touch_an_already_finished_process(self, monkeypatch):
        events = []

        class FakeProc:
            def __init__(self, args):
                self.args = args

            def poll(self):
                return 0  # already exited on its own

            def terminate(self):
                events.append(f"terminate:{self.args[-1]}")

            def wait(self, timeout=None):
                pass

        def fake_popen(args, **kwargs):
            events.append(f"popen:{args[-1]}")
            return FakeProc(args)

        monkeypatch.setattr("subprocess.Popen", fake_popen)

        tts = TTSManager(piper_dir="/tmp")
        tts._start_exclusive(["aplay", "first.wav"])
        tts._start_exclusive(["aplay", "second.wav"])

        assert events == ["popen:first.wav", "popen:second.wav"]
