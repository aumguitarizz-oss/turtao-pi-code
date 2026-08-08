from __future__ import annotations

import logging
import subprocess
import threading
from pathlib import Path

logger = logging.getLogger(__name__)

_PARENT = Path(__file__).resolve().parent.parent.parent
DEFAULT_PIPER_DIR = _PARENT / "piper"
DEFAULT_MODEL = "en_US-amy-medium.onnx"


class TTSManager:
    def __init__(self, piper_dir: Path | None = None, model_name: str = DEFAULT_MODEL) -> None:
        self._piper_dir = Path(piper_dir) if piper_dir else DEFAULT_PIPER_DIR
        self._model_name = model_name
        self._lock = threading.Lock()
        self._current_proc: subprocess.Popen | None = None

    def speak(self, text: str) -> None:
        threading.Thread(
            target=self._execute_tts,
            args=(text,),
            daemon=True,
        ).start()

    def play_file(self, path: str) -> None:
        threading.Thread(
            target=self._execute_play_file,
            args=(path,),
            daemon=True,
        ).start()

    def _start_exclusive(self, args: list[str], **popen_kwargs) -> subprocess.Popen:
        """Kills whatever's currently playing and starts a new process in
        its place, atomically. Every play_file()/speak() call used to spawn
        an independent thread with zero coordination between them --
        self._lock was declared but never actually used -- so a burst of
        taps piled up concurrent aplay processes all competing for the same
        Bluetooth sink. They backed up and drained out later, in whatever
        order the OS/BlueALSA happened to unblock them, which surfaced as a
        long delay followed by an earlier tap's sound playing well after
        the user had moved on. Now at most one audio process ever runs, and
        the most recent request always wins immediately instead of queuing."""
        with self._lock:
            old = self._current_proc
            if old is not None and old.poll() is None:
                old.terminate()
                try:
                    old.wait(timeout=1)
                except subprocess.TimeoutExpired:
                    old.kill()
            proc = subprocess.Popen(args, **popen_kwargs)
            self._current_proc = proc
            return proc

    def _execute_play_file(self, path: str) -> None:
        try:
            proc = self._start_exclusive(["aplay", path], stderr=subprocess.DEVNULL)
            proc.wait()
            # A negative returncode means a newer request preempted this
            # one via terminate()/kill() -- expected, not a real failure.
            if proc.returncode not in (0, None) and proc.returncode > 0:
                logger.warning("aplay exited with code %d playing %s", proc.returncode, path)
        except OSError as e:
            logger.error("Audio playback subprocess error: %s", e)
        finally:
            Path(path).unlink(missing_ok=True)

    def _execute_tts(self, text: str) -> None:
        piper_bin = self._piper_dir / "piper"
        model_path = self._piper_dir / self._model_name
        if not piper_bin.exists() or not model_path.exists():
            logger.warning("Piper binary or model not found")
            return
        try:
            args = [
                str(piper_bin),
                "--model", str(model_path),
                "--output-raw",
            ]
            echo = subprocess.Popen(
                ["echo", text],
                stdout=subprocess.PIPE,
            )
            piper = subprocess.Popen(
                args,
                stdin=echo.stdout,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
            )
            if echo.stdout:
                echo.stdout.close()
            aplay_args = ["aplay", "-r", "22050", "-f", "S16_LE", "-t", "raw", "-"]
            self._start_exclusive(aplay_args, stdin=piper.stdout, stderr=subprocess.DEVNULL)
            if piper.stdout:
                piper.stdout.close()
        except OSError as e:
            logger.error("TTS subprocess error: %s", e)


