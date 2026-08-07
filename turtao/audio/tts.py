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

    def _execute_play_file(self, path: str) -> None:
        try:
            proc = subprocess.Popen(
                ["aplay", path],
                stderr=subprocess.DEVNULL,
            )
            proc.wait()
            if proc.returncode != 0:
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
            subprocess.Popen(
                aplay_args,
                stdin=piper.stdout,
                stderr=subprocess.DEVNULL,
            )
            if piper.stdout:
                piper.stdout.close()
        except OSError as e:
            logger.error("TTS subprocess error: %s", e)


