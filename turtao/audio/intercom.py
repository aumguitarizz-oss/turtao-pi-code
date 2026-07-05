from __future__ import annotations

import logging
import threading
from typing import Any

logger = logging.getLogger(__name__)


class IntercomBridge:
    FORMAT = 16
    CHANNELS = 1
    RATE = 16000
    FRAMES_PER_BUFFER = 1024

    def __init__(self) -> None:
        self._stream: Any = None
        self._audio: Any = None
        self._running = False
        self._lock = threading.Lock()

    @property
    def is_running(self) -> bool:
        return self._running

    def feed_pcm(self, data: bytes) -> None:
        if not self._running or self._stream is None:
            return
        try:
            self._stream.write(data)
        except Exception:
            logger.exception("Failed to write PCM data to audio stream")

    def start(self) -> None:
        with self._lock:
            if self._running:
                return
            try:
                import pyaudio
                self._audio = pyaudio.PyAudio()
                self._stream = self._audio.open(
                    format=pyaudio.paInt16,
                    channels=self.CHANNELS,
                    rate=self.RATE,
                    output=True,
                    frames_per_buffer=self.FRAMES_PER_BUFFER,
                )
                self._running = True
                logger.info(
                    "Intercom bridge started (%d Hz, %d ch, %d-bit)",
                    self.RATE,
                    self.CHANNELS,
                    self.FORMAT,
                )
            except Exception:
                logger.exception("Failed to start intercom bridge")
                self._running = False

    def stop(self) -> None:
        with self._lock:
            self._running = False
            if self._stream is not None:
                try:
                    self._stream.stop_stream()
                    self._stream.close()
                except Exception:
                    logger.exception("Failed to close audio stream")
                self._stream = None
            if self._audio is not None:
                try:
                    self._audio.terminate()
                except Exception:
                    logger.exception("Failed to terminate PyAudio")
                self._audio = None
            logger.info("Intercom bridge stopped")
