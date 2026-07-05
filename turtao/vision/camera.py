from __future__ import annotations

import logging
import time

import cv2
import numpy as np

from turtao.hardware.interfaces import CameraInterface
from turtao.state import AppState

logger = logging.getLogger(__name__)

BRIGHTNESS_MIN = 40
BRIGHTNESS_MAX = 220
BRIGHTNESS_INTERVAL = 30
RETRY_DELAY = 1.0


class Camera(CameraInterface):
    def __init__(self, index: int = 0) -> None:
        self._cap = cv2.VideoCapture(index)
        if not self._cap.isOpened():
            logger.error("Failed to open camera at index %d", index)

    def read(self) -> tuple[bool, np.ndarray | None]:
        if self._cap is None or not self._cap.isOpened():
            return False, None
        ret, frame = self._cap.read()
        if not ret:
            return False, None
        return True, frame

    def release(self) -> None:
        if self._cap is not None:
            self._cap.release()


def camera_capture_loop(state: AppState, camera: CameraInterface) -> None:
    """Daemon thread target. Continuously captures frames, pushes to frame_queue.
    Every 30 frames, runs brightness check (40-220 range). Logs warnings on dark/bright."""
    frame_count = 0

    while not state.stop_event.is_set():
        ret, frame = camera.read()
        if not ret or frame is None:
            logger.error("Camera read failed, retrying in %.1fs", RETRY_DELAY)
            time.sleep(RETRY_DELAY)
            continue

        with state:
            state.latest_frame = frame
            state.frame_queue.append(frame)
            state.frame_counter += 1
            frame_count = state.frame_counter

        if frame_count % BRIGHTNESS_INTERVAL == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            mean_brightness = np.mean(gray)
            if mean_brightness < BRIGHTNESS_MIN:
                logger.warning("Frame too dark (mean=%.1f)", mean_brightness)
            elif mean_brightness > BRIGHTNESS_MAX:
                logger.warning("Frame too bright (mean=%.1f)", mean_brightness)

    camera.release()
