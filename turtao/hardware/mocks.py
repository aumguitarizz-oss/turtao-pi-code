from __future__ import annotations

from collections import deque
from typing import Any

import numpy as np

from turtao.hardware.interfaces import CameraInterface, SerialLinkInterface


class MockSerialLink(SerialLinkInterface):
    def __init__(self, responses: list[str] | None = None) -> None:
        self._responses: deque[str] = deque(responses or [])
        self._written: list[str] = []
        self._connected = False
        self._fail_on_write = False

    def open(self) -> None:
        self._connected = True

    def readline(self) -> str | None:
        if self._responses:
            return self._responses.popleft()
        return None

    def write(self, data: str) -> None:
        if self._fail_on_write:
            raise ConnectionError("mock write failure")
        self._written.append(data)

    def is_connected(self) -> bool:
        return self._connected

    def close(self) -> None:
        self._connected = False

    def inject_response(self, line: str) -> None:
        self._responses.append(line)

    @property
    def written(self) -> list[str]:
        return self._written

    def set_fail_on_write(self, fail: bool) -> None:
        self._fail_on_write = fail


class MockCamera(CameraInterface):
    def __init__(self, width: int = 640, height: int = 480) -> None:
        self._width = width
        self._height = height
        self._frame_index = 0
        self._released = False
        self._fail_on_read = False

    def read(self) -> tuple[bool, Any]:
        if self._fail_on_read or self._released:
            return False, None
        frame: Any = np.zeros((self._height, self._width, 3), dtype=np.uint8)
        self._frame_index += 1
        return True, frame

    def release(self) -> None:
        self._released = True

    def set_fail_on_read(self, fail: bool) -> None:
        self._fail_on_read = fail

    def get_latest_frame(self) -> bytes | None:
        ret, _frame = self.read()
        if not ret:
            return None
        return b"fake_jpeg"
