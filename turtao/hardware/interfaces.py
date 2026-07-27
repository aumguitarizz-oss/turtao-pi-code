from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class SerialLinkInterface(ABC):
    @abstractmethod
    def open(self) -> None:
        ...

    @abstractmethod
    def readline(self) -> str | None:
        ...

    @abstractmethod
    def write(self, data: str) -> None:
        ...

    @abstractmethod
    def is_connected(self) -> bool:
        ...

    @abstractmethod
    def close(self) -> None:
        ...


class CameraInterface(ABC):
    @abstractmethod
    def read(self) -> tuple[bool, Any]:
        ...

    @abstractmethod
    def release(self) -> None:
        ...

    @abstractmethod
    def get_latest_frame(self) -> bytes | None:
        ...
