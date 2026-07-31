from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np


class Mode(str, Enum):
    IDLE = "IDLE"
    GUARD = "GUARD"
    PATROL = "PATROL"


class ThreatLabel(str, Enum):
    IDLE = "IDLE"
    SAFE = "SAFE"
    THREAT = "THREAT"


@dataclass
class BatteryData:
    voltage: float = 0.0
    current_ma: int = 0
    percent: float = 0.0
    status: str = "discharging"


@dataclass
class ThreatState:
    active: bool = False
    face_crop: bytes | None = None
    confidence: float = 0.0
    timestamp: float | None = None
    box: tuple[int, int, int, int] | None = None
    landmarks: list[tuple[int, int]] = field(default_factory=list)
    name: str = ""


@dataclass
class Orientation:
    pitch: float = 0.0
    roll: float = 0.0
    yaw: float = 0.0


@dataclass
class SensorData:
    temp_inside_c: float = 0.0
    temp_outside_c: float = 0.0
    humidity_pct: int = 0
    gas_mq2: int = 0
    air_quality_mq135: int = 0
    sound_level: float = 0.0
    motion: bool = False
    orientation: Orientation = field(default_factory=Orientation)
    tof_cm: list[int] = field(default_factory=lambda: [0, 0, 0, 0])
    voltage: float = 0.0
    current_ma: int = 0
    battery_pct: float = 0.0
    motor_controller_ok: bool = True
    firmware_version: str = ""
    ble_devices: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class Event:
    id: str
    type: str
    message: str
    at: str


class AppState:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.mode: Mode = Mode.IDLE
        self.threat_label: ThreatLabel = ThreatLabel.IDLE
        self.threat_state: ThreatState = ThreatState()
        self.sensor_data: SensorData = SensorData()
        self.battery: BatteryData = BatteryData()
        self.heading: int = 0
        self.connected: bool = False
        self.latency_ms: int = 0
        self.pan: int = 90
        self.tilt: int = 90
        self.frame_counter: int = 0
        self.latest_frame: np.ndarray | None = None
        self.latest_annotated_frame: np.ndarray | None = None
        self.latest_persons: list[dict[str, Any]] = []
        self.show_yolo: bool = True
        self.show_mediapipe: bool = False
        self.pose_landmarks: list[tuple[int, int]] = []
        self.frame_queue: deque = deque(maxlen=2)
        self.events: deque[Event] = deque(maxlen=50)
        self.event_counter: int = 0
        self.serial_command_queue: deque[dict[str, Any]] = deque()
        self._stop_event = threading.Event()

    @property
    def stop_event(self) -> threading.Event:
        return self._stop_event

    def acquire(self) -> None:
        self._lock.acquire()

    def release(self) -> None:
        self._lock.release()

    def __enter__(self) -> AppState:
        self._lock.acquire()
        return self

    def __exit__(self, *args: Any) -> None:
        self._lock.release()
