from __future__ import annotations

import threading
import time
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
class FaceDetection:
    box: tuple[int, int, int, int]
    name: str
    label: ThreatLabel
    confidence: float


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
    faces: list[FaceDetection] = field(default_factory=list)


@dataclass
class ImuReading:
    """Raw accel/gyro off the GY-91 (MPU9250 half) — no fused orientation
    exists on real hardware, so there's no pitch/roll/yaw to report."""
    accel_x: float = 0.0
    accel_y: float = 0.0
    accel_z: float = 0.0
    gyro_x: float = 0.0
    gyro_y: float = 0.0
    gyro_z: float = 0.0


@dataclass
class SensorData:
    temp_dht: float | None = 0.0
    humidity: float | None = 0.0
    gas_mq2: float = 0.0
    air_quality_mq135: float = 0.0
    sound_raw: int = 0
    motion: bool = False  # sourced from the firmware's `pir` field
    imu: ImuReading = field(default_factory=ImuReading)
    tof_cm: list[int] = field(default_factory=lambda: [0, 0, 0, 0])


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
        self.pose_landmarks: list[list[tuple[int, int]]] = []  # one list per detected person
        self.frame_queue: deque = deque(maxlen=2)
        # tracker_ids currently past LoiterMonitor's 10s alarm threshold —
        # lets the WS payload mark just that person's box for the app's
        # distinct (not full-THREAT) lingering-person alarm styling.
        self.alarmed_person_ids: set[int] = set()
        self.events: deque[Event] = deque(maxlen=50)
        self.event_counter: int = 0
        self.serial_command_queue: deque[dict[str, Any]] = deque()
        self._stop_event = threading.Event()

    @property
    def stop_event(self) -> threading.Event:
        return self._stop_event

    def emit_event(self, event_type: str, message: str) -> None:
        """Append an Event, thread-safe regardless of whether the caller
        already holds the state lock (acquires its own). Lets both
        core.py's background loops and Flask routes (which only ever get
        `state` injected, not the TurtaoCore orchestrator) log an event the
        same way."""
        with self:
            self.event_counter += 1
            self.events.append(Event(
                id=f"evt_{self.event_counter}",
                type=event_type,
                message=message,
                at=time.strftime("%Y-%m-%dT%H:%M:%S"),
            ))

    def acquire(self) -> None:
        self._lock.acquire()

    def release(self) -> None:
        self._lock.release()

    def __enter__(self) -> AppState:
        self._lock.acquire()
        return self

    def __exit__(self, *args: Any) -> None:
        self._lock.release()
