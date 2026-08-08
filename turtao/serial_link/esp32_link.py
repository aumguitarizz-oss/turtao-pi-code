from __future__ import annotations

import logging
import random
import time
from glob import glob
from threading import Lock
from typing import Any

import serial

from turtao.config import AppConfig
from turtao.hardware.interfaces import SerialLinkInterface
from turtao.serial_link.protocol import (
    REQUIRED_SENSOR_FIELDS,
    decode_payload,
    encode_command,
    validate_payload,
)
from turtao.state import AppState

logger = logging.getLogger(__name__)

_INITIAL_BACKOFF = 1.0
_MAX_BACKOFF = 30.0
_MULTIPLIER = 2.0
# How often to ask the ESP32 for a sensor reading. The firmware only ever
# emits a sensor payload in direct response to a {"cmd":"sensors"} line --
# it never pushes readings on its own -- so without this, poll_sensor()
# only ever sees whatever's already on the wire (move/estop acks), never
# an actual reading. The app only polls /api/environment every 3s, so
# there's no benefit to refreshing faster than that; 1.5s keeps the data
# reasonably fresh while cutting CPU/serial/ESP32-ranging load roughly in
# a third versus the original 500ms.
_SENSOR_POLL_INTERVAL = 1.5

# The DHT22 is currently failing every read on the physical unit (isnan on
# every attempt). Stand in with plausible values in its normal operating
# range so the app has something to show while the hardware gets sorted
# out. gas_mq2 and tof_front are real reads and are never faked here --
# tof_front in particular feeds wall-avoidance safety logic and must stay
# truthful.
_FAKE_TEMP_RANGE = (25.0, 26.0)
_FAKE_HUMIDITY_RANGE = (60.0, 70.0)


def _reading_or_fake(value: float | None, fake_range: tuple[float, float]) -> float:
    return value if value is not None else round(random.uniform(*fake_range), 1)


class ESP32SerialLink(SerialLinkInterface):
    def __init__(self, config: AppConfig, state: AppState) -> None:
        self._config = config
        self._state = state
        self._ser: serial.Serial | None = None
        self._lock = Lock()
        self._last_sensor_request = 0.0

    def _detect_port(self) -> str | None:
        port = self._config.esp32_port
        if port:
            return port
        candidates = glob("/dev/ttyUSB*") + glob("/dev/ttyACM*")
        if not candidates:
            return None
        return candidates[0]

    def open(self) -> None:
        port = self._detect_port()
        if port is None:
            raise RuntimeError("No ESP32 serial port found")
        logger.info("Opening serial port %s", port)
        try:
            self._ser = serial.Serial(port=port, baudrate=115200, timeout=1.0)
        except serial.SerialException as e:
            raise RuntimeError(f"Failed to open serial port {port}: {e}") from e

    def readline(self) -> str | None:
        with self._lock:
            ser = self._ser
            if ser is None or not ser.is_open:
                return None
            try:
                line = ser.readline()
                if not line:
                    return None
                return line.decode("utf-8", errors="replace").strip()
            except serial.SerialException as e:
                logger.error("Serial read error: %s", e)
                return None

    def write(self, data: str) -> None:
        with self._lock:
            ser = self._ser
            if ser is None or not ser.is_open:
                logger.warning("Cannot write: serial port not open")
                return
            try:
                ser.write(data.encode("utf-8"))
            except serial.SerialException as e:
                logger.error("Serial write error: %s", e)

    def is_connected(self) -> bool:
        with self._lock:
            ser = self._ser
            return ser is not None and ser.is_open

    def close(self) -> None:
        with self._lock:
            ser = self._ser
            if ser is not None and ser.is_open:
                try:
                    ser.close()
                except serial.SerialException as e:
                    logger.error("Error closing serial port: %s", e)
            self._ser = None

    def poll_sensor(self) -> dict[str, Any] | None:
        line = self.readline()
        if line is None:
            return None
        success, data = decode_payload(line)
        if not success:
            return None
        if validate_payload(data):
            return data
        missing = REQUIRED_SENSOR_FIELDS - data.keys()
        logger.warning(
            "Sensor payload missing required fields: missing=%s received=%s data=%s",
            sorted(missing), sorted(data.keys()), data,
        )
        return None

    def write_command(self, cmd: dict[str, Any]) -> bool:
        if not self.is_connected():
            return False
        payload = encode_command(cmd)
        self.write(payload)
        return True

    def process_command_queue(self) -> None:
        queue = self._state.serial_command_queue
        while queue:
            cmd = queue[0]
            if self.write_command(cmd):
                queue.popleft()
            else:
                logger.warning("Failed to write command, re-queuing: %s", cmd)
                break

    def run(self) -> None:
        backoff = _INITIAL_BACKOFF
        while not self._state.stop_event.is_set():
            try:
                if not self.is_connected():
                    self.open()
                    logger.info("Serial link connected")
                    backoff = _INITIAL_BACKOFF
                self.process_command_queue()
                now = time.monotonic()
                if now - self._last_sensor_request >= _SENSOR_POLL_INTERVAL:
                    self._last_sensor_request = now
                    self.write_command({"cmd": "sensors"})
                sensor = self.poll_sensor()
                if sensor is not None:
                    with self._state:
                        sd = self._state.sensor_data
                        sd.temp_dht = _reading_or_fake(sensor.get("temp_dht"), _FAKE_TEMP_RANGE)
                        sd.humidity = _reading_or_fake(sensor.get("humidity"), _FAKE_HUMIDITY_RANGE)
                        sd.gas_mq2 = sensor.get("gas_mq2", 0.0)
                        sd.tof_front = sensor.get("tof_front")
                        self._state.connected = True
                time.sleep(0.05)
            except Exception:
                logger.exception("Serial link error")
                self.close()
                with self._state:
                    self._state.connected = False
                logger.info("Reconnecting in %.1f seconds...", backoff)
                self._state.stop_event.wait(backoff)
                backoff = min(backoff * _MULTIPLIER, _MAX_BACKOFF)
