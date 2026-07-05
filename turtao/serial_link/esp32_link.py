from __future__ import annotations

import logging
import time
from glob import glob
from threading import Lock
from typing import Any

import serial

from turtao.config import AppConfig
from turtao.hardware.interfaces import SerialLinkInterface
from turtao.serial_link.protocol import decode_payload, encode_command, validate_payload
from turtao.state import AppState

logger = logging.getLogger(__name__)

_INITIAL_BACKOFF = 1.0
_MAX_BACKOFF = 30.0
_MULTIPLIER = 2.0


class ESP32SerialLink(SerialLinkInterface):
    def __init__(self, config: AppConfig, state: AppState) -> None:
        self._config = config
        self._state = state
        self._ser: serial.Serial | None = None
        self._lock = Lock()

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
        ser = self._ser
        if ser is None or not ser.is_open:
            logger.warning("Cannot write: serial port not open")
            return
        try:
            ser.write(data.encode("utf-8"))
        except serial.SerialException as e:
            logger.error("Serial write error: %s", e)

    def is_connected(self) -> bool:
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
        logger.warning("Sensor payload missing required fields")
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
                sensor = self.poll_sensor()
                if sensor is not None:
                    with self._state:
                        self._state.sensor_data.temp_c = sensor.get("temp_dht", 0.0)
                        self._state.sensor_data.humidity_pct = int(sensor.get("humidity_pct", 0))
                        self._state.sensor_data.pressure_hpa = sensor.get("pressure_hpa", 0.0)
                        self._state.sensor_data.gas_mq2 = sensor.get("gas_mq2", 0)
                        self._state.sensor_data.air_quality_mq135 = sensor.get("air_quality_mq135", 0)
                        self._state.sensor_data.sound_level = sensor.get("sound_level", 0.0)
                        self._state.sensor_data.motion = sensor.get("motion", False)
                        self._state.sensor_data.orientation.pitch = sensor.get("pitch", 0.0)
                        self._state.sensor_data.orientation.roll = sensor.get("roll", 0.0)
                        self._state.sensor_data.orientation.yaw = sensor.get("yaw", 0.0)
                        self._state.sensor_data.tof_cm = [
                            sensor.get("tof_fl", 0),
                            sensor.get("tof_fc", 0),
                            sensor.get("tof_fr", 0),
                            sensor.get("tof_down", 0),
                        ]
                        self._state.sensor_data.voltage = sensor.get("voltage", 0.0)
                        self._state.sensor_data.current_ma = sensor.get("current_ma", 0)
                        self._state.sensor_data.battery_pct = sensor.get("battery_pct", 0.0)
                        self._state.sensor_data.motor_controller_ok = sensor.get("motor_controller_ok", True)
                        self._state.sensor_data.firmware_version = sensor.get("firmware_version", "")
                        self._state.sensor_data.ble_devices = sensor.get("ble_devices", [])
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
