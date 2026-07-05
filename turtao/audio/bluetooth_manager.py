from __future__ import annotations

import logging
import subprocess
import time
from pathlib import Path

logger = logging.getLogger(__name__)

_PARENT = Path(__file__).resolve().parent.parent.parent
SILENCE_WAV = _PARENT / "sounds" / "silence.wav"


class BluetoothManager:
    def __init__(self, jbl_mac: str = "") -> None:
        self._mac = jbl_mac
        self._keepalive_proc: subprocess.Popen[bytes] | None = None
        self._connected = False
        if not jbl_mac:
            logger.warning("JBL_MAC not configured; Bluetooth disabled")

    @property
    def is_connected(self) -> bool:
        return self._connected

    def connect(self) -> bool:
        if not self._mac:
            return False
        try:
            result = subprocess.run(
                ["bluetoothctl", "connect", self._mac],
                capture_output=True,
                timeout=30,
            )
            if result.returncode == 0:
                self._connected = True
                logger.info("Connected to JBL Go 3 (%s)", self._mac)
                return True
            stderr = result.stderr.decode(errors="replace").strip()
            logger.warning("bluetoothctl connect failed (rc=%d): %s", result.returncode, stderr)
            return False
        except (subprocess.TimeoutExpired, OSError) as e:
            logger.error("Failed to connect to JBL: %s", e)
            return False

    def start_silent_keepalive(self) -> None:
        if not SILENCE_WAV.exists():
            logger.warning("Silence file not found: %s", SILENCE_WAV)
            return
        try:
            self._keepalive_proc = subprocess.Popen(
                ["aplay", "--loop", str(SILENCE_WAV)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            logger.info("Keepalive started (PID %d)", self._keepalive_proc.pid)
        except OSError as e:
            logger.error("Failed to start keepalive: %s", e)

    def check_keepalive(self) -> None:
        if self._keepalive_proc is None:
            logger.warning("Keepalive not running; starting")
            self.start_silent_keepalive()
            return
        if self._keepalive_proc.poll() is not None:
            logger.warning("Keepalive died (rc=%s); restarting", self._keepalive_proc.returncode)
            self.start_silent_keepalive()

    def disconnect(self) -> None:
        self._connected = False
        if self._keepalive_proc is not None:
            try:
                self._keepalive_proc.terminate()
                self._keepalive_proc.wait(timeout=5)
            except (OSError, subprocess.TimeoutExpired) as e:
                logger.warning("Keepalive terminate failed: %s", e)
                try:
                    self._keepalive_proc.kill()
                    self._keepalive_proc.wait(timeout=3)
                except OSError:
                    pass
            self._keepalive_proc = None
        if self._mac:
            try:
                subprocess.run(
                    ["bluetoothctl", "disconnect", self._mac],
                    capture_output=True,
                    timeout=10,
                )
                logger.info("Disconnected JBL Go 3 (%s)", self._mac)
            except (subprocess.TimeoutExpired, OSError) as e:
                logger.error("Failed to disconnect JBL: %s", e)


def bluetooth_loop(state: object, bt_manager: BluetoothManager) -> None:
    logger.info("Bluetooth loop started; waiting 35 s for BT init")
    time.sleep(35)

    if not bt_manager.is_connected:
        bt_manager.connect()

    bt_manager.start_silent_keepalive()

    while True:
        time.sleep(60)
        bt_manager.check_keepalive()
