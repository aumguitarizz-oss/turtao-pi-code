from __future__ import annotations

import logging
import subprocess
import time
from pathlib import Path

from turtao.config import MAIN_SPEAKER_MAC

logger = logging.getLogger(__name__)

_PARENT = Path(__file__).resolve().parent.parent.parent
# Reuses the tone install.sh already generates for the (currently
# unrelated) alert-sound provisioning step — no need for a second asset.
CONNECTED_CHIME = _PARENT / "sounds" / "alert.wav"


class BluetoothManager:
    def __init__(self, jbl_mac: str = MAIN_SPEAKER_MAC) -> None:
        self._mac = jbl_mac
        self._connected = False
        if not jbl_mac:
            logger.warning("No speaker MAC configured; Bluetooth disabled")

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
                logger.info("Connected to main speaker (%s)", self._mac)
                self._play_connected_chime()
                return True
            stderr = result.stderr.decode(errors="replace").strip()
            logger.warning("bluetoothctl connect failed (rc=%d): %s", result.returncode, stderr)
            return False
        except (subprocess.TimeoutExpired, OSError) as e:
            logger.error("Failed to connect to main speaker: %s", e)
            return False

    def _play_connected_chime(self) -> None:
        """One-shot audible confirmation that the speaker paired — not a
        keepalive. Once connected, any real playback (TTS, alerts) already
        goes to whatever the default audio sink is, so no ongoing silent
        stream is needed to keep the link "warm" between them."""
        if not CONNECTED_CHIME.exists():
            logger.warning("Connected-chime file not found: %s", CONNECTED_CHIME)
            return
        try:
            subprocess.Popen(
                ["aplay", str(CONNECTED_CHIME)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except OSError as e:
            logger.error("Failed to play connected chime: %s", e)

    def disconnect(self) -> None:
        self._connected = False
        if self._mac:
            try:
                subprocess.run(
                    ["bluetoothctl", "disconnect", self._mac],
                    capture_output=True,
                    timeout=10,
                )
                logger.info("Disconnected main speaker (%s)", self._mac)
            except (subprocess.TimeoutExpired, OSError) as e:
                logger.error("Failed to disconnect main speaker: %s", e)


def bluetooth_loop(state: object, bt_manager: BluetoothManager) -> None:
    """Connect once, ~35s after boot (giving the BT stack time to init).
    No ongoing keepalive loop — real audio playback already routes to the
    speaker once connected, and reconnecting after a drop is a manual/
    future concern, not something this loop handles today."""
    logger.info("Bluetooth loop started; waiting 35 s for BT init")
    time.sleep(35)

    if not bt_manager.is_connected:
        bt_manager.connect()
