from __future__ import annotations

import logging
import subprocess
import threading
from pathlib import Path

from turtao.state import AppState, ThreatLabel

logger = logging.getLogger(__name__)

_PARENT = Path(__file__).resolve().parent.parent.parent
DEFAULT_PIPER_DIR = _PARENT / "piper"
DEFAULT_MODEL = "en_US-amy-medium.onnx"

_BOOT_DONE = False
_PREV_MODE: str = ""
_PREV_THREAT: str = ""
_PREV_GAS: int = 0
_PREV_AIR: int = 0
_PREV_BATTERY: float = 100.0


class TTSManager:
    def __init__(self, piper_dir: Path | None = None, model_name: str = DEFAULT_MODEL) -> None:
        self._piper_dir = Path(piper_dir) if piper_dir else DEFAULT_PIPER_DIR
        self._model_name = model_name
        self._lock = threading.Lock()

    def speak(self, text: str) -> None:
        threading.Thread(
            target=self._execute_tts,
            args=(text,),
            daemon=True,
        ).start()

    def _execute_tts(self, text: str) -> None:
        piper_bin = self._piper_dir / "piper"
        model_path = self._piper_dir / self._model_name
        if not piper_bin.exists() or not model_path.exists():
            logger.warning("Piper binary or model not found")
            return
        try:
            args = [
                str(piper_bin),
                "--model", str(model_path),
                "--output-raw",
            ]
            echo = subprocess.Popen(
                ["echo", text],
                stdout=subprocess.PIPE,
            )
            piper = subprocess.Popen(
                args,
                stdin=echo.stdout,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
            )
            if echo.stdout:
                echo.stdout.close()
            aplay_args = ["aplay", "-r", "22050", "-f", "S16_LE", "-t", "raw", "-"]
            subprocess.Popen(
                aplay_args,
                stdin=piper.stdout,
                stderr=subprocess.DEVNULL,
            )
            if piper.stdout:
                piper.stdout.close()
        except OSError as e:
            logger.error("TTS subprocess error: %s", e)


_MSG_MAP: list[tuple[str, str]] = [
    ("threat", "Intruder detected. Owner has been notified."),
    ("gas_danger", "Warning. Hazardous gas detected."),
    ("poor_air", "Warning. Poor air quality detected."),
    ("tamper", "Warning. Do not touch this device."),
    ("low_battery", "Battery critically low."),
    ("wake_word", "Yes?"),
]


def reset_tts_tracking() -> None:
    global _BOOT_DONE, _PREV_MODE, _PREV_THREAT, _PREV_GAS, _PREV_AIR, _PREV_BATTERY
    _BOOT_DONE = False
    _PREV_MODE = ""
    _PREV_THREAT = ""
    _PREV_GAS = 0
    _PREV_AIR = 0
    _PREV_BATTERY = 100.0


def get_tts_message(state: AppState, boot: bool = False, mode_changed: bool = False) -> str | None:
    global _BOOT_DONE, _PREV_MODE, _PREV_THREAT, _PREV_GAS, _PREV_AIR, _PREV_BATTERY

    if boot and not _BOOT_DONE:
        _BOOT_DONE = True
        return "Turtao online."

    if mode_changed and state.mode.value == "PATROL":
        return "Surveillance active. Beginning patrol."

    if _PREV_THREAT != "THREAT" and state.threat_label == ThreatLabel.THREAT:
        _PREV_THREAT = "THREAT"
        return _msg_for("threat")
    if state.threat_label != ThreatLabel.THREAT:
        _PREV_THREAT = "IDLE"

    gas = state.sensor_data.gas_mq2
    if _PREV_GAS <= 300 and gas > 300:
        _PREV_GAS = gas
        return _msg_for("gas_danger")
    _PREV_GAS = gas

    air = state.sensor_data.air_quality_mq135
    if _PREV_AIR <= 400 and air > 400:
        _PREV_AIR = air
        return _msg_for("poor_air")
    _PREV_AIR = air

    batt = state.battery.percent
    if _PREV_BATTERY >= 15 and batt < 15:
        _PREV_BATTERY = batt
        return _msg_for("low_battery")
    _PREV_BATTERY = batt

    o = state.sensor_data.orientation
    if abs(o.pitch) > 45 or abs(o.roll) > 45:
        return _msg_for("tamper")

    return None


def _msg_for(key: str) -> str | None:
    for k, msg in _MSG_MAP:
        if k == key:
            return msg
    return None


def is_wake_word_tts_enabled() -> bool:
    return _msg_for("wake_word") is not None
