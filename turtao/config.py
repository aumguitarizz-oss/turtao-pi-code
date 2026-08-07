from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
ENV_PATH = BASE_DIR / ".env"
SETTINGS_PATH = BASE_DIR / "settings.json"


class TTSEventToggles(BaseModel):
    threat: bool = True
    low_battery: bool = True
    intruder: bool = True


class Notifications(BaseModel):
    threat: bool = True
    gas_danger: bool = True
    temp_danger: bool = True
    low_battery: bool = True
    tamper: bool = True
    connection_lost: bool = True


class Settings(BaseModel):
    hostname: str = ""
    ble_proximity_enabled: bool = True
    phone_registration: str = ""
    tts_event_toggles: TTSEventToggles = TTSEventToggles()
    intercom_volume: float = 0.7
    face_tolerance: float = 0.52
    anti_spoof_enabled: bool = True
    # When False, enrollment skips the too_far/blurry/too_dark/too_bright
    # quality gate (check_quality) — a face still has to be detected and
    # encodable, but a merely imperfect scan is no longer rejected. Useful
    # for a fast, uninterrupted enrollment (e.g. a live demo) at the cost
    # of weaker embeddings from lower-quality poses.
    strict_face_scan: bool = True
    # MQ2 gas sensor + DHT temperature alert thresholds — a reading outside
    # [low, high] logs an alert event (turtao/core.py's
    # _check_sensor_alerts). Defaults are starting points, not calibrated
    # against real hardware; adjust from the app's Settings tab.
    gas_threshold_low: float = 50.0
    gas_threshold_high: float = 400.0
    temp_threshold_low: float = 0.0
    temp_threshold_high: float = 45.0
    speed: float = 0.8
    safe_mode: bool = False
    auto_flashbang: bool = False
    stealth_mode: bool = False
    notifications: Notifications = Notifications()

    def shallow_merge(self, partial: dict[str, Any]) -> Settings:
        data = self.model_dump()
        for key, value in partial.items():
            if key in data:
                if isinstance(data[key], dict) and isinstance(value, dict):
                    data[key].update(value)
                else:
                    data[key] = value
        return Settings.model_validate(data)

    def asdict(self) -> dict[str, Any]:
        return self.model_dump()

    def save(self) -> None:
        save_settings(self)


class AppConfig(BaseModel):
    flask_port: int = Field(default=5000, ge=1024, le=65535)
    esp32_port: str = ""
    camera_index: int = 0
    jbl_mac: str = ""


def load_dotenv_config() -> AppConfig:
    if ENV_PATH.exists():
        load_dotenv(ENV_PATH)

    return AppConfig(
        flask_port=int(os.getenv("FLASK_PORT", "5000")),
        esp32_port=os.getenv("ESP32_PORT", ""),
        camera_index=int(os.getenv("CAMERA_INDEX", "0")),
        jbl_mac=os.getenv("JBL_MAC", ""),
    )


def load_settings(path: Path | None = None) -> Settings:
    path = path or SETTINGS_PATH
    if not path.exists():
        logger.warning("Settings file not found at %s, using defaults", path)
        return Settings()

    try:
        with open(path) as f:
            data = json.load(f)
        return Settings.model_validate(data)
    except (json.JSONDecodeError, ValueError) as e:
        logger.warning("Corrupt settings file at %s: %s. Using defaults with partial fallback.", path, e)
        try:
            with open(path) as f:
                data = json.load(f)
            return Settings.model_validate(data)
        except Exception:
            return Settings()


def save_settings(settings: Settings, path: Path | None = None) -> None:
    path = path or SETTINGS_PATH
    with open(path, "w") as f:
        json.dump(settings.model_dump(), f, indent=2)
