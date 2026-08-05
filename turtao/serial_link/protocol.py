from __future__ import annotations

import json
import logging
from typing import Any, TypedDict

logger = logging.getLogger(__name__)

REQUIRED_SENSOR_FIELDS: set[str] = {
    "temp_inside_c", "temp_outside_c", "humidity_pct",
    "gas_mq2", "air_quality_mq135", "sound_level", "motion",
    "pitch", "roll", "yaw", "accel_x", "accel_y", "accel_z",
    "tof_fl", "tof_fc", "tof_fr", "tof_down",
    "voltage", "current_ma", "battery_pct",
    "motor_controller_ok", "firmware_version", "ble_devices",
}


class InvalidPayload(TypedDict):
    error: str
    raw: str


def encode_command(cmd: dict[str, Any]) -> str:
    return json.dumps(cmd, separators=(",", ":")) + "\n"


def decode_payload(line: str) -> tuple[bool, dict[str, Any]]:
    try:
        data = json.loads(line)
    except json.JSONDecodeError:
        logger.warning("Failed to decode payload: %r", line)
        return False, {"error": "invalid_json", "raw": line}
    if not isinstance(data, dict):
        logger.warning("Decoded payload is not an object: %r", data)
        return False, {"error": "invalid_json", "raw": line}
    return True, data


def validate_payload(data: dict[str, Any]) -> bool:
    if not isinstance(data, dict):
        return False
    return REQUIRED_SENSOR_FIELDS.issubset(data.keys())
