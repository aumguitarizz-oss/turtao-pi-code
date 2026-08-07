from __future__ import annotations

import json
import logging
from typing import Any, TypedDict

logger = logging.getLogger(__name__)

# Matches the stripped-build ESP32-S3 firmware's sendSensorPayload()
# exactly — see inos/turtao_esp32_firmware1_copy_20260807222228.ino.
# Any field can independently be null (a missing sensor or failed
# read), so this only checks presence, not value.
REQUIRED_SENSOR_FIELDS: set[str] = {
    "tof_front", "gas_mq2", "temp_dht", "humidity",
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
