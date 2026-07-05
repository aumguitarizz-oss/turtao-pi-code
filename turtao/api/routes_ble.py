import json
import logging
from flask import Blueprint, request
from turtao.api.errors import APIError

logger = logging.getLogger(__name__)

ble_bp = Blueprint("ble", __name__)

_deps: dict = {}


def inject_deps(**kwargs) -> None:
    _deps.update(kwargs)


@ble_bp.route("/api/ble/devices")
def ble_devices():
    st = _deps.get("state")
    if st is None:
        return {"error": "SERVICE_UNAVAILABLE", "detail": "State not available"}, 503

    devices = getattr(st, "ble_devices", [])
    return [
        {
            "id": d.get("id", ""),
            "name": d.get("name", ""),
            "rssi": d.get("rssi", 0),
            "owner": d.get("owner", False),
        }
        for d in devices
    ]


@ble_bp.route("/api/ble/register", methods=["POST"])
def ble_register():
    body = request.get_json(silent=True)
    if not body or "mac" not in body:
        raise APIError("mac field required", "VALIDATION_ERROR", 400)

    mac = body["mac"].strip()
    if not mac:
        raise APIError("mac must not be empty", "VALIDATION_ERROR", 400)

    s = _deps.get("settings")
    if s is not None:
        s.phone_registration = mac
        try:
            s.save()
        except Exception as exc:
            raise APIError(str(exc), "SAVE_FAILURE", 500) from exc

    ser = _deps.get("serial")
    if ser is not None:
        try:
            cmd = json.dumps({"cmd": "ble_register", "mac": mac})
            ser.write(cmd + "\n")
        except Exception:
            logger.exception("Failed to send BLE registration to serial")

    return {"ok": True, "mac": mac}
