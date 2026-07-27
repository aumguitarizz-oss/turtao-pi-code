import json
import logging
import time

from flask import Blueprint, request

from turtao.api.errors import APIError

logger = logging.getLogger(__name__)

misc_bp = Blueprint("misc", __name__)

_deps: dict = {}
_start_time: float = time.time()


def inject_deps(**kwargs) -> None:
    _deps.update(kwargs)


@misc_bp.route("/api/health")
def health():
    return {"status": "ok", "uptime_s": int(time.time() - _start_time)}


@misc_bp.route("/api/led", methods=["POST"])
def led():
    body = request.get_json(silent=True)
    if not body or "mode" not in body:
        raise APIError("mode field required", "VALIDATION_ERROR", 400)

    valid_modes = {"off", "illuminate", "strobe", "flashbang"}
    mode = body["mode"]
    if mode not in valid_modes:
        raise APIError(
            f"Invalid mode '{mode}'. Must be one of: {', '.join(sorted(valid_modes))}",
            "VALIDATION_ERROR",
            400,
        )

    ser = _deps.get("serial")
    if ser is not None:
        try:
            cmd = json.dumps({"cmd": "led", "mode": mode})
            ser.write(cmd + "\n")
        except Exception:
            logger.exception("Failed to send LED command to serial")

    return {"ok": True, "mode": mode}


@misc_bp.route("/api/events")
def events():
    st = _deps.get("state")
    if st is None:
        return {"error": "SERVICE_UNAVAILABLE", "detail": "State not available"}, 503

    return [
        {"id": e.id, "type": e.type, "message": e.message, "at": e.at}
        for e in st.events
    ]


@misc_bp.route("/api/map")
def map():
    return {"grid": [], "trail": []}


@misc_bp.route("/api/ota/version")
def ota_version():
    st = _deps.get("state")
    if st is None:
        return {"error": "SERVICE_UNAVAILABLE", "detail": "State not available"}, 503
    return {"version": st.sensor_data.firmware_version or "0.0.0"}


@misc_bp.route("/api/ota/update", methods=["POST"])
def ota_update():
    ser = _deps.get("serial")
    if ser is None:
        return {"error": "SERVICE_UNAVAILABLE", "detail": "Serial not available"}, 503

    try:
        ser.write(json.dumps({"cmd": "ota_update"}) + "\n")
    except Exception as exc:
        raise APIError(str(exc), "OTA_UPDATE_FAILURE", 500) from exc

    return {"ok": True, "status": "started"}
