import time
import logging
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
            ser.write(f"LED {mode}\n".encode())
        except Exception:
            logger.exception("Failed to send LED command to serial")

    return {"ok": True}


@misc_bp.route("/api/events")
def events():
    st = _deps.get("state")
    if st is None:
        return {"error": "SERVICE_UNAVAILABLE", "detail": "State not available"}, 503

    event_log = getattr(st, "event_log", [])
    capped = list(event_log)[:50]
    return [
        {
            "id": e.id,
            "type": e.type,
            "message": e.message,
            "at": e.at,
        }
        for e in capped
    ]


@misc_bp.route("/api/map")
def map():
    st = _deps.get("state")
    if st is None:
        return {"error": "SERVICE_UNAVAILABLE", "detail": "State not available"}, 503

    grid = getattr(st, "map_grid", [])
    trail = getattr(st, "map_trail", [])
    return {"grid": grid, "trail": trail}


@misc_bp.route("/api/ota/version")
def ota_version():
    ser = _deps.get("serial")
    if ser is None:
        return {"error": "SERVICE_UNAVAILABLE", "detail": "Serial not available"}, 503

    try:
        ver = ser.query("OTA_VERSION\n")
    except Exception as exc:
        raise APIError(str(exc), "OTA_QUERY_FAILURE", 500) from exc

    return {"version": ver.strip()}


@misc_bp.route("/api/ota/update", methods=["POST"])
def ota_update():
    ser = _deps.get("serial")
    if ser is None:
        return {"error": "SERVICE_UNAVAILABLE", "detail": "Serial not available"}, 503

    try:
        ser.write(b"OTA_UPDATE\n")
    except Exception as exc:
        raise APIError(str(exc), "OTA_UPDATE_FAILURE", 500) from exc

    return {"ok": True}
