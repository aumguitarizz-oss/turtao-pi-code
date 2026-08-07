import time

from flask import Blueprint

misc_bp = Blueprint("misc", __name__)

_deps: dict = {}
_start_time: float = time.time()


def inject_deps(**kwargs) -> None:
    _deps.update(kwargs)


@misc_bp.route("/api/health")
def health():
    return {"status": "ok", "uptime_s": int(time.time() - _start_time)}


@misc_bp.route("/api/events")
def events():
    st = _deps.get("state")
    if st is None:
        return {"error": "SERVICE_UNAVAILABLE", "detail": "State not available"}, 503

    return [
        {"id": e.id, "type": e.type, "message": e.message, "at": e.at}
        for e in reversed(st.events)
    ]


@misc_bp.route("/api/map")
def map():
    return {"grid": [], "trail": []}


@misc_bp.route("/api/ota/version")
def ota_version():
    # The confirmed firmware has no version-report field in its sensor
    # payload, and no query command for it — there is no way to read this
    # from real hardware today.
    return {
        "error": "SERVICE_UNAVAILABLE",
        "detail": "Firmware version not reported by hardware",
    }, 503
