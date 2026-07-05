import logging
from flask import Blueprint

logger = logging.getLogger(__name__)

environment_bp = Blueprint("environment", __name__)

_deps: dict = {}


def inject_deps(**kwargs) -> None:
    _deps.update(kwargs)


@environment_bp.route("/api/environment")
def get_environment():
    st = _deps.get("state")
    if st is None:
        return {"error": "SERVICE_UNAVAILABLE", "detail": "State not available"}, 503
    tof = getattr(st, "tof_cm", [0, 0, 0, 0])
    return {
        "temp_c": getattr(st, "temp_c", 0.0),
        "humidity_pct": getattr(st, "humidity_pct", 0),
        "pressure_hpa": getattr(st, "pressure_hpa", 0.0),
        "gas_mq2": getattr(st, "gas_mq2", 0),
        "air_quality_mq135": getattr(st, "air_quality_mq135", 0),
        "sound_level": getattr(st, "sound_level", 0.0),
        "motion": getattr(st, "motion", False),
        "orientation": {
            "pitch": getattr(st, "pitch", 0.0),
            "roll": getattr(st, "roll", 0.0),
            "yaw": getattr(st, "yaw", 0.0),
        },
        "tof_cm": tof,
    }


@environment_bp.route("/api/battery")
def get_battery():
    st = _deps.get("state")
    if st is None:
        return {"error": "SERVICE_UNAVAILABLE", "detail": "State not available"}, 503
    return {
        "voltage": getattr(st, "battery_voltage", 0.0),
        "current_ma": getattr(st, "battery_current", 0),
        "percent": getattr(st, "battery_percent", 0.0),
        "status": getattr(st, "battery_state", "discharging"),
    }
