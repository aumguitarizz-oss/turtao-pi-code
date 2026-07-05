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
    sd = st.sensor_data
    return {
        "temp_c": sd.temp_c,
        "humidity_pct": sd.humidity_pct,
        "pressure_hpa": sd.pressure_hpa,
        "gas_mq2": sd.gas_mq2,
        "air_quality_mq135": sd.air_quality_mq135,
        "sound_level": sd.sound_level,
        "motion": sd.motion,
        "orientation": {
            "pitch": sd.orientation.pitch,
            "roll": sd.orientation.roll,
            "yaw": sd.orientation.yaw,
        },
        "tof_cm": sd.tof_cm,
    }


@environment_bp.route("/api/battery")
def get_battery():
    st = _deps.get("state")
    if st is None:
        return {"error": "SERVICE_UNAVAILABLE", "detail": "State not available"}, 503
    b = st.battery
    return {
        "voltage": b.voltage,
        "current_ma": b.current_ma,
        "percent": b.percent,
        "status": b.status,
    }
