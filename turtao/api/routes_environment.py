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
        "temp_inside_c": sd.temp_inside_c,
        "temp_outside_c": sd.temp_outside_c,
        "humidity_pct": sd.humidity_pct,
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


@environment_bp.route("/api/sensors/test-alert", methods=["POST"])
def test_sensor_alert():
    """Fires one gas_danger and one temp_danger event through the exact
    same state.emit_event path the real threshold check in core.py uses,
    so the Settings tab's "Send test alert" button can verify the whole
    pipeline (Pi event -> app event log) without needing to actually
    create a dangerous gas or temperature reading."""
    st = _deps.get("state")
    if st is None:
        return {"error": "SERVICE_UNAVAILABLE", "detail": "State not available"}, 503
    st.emit_event("gas_danger", "Test alert: MQ2 gas reading out of range")
    st.emit_event("temp_danger", "Test alert: temperature reading out of range")
    return {"ok": True}


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
