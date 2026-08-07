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
        "temp_dht": sd.temp_dht,
        "humidity": sd.humidity,
        "gas_mq2": sd.gas_mq2,
        "air_quality_mq135": sd.air_quality_mq135,
        "sound_raw": sd.sound_raw,
        "motion": sd.motion,
        "imu": {
            "accel_x": sd.imu.accel_x,
            "accel_y": sd.imu.accel_y,
            "accel_z": sd.imu.accel_z,
            "gyro_x": sd.imu.gyro_x,
            "gyro_y": sd.imu.gyro_y,
            "gyro_z": sd.imu.gyro_z,
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
    # No INA219 (or any battery monitoring hardware) on this build — the
    # confirmed ESP32-S3 firmware has no battery-related field anywhere in
    # its sensor payload. Report unavailable rather than fabricating a
    # reading (state.battery was always its 0.0 defaults; nothing fed it).
    return {"error": "SERVICE_UNAVAILABLE", "detail": "No battery hardware on this build"}, 503
