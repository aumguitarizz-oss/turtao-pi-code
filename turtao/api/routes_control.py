import json
import logging
from flask import Blueprint, request
from turtao.api.errors import APIError
from turtao.state import Mode

logger = logging.getLogger(__name__)

control_bp = Blueprint("control", __name__)

_deps: dict = {}

PAN_MIN = 5
PAN_MAX = 175
TILT_MIN = 60
TILT_MAX = 120
MOVE_CLAMP = 0.8


def inject_deps(**kwargs) -> None:
    _deps.update(kwargs)


@control_bp.route("/api/control", methods=["POST"])
def control():
    body = request.get_json(silent=True)
    if not body:
        raise APIError("Request body required", "VALIDATION_ERROR", 400)

    ser = _deps.get("serial")
    st = _deps.get("state")
    settings_obj = _deps.get("settings")

    speed = body.get("speed")
    nerf = body.get("nerf")
    safe_mode = body.get("safe_mode")
    pan = body.get("pan")
    tilt = body.get("tilt")

    if speed is not None:
        if not isinstance(speed, (int, float)):
            raise APIError("speed must be a number", "VALIDATION_ERROR", 400)
        if settings_obj is not None:
            settings_obj.speed = speed
            try:
                settings_obj.save()
            except Exception:
                logger.exception("Failed to persist speed setting")
        if speed == 0 and safe_mode is True:
            if st is not None:
                st.mode = Mode.IDLE
            if ser is not None:
                try:
                    ser.write(json.dumps({"cmd": "estop"}) + "\n")
                except Exception:
                    logger.exception("Failed to send estop")
            return {"ok": True}

    if safe_mode is not None and settings_obj is not None:
        settings_obj.safe_mode = safe_mode
        try:
            settings_obj.save()
        except Exception:
            logger.exception("Failed to persist safe_mode setting")

    if pan is not None or tilt is not None:
        # The API is 0-centered degrees (-90..+90); servo space is 90-centered
        # (PAN_MIN..PAN_MAX / TILT_MIN..TILT_MAX), so shift by +90 before clamping.
        cmd_pan = max(PAN_MIN, min(PAN_MAX, int(pan) + 90)) if pan is not None else None
        cmd_tilt = max(TILT_MIN, min(TILT_MAX, int(tilt) + 90)) if tilt is not None else None
        payload: dict = {"cmd": "pan_tilt"}
        if cmd_pan is not None:
            payload["pan"] = cmd_pan
        if cmd_tilt is not None:
            payload["tilt"] = cmd_tilt
        if ser is not None:
            try:
                ser.write(json.dumps(payload) + "\n")
            except Exception:
                logger.exception("Failed to send pan/tilt command")

    return {"ok": True}


@control_bp.route("/api/move", methods=["POST"])
def move():
    body = request.get_json(silent=True)
    if not body:
        raise APIError("Request body required", "VALIDATION_ERROR", 400)

    ml = body.get("ml")
    mr = body.get("mr")
    if ml is None or mr is None:
        raise APIError("ml and mr fields required", "VALIDATION_ERROR", 400)
    if not isinstance(ml, (int, float)) or not isinstance(mr, (int, float)):
        raise APIError("ml and mr must be numbers", "VALIDATION_ERROR", 400)

    settings_obj = _deps.get("settings")
    speed_mult = settings_obj.speed if settings_obj is not None else 1.0

    ml = max(-MOVE_CLAMP, min(MOVE_CLAMP, float(ml) * speed_mult))
    mr = max(-MOVE_CLAMP, min(MOVE_CLAMP, float(mr) * speed_mult))

    ser = _deps.get("serial")
    if ser is not None:
        try:
            cmd = json.dumps({"cmd": "move", "ml": round(ml, 2), "mr": round(mr, 2)})
            ser.write(cmd + "\n")
        except Exception:
            logger.exception("Failed to send move command")

    return {"ok": True}
