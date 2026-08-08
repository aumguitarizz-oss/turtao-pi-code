import json
import logging

from flask import Blueprint, request

from turtao.api.errors import APIError
from turtao.state import Mode

logger = logging.getLogger(__name__)

control_bp = Blueprint("control", __name__)

_deps: dict = {}

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

    if speed is not None:
        if not isinstance(speed, (int, float)):
            raise APIError("speed must be a number", "VALIDATION_ERROR", 400)

        if speed == 0 and safe_mode is True:
            # Emergency stop: halt immediately, but don't persist 0 as the
            # new default speed -- that would silently zero every future
            # /api/move command (ml/mr are scaled by settings.speed) until
            # someone manually restores it in Settings.
            if st is not None:
                st.mode = Mode.IDLE
            if ser is not None:
                try:
                    ser.write(json.dumps({"cmd": "estop"}) + "\n")
                except Exception:
                    logger.exception("Failed to send estop")
            return {"ok": True}

        if settings_obj is not None:
            settings_obj.speed = speed
            try:
                settings_obj.save()
            except Exception:
                logger.exception("Failed to persist speed setting")

    if safe_mode is not None and settings_obj is not None:
        settings_obj.safe_mode = safe_mode
        try:
            settings_obj.save()
        except Exception:
            logger.exception("Failed to persist safe_mode setting")

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
