import logging
from flask import Blueprint, request
from turtao.api.errors import APIError

logger = logging.getLogger(__name__)

control_bp = Blueprint("control", __name__)

_deps: dict = {}


def inject_deps(**kwargs) -> None:
    _deps.update(kwargs)


@control_bp.route("/api/control", methods=["POST"])
def control():
    body = request.get_json(silent=True)
    if not body:
        raise APIError("Request body required", "VALIDATION_ERROR", 400)

    ser = _deps.get("serial")
    st = _deps.get("state")

    speed = body.get("speed")
    nerf = body.get("nerf")
    safe_mode = body.get("safe_mode")
    pan = body.get("pan")
    tilt = body.get("tilt")

    if speed is not None:
        if not isinstance(speed, (int, float)):
            raise APIError("speed must be a number", "VALIDATION_ERROR", 400)
        if speed == 0 and safe_mode is True:
            if st is not None:
                st.mode = "IDLE"
            if ser is not None:
                try:
                    ser.write(b'{"cmd":"estop"}\n')
                except Exception:
                    logger.exception("Failed to send estop")
            return {"ok": True}

    if ser is not None:
        parts = []
        if speed is not None:
            parts.append(f"SPD {speed}")
        if nerf is not None:
            parts.append(f"NERF {int(nerf)}")
        if safe_mode is not None:
            parts.append(f"SAFE {int(safe_mode)}")
        if pan is not None:
            parts.append(f"PAN {pan}")
        if tilt is not None:
            parts.append(f"TILT {tilt}")
        if parts:
            try:
                ser.write(" ".join(parts).encode() + b"\n")
            except Exception:
                logger.exception("Failed to send control command")

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

    ml = max(-0.8, min(0.8, float(ml)))
    mr = max(-0.8, min(0.8, float(mr)))

    st = _deps.get("state")
    speed_mult = getattr(st, "speed", 1.0) if st is not None else 1.0
    ml *= speed_mult
    mr *= speed_mult

    ser = _deps.get("serial")
    if ser is not None:
        try:
            ser.write(f"MOVE {ml:.2f} {mr:.2f}\n".encode())
        except Exception:
            logger.exception("Failed to send move command")

    return {"ok": True}
