import logging
from flask import Blueprint, request
from turtao.api.errors import APIError

logger = logging.getLogger(__name__)

mode_bp = Blueprint("mode", __name__)

_deps: dict = {}

VALID_MODES = {"IDLE", "GUARD", "PATROL"}


def inject_deps(**kwargs) -> None:
    _deps.update(kwargs)


@mode_bp.route("/api/mode", methods=["GET", "POST"])
def mode():
    st = _deps.get("state")
    if st is None:
        return {"error": "SERVICE_UNAVAILABLE", "detail": "State not available"}, 503

    if request.method == "GET":
        return {"mode": getattr(st, "mode", "IDLE")}

    body = request.get_json(silent=True)
    if not body or "mode" not in body:
        raise APIError("Missing mode field", "VALIDATION_ERROR", 400)

    new_mode = body["mode"].upper()
    if new_mode not in VALID_MODES:
        raise APIError(
            f"Invalid mode '{body['mode']}'. Must be one of: {', '.join(sorted(VALID_MODES))}",
            "VALIDATION_ERROR",
            400,
        )

    try:
        setattr(st, "mode", new_mode)
    except Exception as exc:
        raise APIError(str(exc), "MODE_FAILURE", 500) from exc

    logger.info("Mode set to %s", new_mode)
    return {"mode": new_mode}
