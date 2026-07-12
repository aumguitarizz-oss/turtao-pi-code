import logging
from flask import Blueprint, request
from turtao.api.errors import APIError

logger = logging.getLogger(__name__)

settings_bp = Blueprint("settings", __name__)

_deps: dict = {}

SCALAR_TYPES = (str, int, float, bool)


def inject_deps(**kwargs) -> None:
    _deps.update(kwargs)


@settings_bp.route("/api/settings", methods=["GET", "POST"])
def settings():
    s = _deps.get("settings")
    if s is None:
        return {"error": "SERVICE_UNAVAILABLE", "detail": "Settings not available"}, 503

    if request.method == "GET":
        return s.asdict()

    body = request.get_json(silent=True)
    if not body:
        raise APIError("Request body required", "VALIDATION_ERROR", 400)

    known = s.asdict()
    for key, value in body.items():
        if key not in known:
            raise APIError(f"Unknown setting '{key}'", "VALIDATION_ERROR", 400)
        current = known[key]
        if isinstance(current, dict):
            if not isinstance(value, dict):
                raise APIError(f"Invalid type for '{key}'", "VALIDATION_ERROR", 400)
        elif not isinstance(value, SCALAR_TYPES) and value is not None:
            raise APIError(f"Invalid type for '{key}'", "VALIDATION_ERROR", 400)

    merged = s.shallow_merge(body)
    for key, value in vars(merged).items():
        setattr(s, key, value)

    try:
        s.save()
    except Exception as exc:
        raise APIError(str(exc), "SAVE_FAILURE", 500) from exc

    return s.asdict()


@settings_bp.route("/api/settings/test-voice", methods=["POST"])
def test_voice():
    tts = _deps.get("tts")
    if tts is None:
        return {"error": "SERVICE_UNAVAILABLE", "detail": "TTS not available"}, 503

    try:
        tts.speak("Turtao online")
    except Exception as exc:
        raise APIError(str(exc), "TTS_FAILURE", 500) from exc

    return {"ok": True}
