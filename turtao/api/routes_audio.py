import logging
import tempfile
from pathlib import Path

from flask import Blueprint, request

from turtao.api.errors import APIError

logger = logging.getLogger(__name__)

audio_bp = Blueprint("audio", __name__)

_deps: dict = {}


def inject_deps(**kwargs) -> None:
    _deps.update(kwargs)


@audio_bp.route("/api/audio/play", methods=["POST"])
def play_audio():
    if "audio" not in request.files:
        raise APIError("audio field required", "VALIDATION_ERROR", 400)

    audio_file = request.files["audio"]
    if audio_file.filename == "":
        raise APIError("audio field required", "VALIDATION_ERROR", 400)

    tts = _deps.get("tts")
    if tts is None:
        return {"error": "SERVICE_UNAVAILABLE", "detail": "TTS not available"}, 503

    with tempfile.NamedTemporaryFile(suffix=".m4a", delete=False) as tmp:
        audio_file.save(tmp.name)
        tmp_path = tmp.name

    try:
        tts.play_file(tmp_path)
    except Exception as exc:
        Path(tmp_path).unlink(missing_ok=True)
        raise APIError(str(exc), "PLAYBACK_FAILURE", 500) from exc

    return {"ok": True}
