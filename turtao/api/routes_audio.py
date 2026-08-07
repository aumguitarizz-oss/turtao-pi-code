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

    # aplay (what TTSManager._execute_play_file shells out to) only decodes
    # raw PCM/WAV -- it cannot play AAC/M4A at all. This used to hardcode
    # ".m4a" regardless of what was actually uploaded, which silently broke
    # playback for the app's own AAC-encoded soundboard recordings (now
    # switched to WAV). Preserve the real extension so a mismatch here is
    # at least visible instead of masked by a wrong-but-plausible suffix.
    suffix = Path(audio_file.filename).suffix or ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        audio_file.save(tmp.name)
        tmp_path = tmp.name

    try:
        tts.play_file(tmp_path)
    except Exception as exc:
        Path(tmp_path).unlink(missing_ok=True)
        raise APIError(str(exc), "PLAYBACK_FAILURE", 500) from exc

    return {"ok": True}
