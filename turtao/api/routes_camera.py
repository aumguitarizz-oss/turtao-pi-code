import logging
from flask import Blueprint, Response, send_file

logger = logging.getLogger(__name__)

camera_bp = Blueprint("camera", __name__)

_deps: dict = {}


def inject_deps(**kwargs) -> None:
    _deps.update(kwargs)


@camera_bp.route("/api/stream")
def stream():
    cam = _deps.get("camera")
    if cam is None:
        return {"error": "SERVICE_UNAVAILABLE", "detail": "Camera not available"}, 503
    return Response(cam.generate_mjpeg_stream(), mimetype="multipart/x-mixed-replace; boundary=frame")


@camera_bp.route("/api/snapshot")
def snapshot():
    cam = _deps.get("camera")
    if cam is None:
        return {"error": "SERVICE_UNAVAILABLE", "detail": "Camera not available"}, 503
    jpeg_bytes = cam.get_latest_frame()
    if jpeg_bytes is None:
        return {"error": "NO_FRAME", "detail": "No frame available"}, 503
    return send_file(jpeg_bytes, mimetype="image/jpeg")
