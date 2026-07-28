import io
import logging

import cv2
import numpy as np
from flask import Blueprint, request, send_file

from turtao.api.errors import APIError

logger = logging.getLogger(__name__)

faces_bp = Blueprint("faces", __name__)

_deps: dict = {}


def inject_deps(**kwargs) -> None:
    _deps.update(kwargs)


def _enrollment_status(enroll) -> dict:
    s = enroll.get_status()
    active = s.get("status") == "active"
    return {
        "active": active,
        "pose_index": (s.get("pose", 1) - 1) if active else 0,
        "poses_total": s.get("total_poses", 5),
        "quality_issue": s.get("quality_issue") or None,
        "processing": enroll.is_processing,
    }


@faces_bp.route("/api/faces")
def list_faces():
    engine = _deps.get("face_engine")
    if engine is None:
        return {"error": "SERVICE_UNAVAILABLE", "detail": "Face engine not available"}, 503
    faces = engine.list_faces()
    return [
        {
            "name": f.name,
            "thumb_url": f"/api/faces/{f.name}/thumb",
            "poses": f.pose_count,
        }
        for f in faces
    ]


@faces_bp.route("/api/faces/unknowns")
def list_unknowns():
    engine = _deps.get("face_engine")
    if engine is None:
        return {"error": "SERVICE_UNAVAILABLE", "detail": "Face engine not available"}, 503
    unknowns = engine.list_unknowns()
    return [
        {
            "id": u.id,
            "image_url": f"/api/faces/unknowns/{u.id}/thumb",
            "first_seen": u.first_seen,
            "cluster_count": u.cluster_count,
        }
        for u in unknowns
    ]


@faces_bp.route("/api/faces/<name>/thumb")
def face_thumb(name: str):
    engine = _deps.get("face_engine")
    if engine is None:
        return {"error": "SERVICE_UNAVAILABLE", "detail": "Face engine not available"}, 503
    jpeg = engine.get_thumb(name)
    if jpeg is None:
        raise APIError("Face not found", "NOT_FOUND", 404)
    return send_file(io.BytesIO(jpeg), mimetype="image/jpeg")


@faces_bp.route("/api/faces/unknowns/<id>/thumb")
def unknown_thumb(id: str):
    engine = _deps.get("face_engine")
    if engine is None:
        return {"error": "SERVICE_UNAVAILABLE", "detail": "Face engine not available"}, 503
    jpeg = engine.get_unknown_thumb(id)
    if jpeg is None:
        raise APIError("Unknown face not found", "NOT_FOUND", 404)
    return send_file(io.BytesIO(jpeg), mimetype="image/jpeg")


@faces_bp.route("/api/faces/enroll/status")
def enroll_status():
    enroll = _deps.get("enrollment")
    if enroll is None:
        return {"error": "SERVICE_UNAVAILABLE", "detail": "Enrollment not available"}, 503
    return _enrollment_status(enroll)


@faces_bp.route("/api/faces/enroll/start", methods=["POST"])
def enroll_start():
    body = request.get_json(silent=True)
    if not body or "name" not in body:
        raise APIError("name field required", "VALIDATION_ERROR", 400)

    name = body["name"].strip()
    if not name:
        raise APIError("name must not be empty", "VALIDATION_ERROR", 400)

    enroll = _deps.get("enrollment")
    if enroll is None:
        return {"error": "SERVICE_UNAVAILABLE", "detail": "Enrollment not available"}, 503

    status = enroll.get_status()
    if status.get("status") == "active":
        raise APIError("Enrollment already in progress", "ENROLL_IN_PROGRESS", 409)

    engine = _deps.get("face_engine")
    if engine is not None and engine.face_exists(name):
        raise APIError(f"Face '{name}' already exists", "FACE_EXISTS", 409)

    try:
        enroll.start_enrollment(name)
    except Exception as exc:
        raise APIError(str(exc), "ENROLL_FAILURE", 500) from exc

    return _enrollment_status(enroll)


@faces_bp.route("/api/faces/enroll/capture", methods=["POST"])
def enroll_capture():
    enroll = _deps.get("enrollment")
    if enroll is None:
        return {"error": "SERVICE_UNAVAILABLE", "detail": "Enrollment not available"}, 503

    if enroll.get_status().get("status") != "active":
        raise APIError("No active enrollment", "NO_ENROLLMENT", 409)

    state = _deps.get("state")
    if state is None or state.latest_frame is None:
        raise APIError("No camera frame available", "NO_FRAME", 503)

    if not enroll.try_begin_processing():
        raise APIError("Capture already in progress", "CAPTURE_IN_PROGRESS", 409)

    try:
        enroll.capture_pose(lambda: state.latest_frame)
    except Exception as exc:
        raise APIError(str(exc), "CAPTURE_FAILURE", 500) from exc
    finally:
        enroll.release_processing()

    status = _enrollment_status(enroll)
    if not status["active"]:
        engine = _deps.get("face_engine")
        if engine is not None:
            engine.load_embeddings(str(engine._embeddings_dir))
    return status


MAX_UPLOAD_FRAMES = 10


@faces_bp.route("/api/faces/enroll/capture_upload", methods=["POST"])
def enroll_capture_upload():
    enroll = _deps.get("enrollment")
    if enroll is None:
        return {"error": "SERVICE_UNAVAILABLE", "detail": "Enrollment not available"}, 503

    if enroll.get_status().get("status") != "active":
        raise APIError("No active enrollment", "NO_ENROLLMENT", 409)

    files = request.files.getlist("frames")
    if not files:
        raise APIError("frames field required", "VALIDATION_ERROR", 400)
    if len(files) > MAX_UPLOAD_FRAMES:
        raise APIError(f"Too many frames (max {MAX_UPLOAD_FRAMES})", "VALIDATION_ERROR", 400)

    decoded = []
    for f in files:
        data = f.read()
        arr = np.frombuffer(data, dtype=np.uint8)
        try:
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        except cv2.error:
            frame = None
        if frame is None:
            raise APIError("Undecodable frame in upload", "VALIDATION_ERROR", 400)
        decoded.append(frame)

    if not enroll.try_begin_processing():
        raise APIError("Capture already in progress", "CAPTURE_IN_PROGRESS", 409)

    engine = _deps.get("face_engine")

    def _reload_on_complete() -> None:
        if engine is not None:
            engine.load_embeddings(str(engine._embeddings_dir))

    enroll.start_capture_burst(decoded, on_complete=_reload_on_complete)
    return {"ok": True}, 202


@faces_bp.route("/api/faces/enroll/cancel", methods=["POST"])
def enroll_cancel():
    enroll = _deps.get("enrollment")
    if enroll is None:
        return {"error": "SERVICE_UNAVAILABLE", "detail": "Enrollment not available"}, 503

    try:
        enroll.cancel_enrollment()
    except Exception as exc:
        raise APIError(str(exc), "CANCEL_FAILURE", 500) from exc

    return _enrollment_status(enroll)


@faces_bp.route("/api/faces/promote", methods=["POST"])
def promote():
    body = request.get_json(silent=True)
    if not body or "face_id" not in body or "name" not in body:
        raise APIError("face_id and name fields required", "VALIDATION_ERROR", 400)

    face_id = body["face_id"].strip()
    name = body["name"].strip()
    if not face_id or not name:
        raise APIError("face_id and name must not be empty", "VALIDATION_ERROR", 400)

    engine = _deps.get("face_engine")
    if engine is None:
        return {"error": "SERVICE_UNAVAILABLE", "detail": "Face engine not available"}, 503

    try:
        engine.promote_unknown(face_id, name)
    except ValueError as exc:
        raise APIError(str(exc), "NOT_FOUND", 404) from exc
    except Exception as exc:
        raise APIError(str(exc), "PROMOTE_FAILURE", 500) from exc

    return {"ok": True}


@faces_bp.route("/api/faces/<name>", methods=["DELETE"])
def delete_face(name: str):
    engine = _deps.get("face_engine")
    if engine is None:
        return {"error": "SERVICE_UNAVAILABLE", "detail": "Face engine not available"}, 503

    try:
        engine.delete_face(name)
    except ValueError as exc:
        raise APIError(str(exc), "NOT_FOUND", 404) from exc
    except Exception as exc:
        raise APIError(str(exc), "DELETE_FAILURE", 500) from exc

    return {"ok": True}


@faces_bp.route("/api/faces/unknowns/<id>", methods=["DELETE"])
def delete_unknown(id: str):
    engine = _deps.get("face_engine")
    if engine is None:
        return {"error": "SERVICE_UNAVAILABLE", "detail": "Face engine not available"}, 503

    try:
        engine.delete_unknown(id)
    except ValueError as exc:
        raise APIError(str(exc), "NOT_FOUND", 404) from exc
    except Exception as exc:
        raise APIError(str(exc), "DELETE_FAILURE", 500) from exc

    return {"ok": True}
