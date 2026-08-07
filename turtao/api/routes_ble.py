from flask import Blueprint, request

from turtao.api.errors import APIError

ble_bp = Blueprint("ble", __name__)

_deps: dict = {}


def inject_deps(**kwargs) -> None:
    _deps.update(kwargs)


@ble_bp.route("/api/ble/devices")
def ble_devices():
    # No BLE scanning exists on real hardware — the confirmed ESP32-S3
    # firmware has no BLE handling at all, and nothing on the Pi side scans
    # for nearby devices either. This never had a real data source.
    return {"error": "SERVICE_UNAVAILABLE", "detail": "BLE device scanning not implemented"}, 503


@ble_bp.route("/api/ble/register", methods=["POST"])
def ble_register():
    body = request.get_json(silent=True)
    if not body or "mac" not in body:
        raise APIError("mac field required", "VALIDATION_ERROR", 400)

    mac = body["mac"].strip()
    if not mac:
        raise APIError("mac must not be empty", "VALIDATION_ERROR", 400)

    s = _deps.get("settings")
    if s is not None:
        s.phone_registration = mac
        try:
            s.save()
        except Exception as exc:
            raise APIError(str(exc), "SAVE_FAILURE", 500) from exc

    return {"ok": True, "mac": mac}
