import logging

from flask import Blueprint

logger = logging.getLogger(__name__)

alert_bp = Blueprint("alert", __name__)

_deps: dict = {}


def inject_deps(**kwargs) -> None:
    _deps.update(kwargs)


@alert_bp.route("/api/alert")
def get_alert():
    st = _deps.get("state")
    if st is None:
        return {"error": "SERVICE_UNAVAILABLE", "detail": "State not available"}, 503
    return {
        "threat": st.threat_state.active,
        "confidence": st.threat_state.confidence,
    }
