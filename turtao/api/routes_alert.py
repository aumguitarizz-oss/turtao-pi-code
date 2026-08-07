import logging

from flask import Blueprint

from turtao.state import ThreatLabel

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
    # threat_state.active means "a face is currently resolved" (set True for
    # both SAFE/enrolled and THREAT/unknown matches) — threat_label is the
    # field that actually distinguishes the two. Using .active here reported
    # a recognized, enrolled owner as an intruder.
    return {
        "threat": st.threat_label == ThreatLabel.THREAT,
        "confidence": st.threat_state.confidence,
    }
