import json
import logging
import threading
import time

from flask_sock import Sock

from turtao.state import Mode

logger = logging.getLogger(__name__)


class StatusBroadcaster:
    def __init__(self, state) -> None:
        self._clients: set = set()
        self._lock = threading.Lock()
        self._state = state

    def add_client(self, ws) -> None:
        with self._lock:
            self._clients.add(ws)

    def remove_client(self, ws) -> None:
        with self._lock:
            self._clients.discard(ws)

    def broadcast(self) -> None:
        payload = self._build_status()
        dead: list = []
        with self._lock:
            clients = list(self._clients)
        for ws in clients:
            try:
                ws.send(json.dumps(payload))
            except Exception:
                dead.append(ws)
        if dead:
            with self._lock:
                for ws in dead:
                    self._clients.discard(ws)

    def _build_status(self) -> dict:
        st = self._state
        ts = st.threat_state
        b = st.battery
        sd = st.sensor_data
        return {
            "event": "status",
            "data": {
                "connected": st.connected,
                "threat": {
                    "active": ts.active,
                    "face_crop": ts.face_crop if isinstance(ts.face_crop, str) else None,
                    "confidence": ts.confidence,
                    "timestamp": ts.timestamp,
                    "box": self._normalized_box(),
                    "name": ts.name or None,
                },
                "mode": st.mode.value if isinstance(st.mode, Mode) else st.mode,
                "battery": {
                    "percent": int(b.percent),
                    "voltage": b.voltage,
                    "current": b.current_ma,
                    "state": b.status.upper(),
                },
                "heading": st.heading,
                "tof": {
                    "fl": sd.tof_cm[0] if len(sd.tof_cm) > 0 else 0,
                    "fc": sd.tof_cm[1] if len(sd.tof_cm) > 1 else 0,
                    "fr": sd.tof_cm[2] if len(sd.tof_cm) > 2 else 0,
                },
                "latency_ms": st.latency_ms,
            },
        }

    def _normalized_box(self) -> list[float] | None:
        """The current summary face's box as [left, top, width, height],
        normalised 0..1 against the live frame's actual dimensions —
        matches the app's Rect.fromLTWH(...) contract directly, so the
        app never needs to know the camera's real pixel resolution."""
        box = self._state.threat_state.box
        frame = self._state.latest_frame
        if box is None or frame is None:
            return None
        try:
            h, w = frame.shape[:2]
        except (AttributeError, ValueError):
            return None
        if w <= 0 or h <= 0:
            return None
        left, top, right, bottom = box
        return [left / w, top / h, (right - left) / w, (bottom - top) / h]


_broadcaster: StatusBroadcaster | None = None
_broadcaster_lock = threading.Lock()


def get_broadcaster(state) -> StatusBroadcaster:
    """Return the process-wide StatusBroadcaster, creating it on first use.

    Shared between register_status_ws() and ws_broadcast_loop() so that
    clients registered on one are actually reached by broadcasts from the
    other, regardless of which is set up first.
    """
    global _broadcaster
    with _broadcaster_lock:
        if _broadcaster is None:
            _broadcaster = StatusBroadcaster(state)
        return _broadcaster


def handle_status_client(ws, broadcaster: StatusBroadcaster) -> None:
    """Run the receive loop for one connected /ws/status client.

    The client never sends anything, so a receive timeout is expected and
    must not be treated as a disconnect -- only an actual exception from
    receive() (e.g. ConnectionClosed) ends the loop.
    """
    broadcaster.add_client(ws)
    try:
        while True:
            try:
                ws.receive(timeout=2)
            except Exception:
                break
    finally:
        broadcaster.remove_client(ws)


def register_status_ws(sock: Sock, state) -> None:
    broadcaster = get_broadcaster(state)

    @sock.route("/ws/status")
    def status_ws(ws):
        handle_status_client(ws, broadcaster)


def ws_broadcast_loop(state) -> None:
    broadcaster = get_broadcaster(state)

    def _loop():
        while True:
            try:
                broadcaster.broadcast()
            except Exception:
                logger.exception("Broadcast error")
            time.sleep(2)

    t = threading.Thread(target=_loop, daemon=True, name="ws-broadcast")
    t.start()
