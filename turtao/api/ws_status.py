import json
import threading
import time
import logging
from flask_sock import Sock

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
                    "face_crop": ts.face_crop,
                    "confidence": ts.confidence,
                    "timestamp": ts.timestamp,
                },
                "mode": st.mode.value if hasattr(st.mode, "value") else st.mode,
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


def register_status_ws(sock: Sock, state) -> None:
    broadcaster = StatusBroadcaster(state)

    @sock.route("/ws/status")
    def status_ws(ws):
        broadcaster.add_client(ws)
        try:
            while True:
                msg = ws.receive(timeout=2)
                if msg is None:
                    break
        except Exception:
            pass
        finally:
            broadcaster.remove_client(ws)


def ws_broadcast_loop(state) -> None:
    def _loop():
        broadcaster = StatusBroadcaster(state)
        while True:
            try:
                broadcaster.broadcast()
            except Exception:
                logger.exception("Broadcast error")
            time.sleep(2)

    t = threading.Thread(target=_loop, daemon=True, name="ws-broadcast")
    t.start()
