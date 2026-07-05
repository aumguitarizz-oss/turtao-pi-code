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
        tof = getattr(st, "tof_cm", [0, 0, 0, 0])
        return {
            "event": "status",
            "data": {
                "connected": getattr(st, "connected", False),
                "threat": {
                    "active": getattr(st, "threat_active", False),
                    "face_crop": getattr(st, "threat_face_crop", None),
                    "confidence": getattr(st, "threat_confidence", 0.0),
                    "timestamp": getattr(st, "threat_timestamp", None),
                },
                "mode": getattr(st, "mode", "IDLE"),
                "battery": {
                    "percent": getattr(st, "battery_percent", 0),
                    "voltage": getattr(st, "battery_voltage", 0.0),
                    "current": getattr(st, "battery_current", 0),
                    "state": getattr(st, "battery_state", "discharging"),
                },
                "heading": getattr(st, "heading", 0),
                "tof": {
                    "fl": tof[0] if len(tof) > 0 else 0,
                    "fc": tof[1] if len(tof) > 1 else 0,
                    "fr": tof[2] if len(tof) > 2 else 0,
                },
                "latency_ms": getattr(st, "latency_ms", 0),
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
