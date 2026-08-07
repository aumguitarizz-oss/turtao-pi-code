import json
import logging
import threading
import time
from datetime import datetime, timezone

from flask_sock import Sock

from turtao.state import Mode, ThreatLabel

logger = logging.getLogger(__name__)


class StatusBroadcaster:
    def __init__(self, state) -> None:
        self._clients: set = set()
        self._lock = threading.Lock()
        self._state = state
        self._broadcast_count = 0

    def add_client(self, ws) -> None:
        with self._lock:
            self._clients.add(ws)

    def remove_client(self, ws) -> None:
        with self._lock:
            self._clients.discard(ws)

    def broadcast(self) -> None:
        payload = self._build_status()
        self._broadcast_count += 1
        with self._lock:
            client_count = len(self._clients)
            clients = list(self._clients)
        # Time-based (every ~10s at the caller's 2s cadence), not
        # change-based, so it can't scroll out of a `tail` the way an
        # earlier throttled-diagnostic attempt did (see HANDOFF.md) — the
        # box/persons data is populated far upstream of this broadcast, so
        # this pins down whether a "box doesn't show" report is a transport
        # problem (clients=0, or box/persons present but nothing arrives)
        # or a data problem (latest_frame is None / box stays null here).
        if self._broadcast_count % 5 == 0:
            data = payload["data"]
            logger.info(
                "DIAGNOSTIC: ws broadcast — clients=%d latest_frame=%s "
                "persons=%d threat_active=%s box=%s",
                client_count,
                "present" if self._state.latest_frame is not None else "None",
                len(data["persons"]),
                data["threat"]["active"],
                data["threat"]["box"],
            )
        dead: list = []
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
        # threat_label lives on AppState directly while box/name/active live
        # on the nested threat_state, written as separate statements inside
        # the same `with self.state:` block on the writer side (core.py /
        # face_recognition_engine.py). Reading them here without holding the
        # same lock let a broadcast interleave mid-write and pair a fresh
        # box/name with a stale label (or vice versa) — hold the lock for
        # the whole read so the snapshot is consistent.
        with st:
            ts = st.threat_state
            b = st.battery
            sd = st.sensor_data
            return {
                "event": "status",
                "data": {
                    "connected": st.connected,
                    "threat": {
                        "active": ts.active,
                        # active means "a face is currently resolved" (true for
                        # both SAFE/enrolled and THREAT/unknown matches) — label
                        # is what actually distinguishes them. The app uses this
                        # to color the box/banner instead of treating every
                        # resolved face as an intruder.
                        "label": st.threat_label.value
                        if isinstance(st.threat_label, ThreatLabel)
                        else st.threat_label,
                        "face_crop": ts.face_crop if isinstance(ts.face_crop, str) else None,
                        "confidence": ts.confidence,
                        "timestamp": self._iso_timestamp(ts.timestamp),
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
                    "tof_front": sd.tof_front,
                    "latency_ms": st.latency_ms,
                    "persons": self._normalized_persons(),
                },
            }

    @staticmethod
    def _iso_timestamp(epoch: float | None) -> str | None:
        """ThreatState.timestamp is a raw time.time() float; the app's
        WsThreat.fromJson casts this field as a String and feeds it to
        DateTime.tryParse, so it must go over the wire as ISO-8601, not a
        bare number — sending the float directly throws a type-cast
        exception inside the app's WS message handler with no try/catch
        around it, permanently killing the status stream for that client."""
        if epoch is None:
            return None
        return datetime.fromtimestamp(epoch, tz=timezone.utc).isoformat()

    def _normalized_persons(self) -> list[dict]:
        """YOLO-tracked persons this frame, each box as [left, top, width,
        height] normalised 0..1 against the live frame — same contract as
        threat.box. Always computed/published regardless of --gui; it's
        the app's toggle, not the Pi's, that decides whether to draw it."""
        frame = self._state.latest_frame
        if frame is None:
            return []
        try:
            h, w = frame.shape[:2]
        except (AttributeError, ValueError):
            return []
        if w <= 0 or h <= 0:
            return []
        alarmed_ids = getattr(self._state, "alarmed_person_ids", set())
        out = []
        for p in getattr(self._state, "latest_persons", []):
            if p.get("class_name") != "Person":
                continue
            left, top, right, bottom = p["bbox"]
            tracker_id = p.get("tracker_id", -1)
            out.append({
                "box": [left / w, top / h, (right - left) / w, (bottom - top) / h],
                "tracker_id": tracker_id,
                # true once this tracker_id has been unresolved (no
                # matching face) for LoiterMonitor's 10s alarm threshold —
                # distinct from threat.label, which only reflects an
                # actually-resolved face match.
                "alarm": tracker_id in alarmed_ids,
            })
        return out

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
