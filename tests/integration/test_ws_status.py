import json
import pytest
from unittest.mock import MagicMock
from turtao.api.ws_status import StatusBroadcaster
from tests.integration.conftest import MockState


class TestStatusBroadcaster:
    def test_build_status_contains_expected_keys(self, mock_state):
        mock_state.mode = "GUARD"
        mock_state.connected = True
        mock_state.threat_state.active = True
        mock_state.threat_state.confidence = 0.87
        mock_state.battery.percent = 75.0
        mock_state.battery.voltage = 12.3
        mock_state.heading = 90
        mock_state.sensor_data.tof_cm = [50, 120, 200, 300]
        mock_state.latency_ms = 15

        broadcaster = StatusBroadcaster(mock_state)
        payload = broadcaster._build_status()

        assert payload["event"] == "status"
        data = payload["data"]
        assert data["connected"] is True
        assert data["mode"] == "GUARD"
        assert data["threat"]["active"] is True
        assert data["threat"]["confidence"] == 0.87
        assert data["battery"]["percent"] == 75.0
        assert data["battery"]["voltage"] == 12.3
        assert data["heading"] == 90
        assert data["tof"]["fl"] == 50
        assert data["tof"]["fc"] == 120
        assert data["tof"]["fr"] == 200
        assert data["latency_ms"] == 15

    def test_build_status_defaults(self, mock_state):
        broadcaster = StatusBroadcaster(mock_state)
        payload = broadcaster._build_status()
        data = payload["data"]
        assert data["connected"] is False
        assert data["mode"] == "IDLE"
        assert data["threat"]["active"] is False
        assert data["threat"]["confidence"] == 0.0
        assert data["battery"]["percent"] == 0.0
        assert data["battery"]["voltage"] == 0.0
        assert data["heading"] == 0
        assert data["tof"]["fl"] == 0
        assert data["tof"]["fc"] == 0
        assert data["tof"]["fr"] == 0
        assert data["latency_ms"] == 0

    def test_add_and_remove_client(self, mock_state):
        broadcaster = StatusBroadcaster(mock_state)
        mock_ws = object()
        broadcaster.add_client(mock_ws)
        assert mock_ws in broadcaster._clients
        broadcaster.remove_client(mock_ws)
        assert mock_ws not in broadcaster._clients

    def test_remove_nonexistent_client_does_not_raise(self, mock_state):
        broadcaster = StatusBroadcaster(mock_state)
        broadcaster.remove_client(object())

    def test_add_multiple_clients(self, mock_state):
        broadcaster = StatusBroadcaster(mock_state)
        ws1, ws2 = object(), object()
        broadcaster.add_client(ws1)
        broadcaster.add_client(ws2)
        assert len(broadcaster._clients) == 2

    def test_broadcast_to_multiple_clients(self, mock_state):
        broadcaster = StatusBroadcaster(mock_state)
        mock_ws1 = MagicMock()
        mock_ws2 = MagicMock()
        broadcaster.add_client(mock_ws1)
        broadcaster.add_client(mock_ws2)
        broadcaster.broadcast()
        assert mock_ws1.send.called
        assert mock_ws2.send.called
        sent_data = json.loads(mock_ws1.send.call_args[0][0])
        assert sent_data["event"] == "status"

    def test_broadcast_removes_dead_clients(self, mock_state):
        broadcaster = StatusBroadcaster(mock_state)
        dead_ws = MagicMock()
        dead_ws.send.side_effect = Exception("connection lost")
        live_ws = MagicMock()
        broadcaster.add_client(dead_ws)
        broadcaster.add_client(live_ws)
        broadcaster.broadcast()
        assert dead_ws not in broadcaster._clients
        assert live_ws in broadcaster._clients

    def test_broadcast_empty_clients(self, mock_state):
        broadcaster = StatusBroadcaster(mock_state)
        broadcaster.broadcast()


class TestStatusWebSocket:
    def test_websocket_route_registered(self, app, mock_state):
        # Flask's synchronous test client has no .websocket() support (that's
        # Quart-only), so this only verifies wiring: the route is registered.
        # Client lifecycle behavior (broadcast delivery, timeout resilience)
        # is covered directly by TestHandleStatusClient below.
        from flask_sock import Sock
        from turtao.api.ws_status import register_status_ws
        sock = Sock(app)
        register_status_ws(sock, mock_state)
        assert "/ws/status" in [rule.rule for rule in app.url_map.iter_rules()]


class TestHandleStatusClient:
    def test_registered_client_receives_broadcast(self):
        from turtao.state import AppState

        state = AppState()
        broadcaster = StatusBroadcaster(state)
        ws = MagicMock()

        broadcaster.add_client(ws)
        broadcaster.broadcast()

        assert ws.send.called
        sent = json.loads(ws.send.call_args[0][0])
        assert sent["event"] == "status"

    def test_client_stays_connected_past_receive_timeout(self):
        from turtao.state import AppState
        from turtao.api.ws_status import handle_status_client

        state = AppState()
        broadcaster = StatusBroadcaster(state)
        ws = MagicMock()
        # First few calls simulate flask-sock's receive() timing out (returns
        # None) since the client never sends anything; the last call
        # simulates the client actually disconnecting.
        ws.receive.side_effect = [None, None, None, Exception("closed")]

        handle_status_client(ws, broadcaster)

        assert ws.receive.call_count == 4
        assert ws not in broadcaster._clients
