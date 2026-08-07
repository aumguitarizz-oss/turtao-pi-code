import json
import pytest


class TestModeRoutes:
    def test_get_mode_returns_current(self, client, mock_state):
        mock_state.mode = "GUARD"
        resp = client.get("/api/mode")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["mode"] == "GUARD"

    def test_get_default_mode_idle(self, client):
        resp = client.get("/api/mode")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["mode"] == "IDLE"

    def test_post_valid_mode_idle(self, client, mock_serial):
        resp = client.post("/api/mode", json={"mode": "IDLE"})
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["mode"] == "IDLE"

    def test_post_valid_mode_guard(self, client, mock_serial):
        resp = client.post("/api/mode", json={"mode": "GUARD"})
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["mode"] == "GUARD"

    def test_post_valid_mode_patrol(self, client, mock_serial):
        resp = client.post("/api/mode", json={"mode": "PATROL"})
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["mode"] == "PATROL"

    def test_post_valid_mode_case_insensitive(self, client, mock_serial):
        resp = client.post("/api/mode", json={"mode": "guard"})
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["mode"] == "GUARD"

    def test_post_invalid_mode_returns_validation_error(self, client):
        resp = client.post("/api/mode", json={"mode": "PLAY"})
        assert resp.status_code == 400
        data = resp.get_json()
        assert data["error"] == "VALIDATION_ERROR"

    def test_post_missing_mode_field_returns_validation_error(self, client):
        resp = client.post("/api/mode", json={})
        assert resp.status_code == 400
        data = resp.get_json()
        assert data["error"] == "VALIDATION_ERROR"

    def test_post_no_body_returns_validation_error(self, client):
        resp = client.post("/api/mode", json=None)
        assert resp.status_code == 400
        data = resp.get_json()
        assert data["error"] == "VALIDATION_ERROR"

    def test_post_non_idle_mode_sends_nothing_to_serial(self, client, mock_serial):
        # "mode" isn't a command the confirmed firmware recognizes — mode
        # is tracked Pi-side only. Only an IDLE transition sends anything
        # (a real move e-stop), covered separately.
        mock_serial.open()
        client.post("/api/mode", json={"mode": "PATROL"})
        assert mock_serial.written == []

    def test_state_updated_on_post(self, client, mock_state):
        client.post("/api/mode", json={"mode": "PATROL"})
        assert mock_state.mode == "PATROL"

    def test_switching_to_idle_sends_explicit_stop(self, client, mock_serial):
        client.post("/api/mode", json={"mode": "PATROL"})
        written_before = len(mock_serial.written)
        client.post("/api/mode", json={"mode": "IDLE"})
        new_payloads = [json.loads(w) for w in mock_serial.written[written_before:]]
        assert {"cmd": "move", "ml": 0.0, "mr": 0.0} in new_payloads
