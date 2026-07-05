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

    def test_post_sends_mode_to_serial(self, client, mock_serial):
        mock_serial.open()
        client.post("/api/mode", json={"mode": "PATROL"})
        assert len(mock_serial.written) >= 1
        written = mock_serial.written[0]
        assert b"MODE PATROL" in written

    def test_state_updated_on_post(self, client, mock_state):
        client.post("/api/mode", json={"mode": "PATROL"})
        assert mock_state.mode == "PATROL"
