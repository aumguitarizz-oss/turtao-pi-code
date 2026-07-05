import pytest


class TestControlRoute:
    def test_post_control_sends_commands(self, client, mock_serial):
        mock_serial.open()
        resp = client.post("/api/control", json={"speed": 0.5, "safe_mode": True})
        assert resp.status_code == 200
        assert resp.get_json() == {"ok": True}
        assert len(mock_serial.written) > 0
        written = b" ".join(mock_serial.written)
        assert b"SPD 0.5" in written
        assert b"SAFE 1" in written

    def test_post_control_speed_only(self, client, mock_serial):
        mock_serial.open()
        resp = client.post("/api/control", json={"speed": 0.3})
        assert resp.status_code == 200
        assert mock_serial.written[0] == b"SPD 0.3\n"

    def test_post_control_nerf_flag(self, client, mock_serial):
        mock_serial.open()
        client.post("/api/control", json={"nerf": True})
        assert mock_serial.written[0] == b"NERF 1\n"

    def test_post_control_pan_tilt(self, client, mock_serial):
        mock_serial.open()
        client.post("/api/control", json={"pan": 120, "tilt": 60})
        written = b" ".join(mock_serial.written)
        assert b"PAN 120" in written
        assert b"TILT 60" in written

    def test_estop_sent_when_speed_zero_and_safe_mode(self, client, mock_serial, mock_state):
        mock_serial.open()
        resp = client.post("/api/control", json={"speed": 0, "safe_mode": True})
        assert resp.status_code == 200
        assert resp.get_json() == {"ok": True}
        assert len(mock_serial.written) == 1
        assert mock_serial.written[0] == b'{"cmd":"estop"}\n'

    def test_estop_sets_mode_idle(self, client, mock_serial, mock_state):
        mock_state.mode = "PATROL"
        client.post("/api/control", json={"speed": 0, "safe_mode": True})
        assert mock_state.mode == "IDLE"

    def test_estop_not_sent_when_speed_nonzero(self, client, mock_serial):
        mock_serial.open()
        client.post("/api/control", json={"speed": 0.5, "safe_mode": True})
        assert not any(b"estop" in w for w in mock_serial.written)

    def test_post_control_no_body_returns_error(self, client):
        resp = client.post("/api/control", json=None)
        assert resp.status_code == 400

    def test_post_control_invalid_speed_type(self, client):
        resp = client.post("/api/control", json={"speed": "fast"})
        assert resp.status_code == 400
        assert resp.get_json()["error"] == "VALIDATION_ERROR"


class TestMoveRoute:
    def test_post_move_sends_command(self, client, mock_serial):
        mock_serial.open()
        resp = client.post("/api/move", json={"ml": 0.5, "mr": 0.3})
        assert resp.status_code == 200
        assert resp.get_json() == {"ok": True}
        assert mock_serial.written[0] == b"MOVE 0.50 0.30\n"

    def test_move_clamps_ml_to_max(self, client, mock_serial):
        mock_serial.open()
        client.post("/api/move", json={"ml": 2.0, "mr": 0.0})
        assert b"MOVE 0.80" in mock_serial.written[0]

    def test_move_clamps_mr_to_max(self, client, mock_serial):
        mock_serial.open()
        client.post("/api/move", json={"ml": 0.0, "mr": 1.5})
        assert b"0.80\n" in mock_serial.written[0]

    def test_move_clamps_ml_to_min(self, client, mock_serial):
        mock_serial.open()
        client.post("/api/move", json={"ml": -2.0, "mr": 0.0})
        assert b"MOVE -0.80" in mock_serial.written[0]

    def test_move_clamps_mr_to_min(self, client, mock_serial):
        mock_serial.open()
        client.post("/api/move", json={"ml": 0.0, "mr": -1.5})
        assert b"-0.80\n" in mock_serial.written[0]

    def test_move_missing_ml_returns_error(self, client):
        resp = client.post("/api/move", json={"mr": 0.5})
        assert resp.status_code == 400

    def test_move_missing_mr_returns_error(self, client):
        resp = client.post("/api/move", json={"ml": 0.5})
        assert resp.status_code == 400

    def test_move_non_numeric_value_returns_error(self, client):
        resp = client.post("/api/move", json={"ml": "abc", "mr": 0.5})
        assert resp.status_code == 400
        assert resp.get_json()["error"] == "VALIDATION_ERROR"

    def test_move_no_body_returns_error(self, client):
        resp = client.post("/api/move", json=None)
        assert resp.status_code == 400
