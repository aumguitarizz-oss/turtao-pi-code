import json
import pytest


class TestControlRoute:
    def test_pan_tilt_zero_centers_maps_to_servo_ninety(self, client, mock_serial):
        mock_serial.open()
        resp = client.post("/api/control", json={"pan": 0, "tilt": 0})
        assert resp.status_code == 200
        payload = json.loads(mock_serial.written[-1])
        assert payload["pan"] == 90
        assert payload["tilt"] == 90

    def test_pan_tilt_clamped_to_servo_range(self, client, mock_serial):
        mock_serial.open()
        client.post("/api/control", json={"pan": -90, "tilt": 90})
        payload = json.loads(mock_serial.written[-1])
        assert payload["pan"] == 5
        assert payload["tilt"] == 120

    def test_post_control_sends_commands(self, client, mock_serial, mock_settings):
        # A plain speed/safe_mode update (no estop, no pan/tilt) is only
        # persisted to settings; the route sends nothing to the ESP32 for it.
        mock_serial.open()
        resp = client.post("/api/control", json={"speed": 0.5, "safe_mode": True})
        assert resp.status_code == 200
        assert resp.get_json() == {"ok": True}
        assert mock_settings.speed == 0.5
        assert mock_settings.safe_mode is True
        assert mock_serial.written == []

    def test_post_control_speed_only(self, client, mock_serial, mock_settings):
        mock_serial.open()
        resp = client.post("/api/control", json={"speed": 0.3})
        assert resp.status_code == 200
        assert mock_settings.speed == 0.3
        assert mock_serial.written == []

    def test_post_control_nerf_flag(self, client, mock_serial):
        # "nerf" is accepted in the body but not currently wired to any
        # behavior in the route -- confirm it's a harmless no-op.
        mock_serial.open()
        resp = client.post("/api/control", json={"nerf": True})
        assert resp.status_code == 200
        assert resp.get_json() == {"ok": True}
        assert mock_serial.written == []

    def test_post_control_pan_tilt(self, client, mock_serial):
        mock_serial.open()
        client.post("/api/control", json={"pan": 120, "tilt": 60})
        payload = json.loads(mock_serial.written[-1])
        assert payload["cmd"] == "pan_tilt"
        # 0-centered API degrees shift by +90 into 90-centered servo space,
        # then clamp to [PAN_MIN, PAN_MAX] / [TILT_MIN, TILT_MAX].
        assert payload["pan"] == 175  # 120 + 90 = 210, clamped to PAN_MAX
        assert payload["tilt"] == 120  # 60 + 90 = 150, clamped to TILT_MAX

    def test_estop_sent_when_speed_zero_and_safe_mode(self, client, mock_serial, mock_state):
        mock_serial.open()
        resp = client.post("/api/control", json={"speed": 0, "safe_mode": True})
        assert resp.status_code == 200
        assert resp.get_json() == {"ok": True}
        assert len(mock_serial.written) == 1
        assert json.loads(mock_serial.written[0]) == {"cmd": "estop"}

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
    def test_post_move_sends_command(self, client, mock_serial, mock_settings):
        # /api/move scales ml/mr by the current settings.speed multiplier
        # (mock_settings.speed defaults to 0.8).
        mock_serial.open()
        resp = client.post("/api/move", json={"ml": 0.5, "mr": 0.3})
        assert resp.status_code == 200
        assert resp.get_json() == {"ok": True}
        expected_ml = round(0.5 * mock_settings.speed, 2)
        expected_mr = round(0.3 * mock_settings.speed, 2)
        assert json.loads(mock_serial.written[0]) == {"cmd": "move", "ml": expected_ml, "mr": expected_mr}

    def test_move_clamps_ml_to_max(self, client, mock_serial):
        mock_serial.open()
        client.post("/api/move", json={"ml": 2.0, "mr": 0.0})
        assert json.loads(mock_serial.written[0])["ml"] == 0.8

    def test_move_clamps_mr_to_max(self, client, mock_serial):
        mock_serial.open()
        client.post("/api/move", json={"ml": 0.0, "mr": 1.5})
        assert json.loads(mock_serial.written[0])["mr"] == 0.8

    def test_move_clamps_ml_to_min(self, client, mock_serial):
        mock_serial.open()
        client.post("/api/move", json={"ml": -2.0, "mr": 0.0})
        assert json.loads(mock_serial.written[0])["ml"] == -0.8

    def test_move_clamps_mr_to_min(self, client, mock_serial):
        mock_serial.open()
        client.post("/api/move", json={"ml": 0.0, "mr": -1.5})
        assert json.loads(mock_serial.written[0])["mr"] == -0.8

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
