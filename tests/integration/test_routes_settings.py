import pytest


class TestSettingsRoutes:
    def test_get_settings_returns_full(self, client, mock_settings):
        mock_settings.hostname = "testbot"
        mock_settings.speed = 0.5
        resp = client.get("/api/settings")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["hostname"] == "testbot"
        assert data["speed"] == 0.5
        assert data["ble_proximity_enabled"] is True
        assert data["face_tolerance"] == 0.52
        assert data["safe_mode"] is False

    def test_post_partial_update_changes_only_specified(self, client, mock_settings):
        resp = client.post("/api/settings", json={"speed": 0.3})
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["speed"] == 0.3
        assert data["hostname"] == ""
        assert data["face_tolerance"] == 0.52

    def test_post_multiple_fields(self, client, mock_settings):
        resp = client.post("/api/settings", json={
            "hostname": "newbot",
            "safe_mode": True,
            "auto_flashbang": True,
        })
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["hostname"] == "newbot"
        assert data["safe_mode"] is True
        assert data["auto_flashbang"] is True

    def test_post_update_nested_toggles(self, client, mock_settings):
        resp = client.post("/api/settings", json={
            "tts_event_toggles": {"threat": False, "low_battery": False}
        })
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["tts_event_toggles"]["threat"] is False
        assert data["tts_event_toggles"]["low_battery"] is False
        assert data["tts_event_toggles"]["intruder"] is True

    def test_post_full_settings_object_with_nested_dicts(self, client, mock_settings):
        body = mock_settings.asdict()
        body["hostname"] = "fullbot"
        body["speed"] = 0.4
        body["tts_event_toggles"] = {"threat": False, "low_battery": True, "intruder": False}
        body["notifications"] = {
            "threat": False,
            "gas_danger": False,
            "low_battery": True,
            "tamper": True,
            "connection_lost": False,
        }
        resp = client.post("/api/settings", json=body)
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["hostname"] == "fullbot"
        assert data["speed"] == 0.4
        assert data["tts_event_toggles"] == body["tts_event_toggles"]
        assert data["notifications"] == body["notifications"]

    def test_post_invalid_field_returns_validation_error(self, client):
        resp = client.post("/api/settings", json={"nonexistent_field": 42})
        assert resp.status_code == 400
        data = resp.get_json()
        assert data["error"] == "VALIDATION_ERROR"

    def test_post_no_body_returns_error(self, client):
        resp = client.post("/api/settings", json=None)
        assert resp.status_code == 400
        assert resp.get_json()["error"] == "VALIDATION_ERROR"

    def test_post_non_scalar_type_returns_error(self, client):
        resp = client.post("/api/settings", json={"speed": {"nested": "dict"}})
        assert resp.status_code == 400
        assert resp.get_json()["error"] == "VALIDATION_ERROR"

    def test_test_voice_endpoint(self, client, mock_tts):
        resp = client.post("/api/settings/test-voice")
        assert resp.status_code == 200
        assert resp.get_json() == {"ok": True}
