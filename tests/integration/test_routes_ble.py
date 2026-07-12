import json
import pytest


class TestBLERoutes:
    def test_get_devices_returns_list(self, client):
        resp = client.get("/api/ble/devices")
        assert resp.status_code == 200
        data = resp.get_json()
        assert isinstance(data, list)
        assert len(data) == 2
        assert data[0]["id"] == "dev1"
        assert data[0]["name"] == "JBL Go 3"
        assert data[0]["rssi"] == -45
        assert data[0]["owner"] is True

    def test_get_devices_structure(self, client):
        resp = client.get("/api/ble/devices")
        data = resp.get_json()
        for device in data:
            assert "id" in device
            assert "name" in device
            assert "rssi" in device
            assert "owner" in device

    def test_register_with_mac(self, client, mock_serial, mock_settings):
        mock_serial.open()
        resp = client.post("/api/ble/register", json={"mac": "AA:BB:CC:DD:EE:FF"})
        assert resp.status_code == 200
        assert resp.get_json() == {"ok": True, "mac": "AA:BB:CC:DD:EE:FF"}
        assert mock_settings.phone_registration == "AA:BB:CC:DD:EE:FF"
        payload = json.loads(mock_serial.written[0])
        assert payload == {"cmd": "ble_register", "mac": "AA:BB:CC:DD:EE:FF"}

    def test_register_missing_mac_returns_error(self, client):
        resp = client.post("/api/ble/register", json={})
        assert resp.status_code == 400
        assert resp.get_json()["error"] == "VALIDATION_ERROR"

    def test_register_empty_mac_returns_error(self, client):
        resp = client.post("/api/ble/register", json={"mac": ""})
        assert resp.status_code == 400
        assert resp.get_json()["error"] == "VALIDATION_ERROR"

    def test_register_no_body_returns_error(self, client):
        resp = client.post("/api/ble/register", json=None)
        assert resp.status_code == 400
