class TestBLERoutes:
    def test_get_devices_is_unavailable(self, client):
        # No BLE scanning exists on real hardware — the confirmed firmware
        # has no BLE handling, and nothing on the Pi scans for devices
        # either. This never had a real data source.
        resp = client.get("/api/ble/devices")
        assert resp.status_code == 503
        assert resp.get_json()["error"] == "SERVICE_UNAVAILABLE"

    def test_register_with_mac(self, client, mock_serial, mock_settings):
        mock_serial.open()
        resp = client.post("/api/ble/register", json={"mac": "AA:BB:CC:DD:EE:FF"})
        assert resp.status_code == 200
        assert resp.get_json() == {"ok": True, "mac": "AA:BB:CC:DD:EE:FF"}
        assert mock_settings.phone_registration == "AA:BB:CC:DD:EE:FF"
        # "ble_register" isn't a command the confirmed firmware recognizes —
        # registration is Pi-side-only (settings persistence), no serial
        # write should happen.
        assert mock_serial.written == []

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
