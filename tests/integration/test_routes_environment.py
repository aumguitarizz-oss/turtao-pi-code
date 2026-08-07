class TestEnvironmentRoutes:
    def test_get_environment_returns_correct_schema(self, client, mock_state):
        mock_state.sensor_data.temp_dht = 31.0
        mock_state.sensor_data.humidity = 60.0
        mock_state.sensor_data.gas_mq2 = 150
        mock_state.sensor_data.tof_front = 300

        resp = client.get("/api/environment")
        assert resp.status_code == 200
        data = resp.get_json()

        assert data["temp_dht"] == 31.0
        assert data["humidity"] == 60.0
        assert "pressure_hpa" not in data
        assert data["gas_mq2"] == 150
        assert data["tof_front"] == 300
        assert "air_quality_mq135" not in data
        assert "motion" not in data
        assert "imu" not in data

    def test_get_environment_defaults(self, client):
        resp = client.get("/api/environment")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["temp_dht"] == 0.0
        assert data["humidity"] == 0.0
        assert data["tof_front"] is None

    def test_get_battery_is_unavailable(self, client):
        # No INA219 (or any battery monitoring hardware) on this build —
        # per requirements doc §3.3, must not fabricate a reading.
        resp = client.get("/api/battery")
        assert resp.status_code == 503
        data = resp.get_json()
        assert data["error"] == "SERVICE_UNAVAILABLE"

    def test_environment_tof_front_null_when_out_of_range(self, client, mock_state):
        # The .ino reports null (not 0) when the ToF is out of range or
        # sees no target -- must round-trip as null, not a fabricated 0.
        mock_state.sensor_data.tof_front = None
        resp = client.get("/api/environment")
        data = resp.get_json()
        assert data["tof_front"] is None

    def test_test_alert_route_emits_gas_and_temp_events(self, client, mock_state):
        resp = client.post("/api/sensors/test-alert")
        assert resp.status_code == 200
        assert resp.get_json() == {"ok": True}
        types = [e.type for e in mock_state.events]
        assert "gas_danger" in types
        assert "temp_danger" in types
