class TestEnvironmentRoutes:
    def test_get_environment_returns_correct_schema(self, client, mock_state):
        mock_state.sensor_data.temp_dht = 31.0
        mock_state.sensor_data.humidity = 60.0
        mock_state.sensor_data.gas_mq2 = 150
        mock_state.sensor_data.air_quality_mq135 = 90
        mock_state.sensor_data.sound_raw = 1871
        mock_state.sensor_data.motion = True
        mock_state.sensor_data.imu.accel_x = 1.2
        mock_state.sensor_data.imu.accel_y = -0.5
        mock_state.sensor_data.imu.accel_z = 0.8
        mock_state.sensor_data.imu.gyro_x = 0.1
        mock_state.sensor_data.imu.gyro_y = -0.2
        mock_state.sensor_data.imu.gyro_z = 0.3
        mock_state.sensor_data.tof_cm = [45, 120, 200, 300]

        resp = client.get("/api/environment")
        assert resp.status_code == 200
        data = resp.get_json()

        assert data["temp_dht"] == 31.0
        assert data["humidity"] == 60.0
        assert "pressure_hpa" not in data
        assert data["gas_mq2"] == 150
        assert data["air_quality_mq135"] == 90
        assert data["sound_raw"] == 1871
        assert data["motion"] is True
        assert data["imu"]["accel_x"] == 1.2
        assert data["imu"]["accel_y"] == -0.5
        assert data["imu"]["accel_z"] == 0.8
        assert data["imu"]["gyro_x"] == 0.1
        assert data["imu"]["gyro_y"] == -0.2
        assert data["imu"]["gyro_z"] == 0.3
        assert data["tof_cm"] == [45, 120, 200, 300]

    def test_get_environment_defaults(self, client):
        resp = client.get("/api/environment")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["temp_dht"] == 0.0
        assert data["humidity"] == 0.0
        assert data["motion"] is False
        assert data["tof_cm"] == [0, 0, 0, 0]

    def test_get_battery_is_unavailable(self, client):
        # No INA219 (or any battery monitoring hardware) on this build —
        # per requirements doc §3.3, must not fabricate a reading.
        resp = client.get("/api/battery")
        assert resp.status_code == 503
        data = resp.get_json()
        assert data["error"] == "SERVICE_UNAVAILABLE"

    def test_environment_tof_cm_structure(self, client, mock_state):
        mock_state.sensor_data.tof_cm = [10, 20, 30, 40]
        resp = client.get("/api/environment")
        data = resp.get_json()
        assert len(data["tof_cm"]) == 4
        assert data["tof_cm"] == [10, 20, 30, 40]

    def test_test_alert_route_emits_gas_and_temp_events(self, client, mock_state):
        resp = client.post("/api/sensors/test-alert")
        assert resp.status_code == 200
        assert resp.get_json() == {"ok": True}
        types = [e.type for e in mock_state.events]
        assert "gas_danger" in types
        assert "temp_danger" in types
