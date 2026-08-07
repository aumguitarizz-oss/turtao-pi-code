import pytest


class TestEnvironmentRoutes:
    def test_get_environment_returns_correct_schema(self, client, mock_state):
        mock_state.sensor_data.temp_inside_c = 31.0
        mock_state.sensor_data.temp_outside_c = 25.5
        mock_state.sensor_data.humidity_pct = 60
        mock_state.sensor_data.gas_mq2 = 150
        mock_state.sensor_data.air_quality_mq135 = 90
        mock_state.sensor_data.sound_level = 35.2
        mock_state.sensor_data.motion = True
        mock_state.sensor_data.orientation.pitch = 1.2
        mock_state.sensor_data.orientation.roll = -0.5
        mock_state.sensor_data.orientation.yaw = 0.8
        mock_state.sensor_data.tof_cm = [45, 120, 200, 300]

        resp = client.get("/api/environment")
        assert resp.status_code == 200
        data = resp.get_json()

        assert data["temp_inside_c"] == 31.0
        assert data["temp_outside_c"] == 25.5
        assert data["humidity_pct"] == 60
        assert "pressure_hpa" not in data
        assert data["gas_mq2"] == 150
        assert data["air_quality_mq135"] == 90
        assert data["sound_level"] == 35.2
        assert data["motion"] is True
        assert data["orientation"]["pitch"] == 1.2
        assert data["orientation"]["roll"] == -0.5
        assert data["orientation"]["yaw"] == 0.8
        assert data["tof_cm"] == [45, 120, 200, 300]

    def test_get_environment_defaults(self, client):
        resp = client.get("/api/environment")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["temp_inside_c"] == 0.0
        assert data["temp_outside_c"] == 0.0
        assert data["humidity_pct"] == 0
        assert data["motion"] is False
        assert data["tof_cm"] == [0, 0, 0, 0]

    def test_get_battery_returns_correct_schema(self, client, mock_state):
        mock_state.battery.voltage = 12.345
        mock_state.battery.current_ma = 150
        mock_state.battery.percent = 85.5
        mock_state.battery.status = "charging"

        resp = client.get("/api/battery")
        assert resp.status_code == 200
        data = resp.get_json()

        assert data["voltage"] == 12.345
        assert data["current_ma"] == 150
        assert data["percent"] == 85.5
        assert data["status"] == "charging"

    def test_get_battery_defaults(self, client):
        resp = client.get("/api/battery")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["voltage"] == 0.0
        assert data["current_ma"] == 0
        assert data["percent"] == 0.0
        assert data["status"] == "discharging"

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
