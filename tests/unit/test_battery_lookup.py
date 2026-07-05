import pytest
from turtao.battery.battery_monitor import voltage_to_percent, get_battery_state, VOLTAGE_TABLE


class TestVoltageToPercent:
    def test_above_max_returns_100(self):
        assert voltage_to_percent(13.0) == 100.0
        assert voltage_to_percent(12.7) == 100.0
        assert voltage_to_percent(12.6) == 100.0

    def test_below_min_returns_0(self):
        assert voltage_to_percent(9.0) == 0.0
        assert voltage_to_percent(9.8) == 0.0
        assert voltage_to_percent(9.9) == 0.0

    def test_exact_table_values(self):
        assert voltage_to_percent(12.6) == 100.0
        assert voltage_to_percent(12.2) == 90.0
        assert voltage_to_percent(11.8) == 75.0
        assert voltage_to_percent(11.5) == 60.0
        assert voltage_to_percent(11.1) == 50.0
        assert voltage_to_percent(10.8) == 35.0
        assert voltage_to_percent(10.5) == 20.0
        assert voltage_to_percent(10.2) == 10.0
        assert voltage_to_percent(9.9) == 0.0

    def test_interpolation_between_points(self):
        mid = (VOLTAGE_TABLE[0][0] + VOLTAGE_TABLE[1][0]) / 2
        expected_percent = (VOLTAGE_TABLE[0][1] + VOLTAGE_TABLE[1][1]) / 2
        result = voltage_to_percent(mid)
        assert result == pytest.approx(expected_percent, abs=0.5)

    def test_monotonic_decreasing(self):
        test_voltages = [v for v, _ in VOLTAGE_TABLE]
        percentages = [voltage_to_percent(v) for v in test_voltages]
        for i in range(len(percentages) - 1):
            assert percentages[i] >= percentages[i + 1]

    def test_interpolation_precision_one_decimal(self):
        result = voltage_to_percent(12.0)
        s = str(result)
        assert "." in s
        decimal_part = s.split(".")[1]
        assert len(decimal_part) <= 1

    def test_mid_range_value(self):
        pct = voltage_to_percent(11.3)
        assert 40.0 <= pct <= 60.0

    def test_near_maximum(self):
        pct = voltage_to_percent(12.5)
        assert 90.0 <= pct <= 100.0

    def test_near_minimum(self):
        pct = voltage_to_percent(10.0)
        assert 0.0 <= pct <= 15.0


class TestGetBatteryState:
    def test_charging_state(self):
        result = get_battery_state(12.345, 85.2, is_charging=True)
        assert result["voltage"] == 12.35
        assert result["current_ma"] == 0
        assert result["percent"] == 85.2
        assert result["status"] == "charging"

    def test_discharging_state(self):
        result = get_battery_state(11.111, 50.0, is_charging=False)
        assert result["voltage"] == 11.11
        assert result["status"] == "discharging"

    def test_voltage_rounding(self):
        result = get_battery_state(11.8888, 75.0, is_charging=False)
        assert result["voltage"] == 11.89

    def test_percent_rounding(self):
        result = get_battery_state(12.0, 92.67, is_charging=False)
        assert result["percent"] == 92.7
