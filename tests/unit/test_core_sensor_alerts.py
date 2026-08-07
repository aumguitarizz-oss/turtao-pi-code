import pytest

from turtao.config import AppConfig, Settings
from turtao.core import TurtaoCore
from turtao.hardware.mocks import MockCamera, MockSerialLink
from turtao.state import AppState


@pytest.fixture
def core(tmp_path):
    state = AppState()
    settings = Settings(
        gas_threshold_low=50.0,
        gas_threshold_high=400.0,
        temp_threshold_low=0.0,
        temp_threshold_high=45.0,
    )
    c = TurtaoCore(
        config=AppConfig(),
        settings=settings,
        state=state,
        serial_link=MockSerialLink(),
        camera=MockCamera(),
        face_data_dir=tmp_path / "face_data",
    )
    return c


class TestCheckSensorAlerts:
    def test_gas_within_range_emits_nothing(self, core):
        core.state.sensor_data.gas_mq2 = 200
        core.state.sensor_data.temp_dht = 22.0
        core._check_sensor_alerts()
        assert len(core.state.events) == 0

    def test_gas_above_high_threshold_emits_gas_danger(self, core):
        core.state.sensor_data.gas_mq2 = 500
        core.state.sensor_data.temp_dht = 22.0
        core._check_sensor_alerts()
        types = [e.type for e in core.state.events]
        assert types == ["gas_danger"]

    def test_gas_below_low_threshold_emits_gas_danger(self, core):
        core.state.sensor_data.gas_mq2 = 10
        core.state.sensor_data.temp_dht = 22.0
        core._check_sensor_alerts()
        types = [e.type for e in core.state.events]
        assert types == ["gas_danger"]

    def test_temp_out_of_range_emits_temp_danger(self, core):
        core.state.sensor_data.gas_mq2 = 200
        core.state.sensor_data.temp_dht = 60.0
        core._check_sensor_alerts()
        types = [e.type for e in core.state.events]
        assert types == ["temp_danger"]

    def test_sustained_out_of_range_emits_only_one_event(self, core):
        # Rising-edge guard: repeated polls while still out of range must
        # not spam an event every second.
        core.state.sensor_data.gas_mq2 = 500
        core.state.sensor_data.temp_dht = 22.0
        core._check_sensor_alerts()
        core._check_sensor_alerts()
        core._check_sensor_alerts()
        assert len(core.state.events) == 1

    def test_alert_re_fires_after_returning_to_normal(self, core):
        core.state.sensor_data.gas_mq2 = 500
        core.state.sensor_data.temp_dht = 22.0
        core._check_sensor_alerts()  # fires
        core.state.sensor_data.gas_mq2 = 200
        core._check_sensor_alerts()  # back to normal, clears the guard
        core.state.sensor_data.gas_mq2 = 500
        core._check_sensor_alerts()  # out of range again, fires again
        types = [e.type for e in core.state.events]
        assert types == ["gas_danger", "gas_danger"]
