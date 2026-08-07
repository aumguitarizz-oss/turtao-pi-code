import pytest

from turtao.config import AppConfig, Settings
from turtao.core import TurtaoCore
from turtao.hardware.mocks import MockCamera, MockSerialLink
from turtao.serial_link.protocol import decode_payload, validate_payload
from turtao.state import AppState

# Exact sample payload matching the stripped-build ESP32-S3 firmware's
# sendSensorPayload() field-for-field — see
# inos/turtao_esp32_firmware1_copy_20260807222228.ino.
GOLDEN_PAYLOAD = (
    '{"tof_front": 340, "gas_mq2": 0.74, "temp_dht": 22.2, "humidity": 74.2}'
)


@pytest.fixture
def core(tmp_path):
    state = AppState()
    c = TurtaoCore(
        config=AppConfig(),
        settings=Settings(),
        state=state,
        serial_link=MockSerialLink(),
        camera=MockCamera(),
        face_data_dir=tmp_path / "face_data",
    )
    return c


class TestGoldenSensorPayload:
    def test_decodes_and_validates(self):
        success, data = decode_payload(GOLDEN_PAYLOAD)
        assert success is True
        assert validate_payload(data) is True

    def test_maps_into_sensor_data_end_to_end(self, core):
        _, data = decode_payload(GOLDEN_PAYLOAD)
        core._apply_sensor_data(data)

        sd = core.state.sensor_data
        assert sd.temp_dht == 22.2
        assert sd.humidity == 74.2
        assert sd.gas_mq2 == 0.74
        assert sd.tof_front == 340
        assert core.state.connected is True
