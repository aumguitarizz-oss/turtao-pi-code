import pytest

from turtao.config import AppConfig, Settings
from turtao.core import TurtaoCore
from turtao.hardware.mocks import MockCamera, MockSerialLink
from turtao.serial_link.protocol import decode_payload, validate_payload
from turtao.state import AppState

# Exact sample payload from turtao-pi-requirements.md §2, matching the
# confirmed ESP32-S3 firmware's sendSensorPayload() field-for-field.
GOLDEN_PAYLOAD = (
    '{"tof_fl": 340, "tof_fc": 18, "tof_fr": null, "tof_down": 27, '
    '"accel_x": -0.71, "accel_y": 0.60, "accel_z": -0.27, '
    '"gyro_x": -4.27, "gyro_y": 3.56, "gyro_z": -0.70, '
    '"gas_mq2": 0.74, "gas_mq135": 0.0, "sound_raw": 1871, '
    '"temp_dht": 22.2, "humidity": 74.2, "pir": true}'
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
        assert sd.air_quality_mq135 == 0.0
        assert sd.sound_raw == 1871
        assert sd.motion is True  # sourced from "pir"
        assert sd.imu.accel_x == -0.71
        assert sd.imu.accel_y == 0.60
        assert sd.imu.accel_z == -0.27
        assert sd.imu.gyro_x == -4.27
        assert sd.imu.gyro_y == 3.56
        assert sd.imu.gyro_z == -0.70
        # tof_fr is null in the sample (out of range / no target) -> the
        # existing "any nonzero" tof_cm guard leaves it as None, not 0.
        assert sd.tof_cm == [340, 18, None, 27]
        assert core.state.connected is True
