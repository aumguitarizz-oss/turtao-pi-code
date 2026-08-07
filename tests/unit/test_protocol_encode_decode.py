import pytest
import json
from turtao.serial_link.protocol import (
    encode_command,
    decode_payload,
    validate_payload,
    REQUIRED_SENSOR_FIELDS,
)


class TestEncodeCommand:
    def test_encodes_to_compact_json_with_newline(self):
        cmd = {"cmd": "move", "ml": 0.5, "mr": 0.5}
        result = encode_command(cmd)
        expected = json.dumps(cmd, separators=(",", ":")) + "\n"
        assert result == expected

    def test_empty_dict(self):
        result = encode_command({})
        assert result == "{}\n"

    def test_nested_dict_encodes(self):
        cmd = {"nested": {"key": "value"}}
        result = encode_command(cmd)
        parsed = json.loads(result.strip())
        assert parsed == cmd


class TestDecodePayload:
    def test_valid_json_round_trip(self):
        # Matches the stripped-build ESP32-S3 firmware's sendSensorPayload().
        valid = {
            "tof_front": 340,
            "gas_mq2": 0.74,
            "temp_dht": 22.2,
            "humidity": 74.2,
        }
        line = json.dumps(valid)
        success, data = decode_payload(line)
        assert success is True
        assert data == valid

    def test_malformed_json_returns_invalid_payload(self):
        success, data = decode_payload("{not valid json}")
        assert success is False
        assert data["error"] == "invalid_json"
        assert "raw" in data

    def test_truncated_json_returns_invalid_payload(self):
        success, data = decode_payload('{"temp_dht": 28.5, "hum')
        assert success is False
        assert data["error"] == "invalid_json"

    def test_garbage_input_returns_invalid(self):
        garbage_inputs = [
            "",
            "   ",
            "hello world",
            "12345",
            "null",
            "undefined",
            "<<EOF",
            "\x00\x01\x02",
        ]
        for inp in garbage_inputs:
            success, data = decode_payload(inp)
            assert success is False, f"Expected failure for input: {inp!r}"
            assert data["error"] == "invalid_json"

    def test_json_array_returns_invalid(self):
        success, data = decode_payload("[1, 2, 3]")
        assert success is False
        assert data["error"] == "invalid_json"

    def test_json_string_returns_invalid(self):
        success, data = decode_payload('"just a string"')
        assert success is False
        assert data["error"] == "invalid_json"

    def test_json_number_returns_invalid(self):
        success, data = decode_payload("42")
        assert success is False
        assert data["error"] == "invalid_json"

    def test_never_raises_exception(self):
        inputs = [
            None,
            123,
            "",
            "not json",
            "{bad",
            "\x00\xff\xfe",
        ]
        for inp in inputs:
            try:
                if not isinstance(inp, str):
                    inp = str(inp) if inp is not None else ""
                decode_payload(inp)
            except Exception as e:
                pytest.fail(f"decode_payload raised {type(e).__name__}: {e}")


class TestValidatePayload:
    def test_complete_payload_validates(self):
        payload = {field: None for field in REQUIRED_SENSOR_FIELDS}
        assert validate_payload(payload) is True

    def test_missing_field_fails(self):
        payload = {field: None for field in REQUIRED_SENSOR_FIELDS}
        del payload["temp_dht"]
        assert validate_payload(payload) is False

    def test_empty_dict_fails(self):
        assert validate_payload({}) is False

    def test_none_fails(self):
        assert validate_payload(None) is False

    def test_non_dict_fails(self):
        assert validate_payload("string") is False
        assert validate_payload(42) is False
        assert validate_payload([1, 2, 3]) is False


class TestValidatePayloadRejectsOldFieldNames:
    """The GY-91's BMP280 half is dead on this unit and was never read —
    temp_bmp/pressure_hpa never existed in the confirmed firmware. Also
    covers the earlier, more ambitious spec's fields (INA219 battery,
    fused pitch/roll/yaw, ble_devices, etc.) that never shipped."""

    def test_rejects_payload_missing_real_fields_even_with_extra_old_ones(self):
        old_style = {
            "temp_dht": 28.5,
            "temp_bmp": 28.1,
            "pressure_hpa": 1013.2,
            "gas_mq2": 120,
            "sound_level": 42.3,
            "motion": False,
            "pitch": 0.5,
            "roll": -0.2,
            "yaw": 1.1,
            "voltage": 12.4,
            "current_ma": 150,
            "battery_pct": 95.0,
            "motor_controller_ok": True,
            "firmware_version": "v2.1.0",
            "ble_devices": [],
            # Missing: humidity, tof_front — so this still fails despite
            # having temp_dht/gas_mq2 (real field names) present.
        }
        assert validate_payload(old_style) is False
