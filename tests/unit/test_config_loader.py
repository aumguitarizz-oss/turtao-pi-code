import json
import logging
import pytest
from pathlib import Path
from turtao.config import load_settings, Settings


class TestLoadSettings:
    def test_valid_settings_file(self, tmp_path: Path):
        settings_path = tmp_path / "settings.json"
        data = {
            "hostname": "testbot",
            "ble_proximity_enabled": False,
            "phone_registration": "AA:BB:CC:DD:EE:FF",
            "tts_event_toggles": {"threat": False, "low_battery": True, "intruder": False},
            "intercom_volume": 0.5,
            "face_tolerance": 0.6,
            "anti_spoof_enabled": False,
            "speed": 0.5,
            "safe_mode": True,
            "auto_flashbang": True,
            "stealth_mode": True,
            "notifications": {"threat": False, "gas_danger": False, "low_battery": False, "tamper": False, "connection_lost": False},
        }
        settings_path.write_text(json.dumps(data))
        result = load_settings(settings_path)
        assert result.hostname == "testbot"
        assert result.ble_proximity_enabled is False
        assert result.phone_registration == "AA:BB:CC:DD:EE:FF"
        assert result.tts_event_toggles.threat is False
        assert result.intercom_volume == 0.5
        assert result.face_tolerance == 0.6
        assert result.anti_spoof_enabled is False
        assert result.speed == 0.5
        assert result.safe_mode is True
        assert result.auto_flashbang is True
        assert result.stealth_mode is True
        assert result.notifications.threat is False

    def test_missing_file_returns_defaults(self, tmp_path: Path, caplog):
        caplog.set_level(logging.WARNING)
        missing = tmp_path / "nonexistent.json"
        result = load_settings(missing)
        assert isinstance(result, Settings)
        assert result.hostname == ""
        assert result.face_tolerance == 0.52
        assert result.speed == 0.8
        assert "Settings file not found" in caplog.text

    def test_corrupted_json_returns_defaults(self, tmp_path: Path, caplog):
        caplog.set_level(logging.WARNING)
        settings_path = tmp_path / "settings.json"
        settings_path.write_text("{invalid json content!!!}")
        result = load_settings(settings_path)
        assert isinstance(result, Settings)
        assert result.speed == 0.8
        assert result.hostname == ""
        assert "Corrupt settings file" in caplog.text

    def test_partial_valid_json_returns_defaults(self, tmp_path: Path, caplog):
        caplog.set_level(logging.WARNING)
        settings_path = tmp_path / "settings.json"
        partial = {"hostname": "partial_bot", "speed": 0.3}
        settings_path.write_text(json.dumps(partial))
        result = load_settings(settings_path)
        assert isinstance(result, Settings)
        assert result.hostname == "partial_bot"
        assert result.speed == 0.3

    def test_default_path_used_when_none_given(self):
        result = load_settings(None)
        assert isinstance(result, Settings)

    def test_empty_object_returns_defaults(self, tmp_path: Path):
        settings_path = tmp_path / "settings.json"
        settings_path.write_text("{}")
        result = load_settings(settings_path)
        assert isinstance(result, Settings)
        assert result.hostname == ""

    def test_extra_unknown_fields_ignored(self, tmp_path: Path):
        settings_path = tmp_path / "settings.json"
        data = {"hostname": "bot", "unknown_field": "ignored"}
        settings_path.write_text(json.dumps(data))
        result = load_settings(settings_path)
        assert result.hostname == "bot"

    def test_wrong_type_field_falls_back_to_defaults(self, tmp_path: Path, caplog):
        caplog.set_level(logging.WARNING)
        settings_path = tmp_path / "settings.json"
        data = {"speed": "not_a_number"}
        settings_path.write_text(json.dumps(data))
        result = load_settings(settings_path)
        assert isinstance(result, Settings)
        assert result.speed == 0.8
