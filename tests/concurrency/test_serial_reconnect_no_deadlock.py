import threading
import time
from unittest.mock import MagicMock, patch

import pytest

from turtao.serial_link.esp32_link import ESP32SerialLink
from turtao.state import AppState
from turtao.config import AppConfig


@pytest.fixture
def state():
    s = AppState()
    yield s
    s.stop_event.set()


@pytest.fixture
def config():
    return AppConfig(esp32_port="COM_TEST")


class TestSerialReconnect:
    def test_reconnect_backoff_does_not_deadlock(self, state, config):
        with patch("turtao.serial_link.esp32_link.serial.Serial") as mock_ser_cls:
            mock_ser = MagicMock()
            mock_ser.is_open = True
            mock_ser.readline.side_effect = ConnectionError("device disconnect")
            mock_ser_cls.return_value = mock_ser

            link = ESP32SerialLink(config, state)

            t = threading.Thread(target=link.run, daemon=True)
            t.start()

            time.sleep(0.5)
            state.stop_event.set()
            t.join(timeout=5)

            assert not t.is_alive(), "Thread did not exit cleanly"

    def test_reconnect_after_initial_failure(self, state, config):
        call_count = 0

        def failing_open():
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                raise RuntimeError("no device")
            mock_ser = MagicMock()
            mock_ser.is_open = True
            mock_ser.readline.return_value = b'{"temp_dht":25.0}\n'
            return mock_ser

        with patch(
            "turtao.serial_link.esp32_link.serial.Serial",
            side_effect=failing_open,
        ) as mock_ser_cls:
            link = ESP32SerialLink(config, state)

            t = threading.Thread(target=link.run, daemon=True)
            t.start()

            time.sleep(0.1)
            state.stop_event.set()
            t.join(timeout=5)

            assert not t.is_alive(), "Thread did not exit cleanly"

    def test_read_error_closes_and_reconnects(self, state, config):
        with patch("turtao.serial_link.esp32_link.serial.Serial") as mock_ser_cls:
            mock_ser = MagicMock()
            mock_ser.is_open = True
            read_attempts = [0]

            def readline_side_effect():
                read_attempts[0] += 1
                if read_attempts[0] < 3:
                    return b'{"temp_dht":25.0}\n'
                raise ConnectionError("read failure")

            mock_ser.readline.side_effect = readline_side_effect
            mock_ser_cls.return_value = mock_ser

            link = ESP32SerialLink(config, state)

            t = threading.Thread(target=link.run, daemon=True)
            t.start()

            time.sleep(0.5)
            state.stop_event.set()
            t.join(timeout=5)

            assert not t.is_alive(), "Thread did not exit cleanly"

    def test_no_leaked_threads_after_stop(self, state, config):
        initial_count = threading.active_count()

        with patch("turtao.serial_link.esp32_link.serial.Serial") as mock_ser_cls:
            mock_ser = MagicMock()
            mock_ser.is_open = True
            mock_ser.readline.side_effect = ConnectionError("fail")
            mock_ser_cls.return_value = mock_ser

            link = ESP32SerialLink(config, state)

            t = threading.Thread(target=link.run, daemon=True)
            t.start()

            time.sleep(0.3)
            state.stop_event.set()
            t.join(timeout=5)

            time.sleep(0.1)
            active_now = threading.active_count()
            assert active_now <= initial_count + 1

    def test_poll_sensor_returns_none_on_decode_failure(self, state, config):
        with patch("turtao.serial_link.esp32_link.serial.Serial") as mock_ser_cls:
            mock_ser = MagicMock()
            mock_ser.is_open = True
            mock_ser.readline.return_value = b"not valid json\n"
            mock_ser_cls.return_value = mock_ser

            link = ESP32SerialLink(config, state)
            link._ser = mock_ser

            result = link.poll_sensor()
            assert result is None

    def test_poll_sensor_valid_payload(self, state, config):
        with patch("turtao.serial_link.esp32_link.serial.Serial") as mock_ser_cls:
            mock_ser = MagicMock()
            mock_ser.is_open = True
            mock_ser.readline.return_value = (
                b'{"tof_fl":50,"tof_fc":100,"tof_fr":150,"tof_down":200,'
                b'"accel_x":0,"accel_y":0,"accel_z":9.81,'
                b'"gyro_x":0,"gyro_y":0,"gyro_z":0,'
                b'"gas_mq2":100,"gas_mq135":80,"sound_raw":30,'
                b'"temp_dht":30.2,"humidity":60,"pir":false}\n'
            )
            mock_ser_cls.return_value = mock_ser

            link = ESP32SerialLink(config, state)
            link._ser = mock_ser

            result = link.poll_sensor()
            assert result is not None
            assert result["temp_dht"] == 30.2
            assert result["gas_mq135"] == 80

    def test_process_command_queue(self, state, config):
        with patch("turtao.serial_link.esp32_link.serial.Serial") as mock_ser_cls:
            mock_ser = MagicMock()
            mock_ser.is_open = True
            mock_ser_cls.return_value = mock_ser

            link = ESP32SerialLink(config, state)
            link._ser = mock_ser
            link._connected = True

            state.serial_command_queue.append({"cmd": "move", "ml": 0.5, "mr": 0.5})
            state.serial_command_queue.append({"cmd": "servo", "pan": 90, "tilt": 90})

            link.process_command_queue()

            assert len(state.serial_command_queue) == 0
            assert mock_ser.write.called
            writes = [c[0][0] for c in mock_ser.write.call_args_list]
            assert any(b"move" in w for w in writes)
