from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import Any

from turtao.api.ws_status import ws_broadcast_loop
from turtao.audio.bluetooth_manager import BluetoothManager, bluetooth_loop
from turtao.audio.tts import TTSManager
from turtao.config import BASE_DIR, AppConfig, Settings
from turtao.hardware.interfaces import CameraInterface, SerialLinkInterface
from turtao.patrol.patrol_loop import patrol_loop, set_safe_mode, set_speed
from turtao.serial_link.esp32_link import ESP32SerialLink
from turtao.serial_link.protocol import encode_command
from turtao.state import AppState, Mode, ThreatLabel
from turtao.vision.antispoof import AntiSpoofDetector
from turtao.vision.camera import Camera, camera_capture_loop
from turtao.vision.enrollment import EnrollmentManager
from turtao.vision.face_recognition_engine import FaceRecognitionEngine
from turtao.vision.person_tracker import PersonTracker
from turtao.vision.pose_tracker import PoseTracker

logger = logging.getLogger(__name__)

class TurtaoCore:
    """Central orchestrator.

    Constructed with injected interfaces for testability.
    Spawns all 8 daemon threads. Manages mode transitions.
    """

    def __init__(
        self,
        config: AppConfig,
        settings: Settings,
        state: AppState,
        serial_link: SerialLinkInterface | None = None,
        camera: CameraInterface | None = None,
        face_data_dir: Path = BASE_DIR / "face_data",
        model_path: str = "models/yolov8n.onnx",
        piper_dir: Path = BASE_DIR / "piper",
    ) -> None:
        self.config = config
        self.settings = settings
        self.state = state

        self.serial = serial_link or ESP32SerialLink(config, state)
        self.camera = camera or Camera(config.camera_index)

        self.face_engine = FaceRecognitionEngine(state, settings.face_tolerance)
        self.face_engine.load_embeddings(str(face_data_dir / "embeddings"))
        self.tracker = PersonTracker(model_path)
        self.antispoof = AntiSpoofDetector()
        self.enrollment = EnrollmentManager(face_data_dir)
        self.pose_tracker = PoseTracker(state)

        self.tts = TTSManager(piper_dir)
        self.bt_manager = BluetoothManager(config.jbl_mac)

        self._threads: list[threading.Thread] = []
        self._start_time: float = 0.0

    def start(self) -> None:
        """Open hardware connections, start all 8 daemon threads."""
        self._start_time = time.time()

        try:
            self.serial.open()
        except RuntimeError as e:
            logger.error("Failed to open serial link: %s", e)

        threads_config: list[tuple[str, Any, tuple[Any, ...]]] = [
            ("_camera_loop", camera_capture_loop, (self.state, self.camera)),
            ("_face_recognition_loop", self._face_recognition_wrapper, ()),
            ("_person_tracker_loop", self._tracker_wrapper, ()),
            ("_serial_loop", self._serial_wrapper, ()),
            ("_bluetooth_manager", bluetooth_loop, (self.state, self.bt_manager)),
            ("_wake_word_loop", self._wake_word_wrapper, ()),
            ("_patrol_loop", self._patrol_wrapper, ()),
            ("_ws_broadcast_loop", ws_broadcast_loop, (self.state,)),
        ]

        for name, target, args in threads_config:
            t = threading.Thread(
                target=target, args=args, name=name, daemon=True
            )
            t.start()
            self._threads.append(t)

        logger.info("TurtaoCore started with %d threads", len(self._threads))

    # ------------------------------------------------------------------
    # Thread wrappers
    # ------------------------------------------------------------------

    def _face_recognition_wrapper(self) -> None:
        """Wrap face_recognition_loop with IDLE-mode skipping."""
        while not self.state.stop_event.is_set():
            with self.state:
                mode = self.state.mode
                frame = self.state.frame_queue[-1] if self.state.frame_queue else None

            if mode == Mode.IDLE:
                with self.state:
                    self.state.threat_label = ThreatLabel.IDLE
                time.sleep(0.1)
                continue

            if frame is not None:
                self.face_engine.process_frame(frame)
            else:
                with self.state:
                    self.state.threat_label = ThreatLabel.IDLE
            time.sleep(0.03)

    def _tracker_wrapper(self) -> None:
        """Pull latest frame and push through YOLO + ByteTrack."""
        while not self.state.stop_event.is_set():
            with self.state:
                active = self.state.mode != Mode.IDLE
                frame = self.state.latest_frame
                show_mp = self.state.show_mediapipe

            self.tracker.set_active(active)
            if active and frame is not None:
                try:
                    persons = self.tracker.process_frame(frame)
                    with self.state:
                        self.state.latest_persons = persons
                except Exception:
                    logger.exception("Person tracker error")
                
                if show_mp:
                    self.pose_tracker.process_frame(frame)
                else:
                    with self.state:
                        self.state.pose_landmarks = []
            else:
                with self.state:
                    self.state.pose_landmarks = []
            time.sleep(0.05)

    def _serial_wrapper(self) -> None:
        """Delegate to ESP32SerialLink.run() or basic fallback."""
        if hasattr(self.serial, "run"):
            try:
                self.serial.run()
            except Exception:
                logger.exception("Serial run loop exited unexpectedly")
        else:
            while not self.state.stop_event.is_set():
                try:
                    line = self.serial.readline()
                    if line is not None:
                        from turtao.serial_link.protocol import decode_payload
                        success, data = decode_payload(line)
                        if success and "error" not in data:
                            self._apply_sensor_data(data)
                except Exception:
                    logger.exception("Serial wrapper error")
                time.sleep(0.05)

    def _apply_sensor_data(self, data: dict[str, Any]) -> None:
        with self.state:
            s = self.state.sensor_data
            s.temp_c = data.get("temp_dht", s.temp_c)
            s.humidity_pct = int(data.get("humidity_pct", s.humidity_pct))
            s.pressure_hpa = data.get("pressure_hpa", s.pressure_hpa)
            s.gas_mq2 = data.get("gas_mq2", s.gas_mq2)
            s.air_quality_mq135 = data.get("air_quality_mq135", s.air_quality_mq135)
            s.sound_level = data.get("sound_level", s.sound_level)
            s.motion = data.get("motion", s.motion)
            s.orientation.pitch = data.get("pitch", s.orientation.pitch)
            s.orientation.roll = data.get("roll", s.orientation.roll)
            s.orientation.yaw = data.get("yaw", s.orientation.yaw)
            s.voltage = data.get("voltage", s.voltage)
            s.current_ma = data.get("current_ma", s.current_ma)
            s.battery_pct = data.get("battery_pct", s.battery_pct)
            s.motor_controller_ok = data.get("motor_controller_ok", s.motor_controller_ok)
            s.firmware_version = data.get("firmware_version", s.firmware_version)
            tof = [
                data.get("tof_fl", 0),
                data.get("tof_fc", 0),
                data.get("tof_fr", 0),
                data.get("tof_down", 0),
            ]
            if any(v != 0 for v in tof):
                s.tof_cm = tof
            self.state.connected = True

    def _wake_word_wrapper(self) -> None:
        """openWakeWord + Whisper STT loop.  Placeholder for now."""
        logger.info("Wake-word loop started (placeholder)")
        while not self.state.stop_event.is_set():
            time.sleep(0.5)

    def _patrol_wrapper(self) -> None:
        """Sync speed/safe_mode from settings, then run patrol loop."""
        set_speed(self.settings.speed)
        set_safe_mode(self.settings.safe_mode)
        try:
            patrol_loop(self.state, self.serial)
        except Exception:
            logger.exception("Patrol loop exited unexpectedly")

    # ------------------------------------------------------------------
    # Mode management
    # ------------------------------------------------------------------

    def set_mode(self, mode: str) -> None:
        """Change operating mode with full transition handling."""
        try:
            new_mode = Mode(mode.upper())
        except ValueError:
            logger.error("Invalid mode: %s", mode)
            return

        with self.state:
            old_mode = self.state.mode
            if old_mode == new_mode:
                return
            self.state.mode = new_mode

        logger.info("Mode transition: %s -> %s", old_mode.value, new_mode.value)

        # CPU governor hints (log-only — actual change requires root)
        if new_mode == Mode.IDLE:
            logger.info("Governor hint: powersave (log only)")
        else:
            logger.info("Governor hint: ondemand (log only)")

        self.tracker.set_active(new_mode != Mode.IDLE)

        if new_mode == Mode.IDLE:
            self._stop_motors()

    def emergency_stop(self) -> None:
        """E-stop: direct serial write (bypass queue), force IDLE."""
        logger.warning("EMERGENCY STOP triggered")
        cmd = encode_command({"cmd": "move", "ml": 0.0, "mr": 0.0})
        try:
            self.serial.write(cmd)
        except Exception as e:
            logger.error("E-stop serial write failed: %s", e)
        with self.state:
            self.state.mode = Mode.IDLE
        self.tracker.set_active(False)

    def _stop_motors(self) -> None:
        cmd = encode_command({"cmd": "move", "ml": 0.0, "mr": 0.0})
        try:
            self.serial.write(cmd)
        except Exception as e:
            logger.error("Stop motors serial write failed: %s", e)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def stop(self) -> None:
        """Signal all threads to stop, clean up hardware."""
        logger.info("TurtaoCore stopping")
        self.state.stop_event.set()
        self._stop_motors()
        try:
            self.serial.close()
        except Exception as e:
            logger.error("Serial close error: %s", e)
        try:
            self.camera.release()
        except Exception as e:
            logger.error("Camera release error: %s", e)
        self.antispoof.release()
        self.bt_manager.disconnect()
        logger.info("TurtaoCore stopped")

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def uptime_s(self) -> int:
        return int(time.time() - self._start_time)

    @property
    def is_running(self) -> bool:
        return not self.state.stop_event.is_set() and self._start_time > 0
