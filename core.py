"""
core.py — TurtaoCore
Raspberry Pi 5 brain for Turtao surveillance robot.

Entry point: python3 core.py  (run by systemd in production)
Dev entry:   gui.py imports TurtaoCore and drives it with a tkinter window.

Architecture corrections vs MVP:
  - INA219 is on ESP32-S3 I2C bus — NOT on Pi I2C. Battery data arrives
    via serial JSON from ESP32.
  - _battery_loop still exists but only monitors self.battery (populated
    from ESP32 JSON) and triggers low-battery TTS. No hardware access.
  - 8 threads: camera, face_recog, serial_esp32, serial_ipst,
    battery_monitor, bluetooth, patrol, wake_word.
  - 12V truck LEDs (white + red) controlled via IRLZ44N MOSFETs driven
    by ESP32 GPIO — NOT through IPST SE.
  - IPST SE handles E-stop button and status indicator LEDs only.
  - Motor/servo commands go through _build_move_cmd() / _build_servo_cmd()
    so Phase 1 → Phase 2 (POP-32i) swap requires no Pi-side rewrite.
  - BLE phone proximity: ESP32 scans, sends ble_phone_present in JSON.
  - ULP wake: ESP32 ULP monitors PIR/sound, sends ulp_wake in JSON.
  - All 4× VL53L0X ToF sensors go through TCA9548A mux on ESP32 I2C.
  - PWM hard cap ±0.8 on all motor commands.
"""

import os
import json
import time
import logging
import threading
import subprocess
import glob
from datetime import datetime, timezone
from queue import Queue, Empty, Full

import numpy as np
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("turtao.core")

# ── environment / config ──────────────────────────────────────────────────────
JBL_MAC              = os.getenv("JBL_MAC", "")
FLASK_PORT           = int(os.getenv("FLASK_PORT", 5000))
ESP32_PORT           = os.getenv("ESP32_PORT", "")
IPST_PORT            = os.getenv("IPST_PORT", "")
CAMERA_INDEX         = int(os.getenv("CAMERA_INDEX", 0))
OBSTACLE_MM          = int(os.getenv("OBSTACLE_THRESHOLD_MM", 200))
DROP_MM              = int(os.getenv("DROP_THRESHOLD_MM", 400))
BRIGHTNESS_THRESHOLD = int(os.getenv("BRIGHTNESS_THRESHOLD", 60))
# Phase 1 = TB6612FNG + PCA9685 via ESP32 directly
# Phase 2 = POP-32i via UART from ESP32 (translation in ESP32 firmware)
# Pi-side JSON format is identical for both phases.
ROBOT_PHASE          = int(os.getenv("ROBOT_PHASE", 1))

# ── paths ─────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
FACE_DIR    = os.path.join(BASE_DIR, "face_data")
EMB_DIR     = os.path.join(FACE_DIR, "embeddings")
IMG_DIR     = os.path.join(FACE_DIR, "images")
UNK_DIR     = os.path.join(FACE_DIR, "unknowns")
PROFILES    = os.path.join(FACE_DIR, "profiles.json")
SETTINGS_F  = os.path.join(BASE_DIR, "settings.json")
PIPER_BIN   = os.path.join(BASE_DIR, "piper", "piper")
PIPER_MODEL = os.path.join(BASE_DIR, "piper", "en_US-amy-medium.onnx")
SOUNDS_DIR  = os.path.join(BASE_DIR, "sounds")

# ── named constants ───────────────────────────────────────────────────────────
ESP32_BAUD            = 115200
IPST_BAUD             = 9600
MOTOR_CAP             = 0.8       # hard PWM cap — never exceed
BT_INIT_SLEEP_SEC     = 35        # wait for BT stack on boot
BT_CHECK_INTERVAL     = 60        # reconnect check (seconds)
SENSOR_POLL_INTERVAL  = 0.5       # how often to request sensor data
FACE_SCALE            = 0.5       # downscale for HOG detection speed
BLUR_THRESHOLD        = 80.0      # Laplacian variance minimum
MIN_FACE_RATIO        = 0.15      # face must cover >15% of frame
JPEG_QUALITY          = 70
ENROLL_SAMPLE_FRAMES  = 30
ENROLL_MIN_QUALITY    = 5
UNKNOWN_SAVE_DEBOUNCE = 2.0       # seconds
TAMPER_DEBOUNCE       = 10.0      # seconds between tamper TTS
GAS_DEBOUNCE          = 30.0      # seconds between gas TTS
BATTERY_WARN_INTERVAL = 60.0      # seconds between low-battery TTS
STROBE_DEFAULT_MS     = 3000      # default strobe duration
BLE_DISARM_COOLDOWN   = 5.0       # seconds between arm/disarm toggles

POSES = [
    "Face directly forward",
    "Turn slightly left (~30°)",
    "Turn slightly right (~30°)",
    "Tilt head slightly up",
    "Tilt head slightly down",
]

# 2S LiPo discharge curve (motor pack). Update if INA219 is on 3S logic pack.
VOLTAGE_TABLE = [
    (8.40, 100), (8.10, 90), (7.80, 75), (7.60, 60),
    (7.40, 50),  (7.20, 35), (7.00, 20), (6.80, 10), (6.60, 0),
]


class TurtaoCore:
    """Single headless controller. One instance per process."""

    def __init__(self):
        # ── locks ─────────────────────────────────────────────────────────
        self.state_lock        = threading.Lock()
        self.frame_lock        = threading.Lock()
        self.sensor_lock       = threading.Lock()
        self.battery_lock      = threading.Lock()
        self.serial_esp32_lock = threading.Lock()
        self.serial_ipst_lock  = threading.Lock()

        self.frame_queue         = Queue(maxsize=2)
        self.esp32_command_queue = Queue(maxsize=64)

        # ── recognition state ─────────────────────────────────────────────
        self.current_threat     = False
        self.current_confidence = 0.0
        self.current_mode       = "IDLE"
        self.current_name       = "Unknown"
        self.latest_frame       = None
        self.latest_annotated_frame = None

        # ── sensor data (populated by ESP32 serial thread) ────────────────
        self.sensor_data = {
            # environment
            "temp_dht": 0,   "humid": 0,
            "temp_bmp": 0,   "pressure": 0,
            "gas_mq2": 0,    "gas_mq135": 0,
            "sound": 0,      "pir": False,
            # IMU
            "accel_x": 0,    "accel_y": 0,   "accel_z": 0,
            "gyro_x": 0,     "gyro_y": 0,    "gyro_z": 0,
            # ToF (4× VL53L0X through TCA9548A mux on ESP32 I2C)
            "tof_fl": 9999,  "tof_fc": 9999,
            "tof_fr": 9999,  "tof_down": 0,
            # battery (INA219 on ESP32 I2C at 0x41, A0 HIGH)
            "battery_voltage": 0.0,
            "battery_current_ma": 0.0,
            "battery_pct": 0,
            "battery_charging": False,
            # BLE phone proximity (ESP32 BLE scanner)
            "ble_phone_present": False,
            # ULP wake trigger (one-shot from ESP32 ULP coprocessor)
            "ulp_wake": False,
            # WiFi RSSI fingerprint for room estimation
            "wifi_rssi": {},
        }

        # ── battery dict (derived from sensor_data, read by GUI + Flask) ──
        self.battery = {
            "voltage": 0.0, "current_ma": 0.0,
            "percentage": 0, "charging": False, "danger": False,
        }

        # ── hardware handles ───────────────────────────────────────────────
        self._camera_ok       = False
        self._serial_esp32    = None
        self._serial_ipst     = None
        self.silent_loop_proc = None

        # ── derived state ─────────────────────────────────────────────────
        self._phone_present      = False   # last known BLE state
        self._last_ble_change    = 0.0
        self._current_room       = "unknown"

        # ── debounce timestamps ───────────────────────────────────────────
        self.last_unknown_save    = 0.0
        self.last_battery_warn    = 0.0
        self.last_tamper_warn     = 0.0
        self.last_gas_mq2_warn    = 0.0
        self.last_gas_mq135_warn  = 0.0

        self.enrollment_session  = None
        self._start_time         = time.time()

        # ── event log (feeds GUI log tab) ─────────────────────────────────
        self._event_log  = []
        self._event_lock = threading.Lock()

        # ── bootstrap ─────────────────────────────────────────────────────
        for d in (EMB_DIR, IMG_DIR, UNK_DIR):
            os.makedirs(d, exist_ok=True)
        if not os.path.exists(PROFILES):
            with open(PROFILES, "w") as f:
                json.dump({}, f)

        self.settings    = self._load_settings()
        self.known_faces = []
        self._load_faces()
        self._init_cv()

        from flask import Flask
        from flask_socketio import SocketIO
        self.app      = Flask(__name__)
        self.socketio = SocketIO(
            self.app,
            async_mode="threading",
            cors_allowed_origins="*",
            logger=False,
            engineio_logger=False,
        )
        self._setup_routes()
        self._setup_socketio()
        log.info("TurtaoCore ready — Phase %d", ROBOT_PHASE)

    # ── hardware status ───────────────────────────────────────────────────────

    @property
    def hw_status(self):
        """Live hardware connection flags."""
        bt_ok  = (self.silent_loop_proc is not None
                  and self.silent_loop_proc.poll() is None)
        # battery OK = ESP32 is sending voltage data
        batt_ok = self.battery["voltage"] > 0.0
        return {
            "camera":    self._camera_ok,
            "esp32":     self._serial_esp32 is not None,
            "ipst":      self._serial_ipst  is not None,
            "battery":   batt_ok,
            "bluetooth": bt_ok,
        }

    # ── event log ─────────────────────────────────────────────────────────────

    def log_event(self, msg):
        ts = datetime.now().strftime("%H:%M:%S")
        with self._event_lock:
            self._event_log.append((ts, msg))
            if len(self._event_log) > 200:
                self._event_log.pop(0)
        log.info(msg)

    def get_events(self, since_index=0):
        with self._event_lock:
            return list(self._event_log[since_index:])

    # ── settings ──────────────────────────────────────────────────────────────

    def _load_settings(self):
        defaults = {
            "tts_enabled":  True,  "tts_threat":  True,
            "tts_gas":      True,  "tts_tamper":  True,
            "tts_patrol":   True,  "tts_wake":    True,
            "speed":        0.8,   "safe_mode":   False,
            "tolerance":    0.52,  "stealth_mode": False,
            "brightness_threshold": BRIGHTNESS_THRESHOLD,
            "jbl_mac":      JBL_MAC,
            "strobe_on_threat": True,
            "ble_auto_disarm":  True,
            # room fingerprints: {"living_room": {"ssid1": -45, "ssid2": -70}}
            "room_fingerprints": {},
        }
        if os.path.exists(SETTINGS_F):
            try:
                with open(SETTINGS_F) as f:
                    defaults.update(json.load(f))
            except Exception as e:
                log.warning("settings load error: %s", e)
        return defaults

    def _save_settings(self):
        try:
            with open(SETTINGS_F, "w") as f:
                json.dump(self.settings, f, indent=2)
        except Exception as e:
            log.warning("settings save error: %s", e)

    # ── cv init ───────────────────────────────────────────────────────────────

    def _init_cv(self):
        try:
            import cv2
            self._cv2 = cv2
        except ImportError:
            log.error("opencv not available")
            self._cv2 = None
        try:
            import face_recognition as fr
            self._fr = fr
        except ImportError:
            log.error("face_recognition not available")
            self._fr = None

    # ── face storage ──────────────────────────────────────────────────────────

    def _load_faces(self):
        faces = []
        for path in glob.glob(os.path.join(EMB_DIR, "*.npy")):
            stem  = os.path.splitext(os.path.basename(path))[0]
            parts = stem.rsplit("_", 1)
            if len(parts) != 2:
                continue
            try:
                faces.append((parts[0], np.load(path)))
            except Exception as e:
                log.warning("Failed loading %s: %s", path, e)
        with self.state_lock:
            self.known_faces = faces
        log.info("Loaded %d face embeddings.", len(faces))

    def _load_profiles(self):
        try:
            with open(PROFILES) as f:
                return json.load(f)
        except Exception:
            return {}

    def _save_profiles(self, data):
        with open(PROFILES, "w") as f:
            json.dump(data, f, indent=2)

    # ── maths ─────────────────────────────────────────────────────────────────

    @staticmethod
    def _cosine_similarity(a, b):
        denom = np.linalg.norm(a) * np.linalg.norm(b)
        return float(np.dot(a, b) / denom) if denom else 0.0

    @staticmethod
    def _calculate_aim(l, t, r, b, frame_w=640, frame_h=480):
        dx = ((l + r) / 2.0) - (frame_w / 2.0)
        dy = ((t + b) / 2.0) - (frame_h / 2.0)
        pan  = 90 + dx * 0.18 if abs(dx) >= 30 else 90
        tilt = 90 + dy * 0.22 if abs(dy) >= 30 else 90
        return max(10, min(170, int(pan))), max(10, min(170, int(tilt)))

    @staticmethod
    def _voltage_to_percent(v):
        table = VOLTAGE_TABLE
        if v >= table[0][0]:
            return 100
        if v <= table[-1][0]:
            return 0
        for i in range(len(table) - 1):
            v_hi, p_hi = table[i]
            v_lo, p_lo = table[i + 1]
            if v_lo <= v <= v_hi:
                ratio = (v - v_lo) / (v_hi - v_lo)
                return max(0, min(100, int(p_lo + ratio * (p_hi - p_lo))))
        return 0

    # ── motor / servo command builders (Phase abstraction) ────────────────────

    def _build_move_cmd(self, ml: float, mr: float) -> dict:
        """
        Build a motor move command.
        Phase 1: ESP32 drives TB6612FNG directly.
        Phase 2: ESP32 forwards to POP-32i via UART.
        Pi-side JSON format is identical — translation in ESP32 firmware.
        """
        ml = max(-MOTOR_CAP, min(MOTOR_CAP, ml))
        mr = max(-MOTOR_CAP, min(MOTOR_CAP, mr))
        return {"cmd": "move", "ml": round(ml, 3), "mr": round(mr, 3)}

    def _build_servo_cmd(self, pan: int, tilt: int) -> dict:
        """
        Build a pan-tilt servo command.
        Phase 1: PCA9685 via ESP32.
        Phase 2: POP-32i handles servos.
        Pi-side JSON format identical.
        """
        pan  = max(10, min(170, pan))
        tilt = max(10, min(170, tilt))
        return {"cmd": "pan_tilt", "pan": pan, "tilt": tilt}

    # ── TTS ───────────────────────────────────────────────────────────────────

    def _speak(self, text):
        if not self.settings.get("tts_enabled", True):
            return
        if self.settings.get("stealth_mode", False):
            return

        def _run():
            try:
                if not os.path.exists(PIPER_BIN):
                    log.warning("Piper binary missing — TTS skipped.")
                    return
                piper = subprocess.Popen(
                    [PIPER_BIN, "--model", PIPER_MODEL, "--output-raw"],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL,
                )
                aplay = subprocess.Popen(
                    ["aplay", "-r", "22050", "-f", "S16_LE", "-t", "raw", "-"],
                    stdin=piper.stdout,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                piper.stdout.close()
                piper.stdin.write(text.encode())
                piper.stdin.close()
                aplay.wait(timeout=15)
            except Exception as e:
                log.debug("TTS: %s", e)

        threading.Thread(target=_run, daemon=True).start()

    def speak(self, text):
        """Public — called from GUI test button."""
        self._speak(text)

    # ── strobe deterrent ──────────────────────────────────────────────────────

    def _trigger_strobe(self, duration_ms: int = STROBE_DEFAULT_MS):
        """Rapid-strobe the 12V truck LEDs via IRLZ44N MOSFETs on ESP32 GPIO."""
        if self.settings.get("stealth_mode", False):
            return
        self._send_esp32_command(
            {"cmd": "led_strobe", "duration_ms": duration_ms}
        )
        self.log_event(f"Strobe triggered ({duration_ms}ms)")

    # ── mode ──────────────────────────────────────────────────────────────────

    def set_mode(self, mode: str):
        with self.state_lock:
            self.current_mode = mode
        try:
            gov = "powersave" if mode == "IDLE" else "ondemand"
            subprocess.run(
                ["cpufreq-set", "-g", gov],
                timeout=3, capture_output=True,
            )
        except Exception:
            pass
        led_map = {
            "IDLE":   "LED:OFF",
            "GUARD":  "LED:GREEN",
            "PATROL": "LED:BLUE_BLINK",
        }
        self._send_ipst_command(led_map.get(mode, "LED:OFF"))
        self.log_event(f"Mode set to {mode}")
        if mode == "PATROL" and self.settings.get("tts_patrol", True):
            self._speak("Surveillance active. Beginning patrol.")

    # ── room estimation (WiFi RSSI fingerprinting) ────────────────────────────

    def _estimate_room(self, rssi_map: dict) -> str:
        """
        Compare current RSSI readings against stored fingerprints.
        Returns the closest room label or 'unknown'.
        Only runs if room_fingerprints is configured in settings.
        """
        fingerprints = self.settings.get("room_fingerprints", {})
        if not fingerprints or not rssi_map:
            return "unknown"
        best_room  = "unknown"
        best_score = float("inf")
        for room, fp in fingerprints.items():
            ssids = set(fp) | set(rssi_map)
            if not ssids:
                continue
            dist = sum(
                (rssi_map.get(s, -100) - fp.get(s, -100)) ** 2
                for s in ssids
            )
            if dist < best_score:
                best_score = dist
                best_room  = room
        return best_room

    # ── serial helpers ────────────────────────────────────────────────────────

    def _send_esp32_command(self, cmd_dict: dict):
        try:
            self.esp32_command_queue.put_nowait(cmd_dict)
        except Full:
            pass

    def send_esp32(self, cmd_dict: dict):
        """Public — called from GUI."""
        self._send_esp32_command(cmd_dict)

    def _send_ipst_command(self, cmd_str: str):
        if self._serial_ipst is None:
            return
        try:
            with self.serial_ipst_lock:
                self._serial_ipst.write(f"{cmd_str}\n".encode())
        except Exception as e:
            log.debug("IPST write: %s", e)

    # ── sensor alert dispatcher ───────────────────────────────────────────────

    def _handle_sensor_alerts(self, data: dict):
        """
        Called from the ESP32 serial thread after sensor_data is updated.
        Handles: tamper, gas, BLE proximity, ULP wake, battery danger.
        All alerts are debounced.
        """
        now = time.time()

        # ── tamper (IMU Z acceleration spike) ─────────────────────────────
        if (abs(data.get("accel_z", 0)) > 15
                and self.settings.get("tts_tamper", True)
                and now - self.last_tamper_warn > TAMPER_DEBOUNCE):
            self.last_tamper_warn = now
            self._speak("Warning. Do not touch this device.")
            self.log_event("[WARN] Tamper detected")

        # ── gas MQ-2 (flammable gas / smoke) ──────────────────────────────
        if (data.get("gas_mq2", 0) > 300
                and self.settings.get("tts_gas", True)
                and now - self.last_gas_mq2_warn > GAS_DEBOUNCE):
            self.last_gas_mq2_warn = now
            self._speak("Warning. Hazardous gas detected.")
            self.log_event("[WARN] Gas MQ2 high")

        # ── gas MQ-135 (air quality) ───────────────────────────────────────
        if (data.get("gas_mq135", 0) > 400
                and self.settings.get("tts_gas", True)
                and now - self.last_gas_mq135_warn > GAS_DEBOUNCE):
            self.last_gas_mq135_warn = now
            self._speak("Warning. Poor air quality detected.")
            self.log_event("[WARN] Gas MQ135 high")

        # ── battery danger (from INA219 on ESP32 I2C, data via JSON) ──────
        if (data.get("battery_voltage", 99) > 0
                and data.get("battery_voltage", 99) < 6.6):
            if now - self.last_battery_warn > BATTERY_WARN_INTERVAL:
                self.last_battery_warn = now
                self._speak("Battery critically low.")
                self.log_event("[WARN] Battery critical")

        # ── BLE phone proximity auto arm/disarm ────────────────────────────
        if self.settings.get("ble_auto_disarm", True):
            phone_now = bool(data.get("ble_phone_present", False))
            if (phone_now != self._phone_present
                    and now - self._last_ble_change > BLE_DISARM_COOLDOWN):
                self._phone_present    = phone_now
                self._last_ble_change  = now
                if phone_now:
                    # owner is home — disarm
                    self.set_mode("IDLE")
                    self.log_event("BLE: owner phone detected — disarmed")
                    self._speak("Welcome home.")
                else:
                    # owner left — arm to GUARD
                    self.set_mode("GUARD")
                    self.log_event("BLE: owner phone left — armed to GUARD")
                    self._speak("Goodbye. Surveillance armed.")

        # ── ULP coprocessor wake (PIR / sound triggered sleep wakeup) ─────
        if data.get("ulp_wake", False):
            with self.state_lock:
                mode = self.current_mode
            if mode == "IDLE":
                self.set_mode("GUARD")
                self.log_event("ULP wake: motion/sound detected — switching to GUARD")

        # ── WiFi RSSI room estimation ──────────────────────────────────────
        rssi = data.get("wifi_rssi", {})
        if rssi and isinstance(rssi, dict):
            self._current_room = self._estimate_room(rssi)

    # ── quality gate ──────────────────────────────────────────────────────────

    def _quality_gate_frame(self, frame, face_location):
        cv2 = self._cv2
        top, right, bottom, left = face_location
        h, w = frame.shape[:2]
        if (right - left) * (bottom - top) / max(1, h * w) < MIN_FACE_RATIO:
            return False
        face_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)[top:bottom, left:right]
        if face_gray.size == 0:
            return False
        if cv2.Laplacian(face_gray, cv2.CV_64F).var() < BLUR_THRESHOLD:
            return False
        mean_val = float(face_gray.mean())
        return 40 <= mean_val <= 220

    # ─────────────────────────────── THREADS ─────────────────────────────────

    def _camera_loop(self):
        cv2 = self._cv2
        if cv2 is None:
            log.error("_camera_loop: opencv unavailable — exiting.")
            return
        cap = None
        frame_count = 0
        while True:
            if cap is None or not cap.isOpened():
                cap = cv2.VideoCapture(CAMERA_INDEX)
                cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                if not cap.isOpened():
                    self._camera_ok = False
                    log.warning("Camera %d not opened, retrying…", CAMERA_INDEX)
                    time.sleep(2)
                    continue
                self._camera_ok = True
                self.log_event(f"Camera {CAMERA_INDEX} connected.")

            ret, frame = cap.read()
            if not ret:
                self._camera_ok = False
                cap.release()
                cap = None
                time.sleep(2)
                continue

            frame_count += 1
            if frame_count % 30 == 0:
                gray_mean = float(
                    cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).mean())
                with self.state_lock:
                    mode = self.current_mode
                thresh = self.settings.get("brightness_threshold",
                                           BRIGHTNESS_THRESHOLD)
                if gray_mean < thresh and mode != "IDLE":
                    # Turn on white truck LED via IRLZ44N MOSFET on ESP32
                    self._send_esp32_command(
                        {"cmd": "led_white", "mode": "on", "pwm": 128})

            with self.frame_lock:
                self.latest_frame = frame.copy()
            try:
                self.frame_queue.put_nowait(frame.copy())
            except Full:
                pass

    def _face_recognition_loop(self):
        cv2 = self._cv2
        fr  = self._fr
        if cv2 is None:
            log.error("_face_recognition_loop: opencv unavailable — exiting.")
            return
        if fr is None:
            log.error("_face_recognition_loop: face_recognition unavailable — exiting.")
            return

        while True:
            try:
                frame = self.frame_queue.get(timeout=1.0)
            except Empty:
                continue

            with self.state_lock:
                mode = self.current_mode
            if mode == "IDLE":
                with self.frame_lock:
                    self.latest_annotated_frame = frame.copy()
                with self.state_lock:
                    self.current_threat = False
                continue

            small     = cv2.resize(frame, (0, 0), fx=FACE_SCALE, fy=FACE_SCALE)
            rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
            locations = fr.face_locations(rgb_small, model="hog")

            if not locations:
                with self.frame_lock:
                    self.latest_annotated_frame = frame.copy()
                with self.state_lock:
                    self.current_threat = False
                continue

            encodings      = fr.face_encodings(rgb_small, locations)
            scale_back     = 1.0 / FACE_SCALE
            locations_full = [
                (int(t * scale_back), int(r * scale_back),
                 int(b * scale_back), int(l * scale_back))
                for (t, r, b, l) in locations
            ]
            annotated = frame.copy()

            for (t, r, b, l), enc in zip(locations_full, encodings):
                best_sim, best_name = -1.0, "Unknown"
                with self.state_lock:
                    known = list(self.known_faces)
                for name, known_enc in known:
                    sim = self._cosine_similarity(enc, known_enc)
                    if sim > best_sim:
                        best_sim, best_name = sim, name

                tol        = self.settings.get("tolerance", 0.52)
                is_threat  = best_sim < tol
                label      = "Unknown" if is_threat else best_name
                confidence = max(0.0, best_sim)

                was_threat = False
                with self.state_lock:
                    was_threat              = self.current_threat
                    self.current_threat     = is_threat
                    self.current_confidence = confidence
                    self.current_name       = label

                if is_threat:
                    pan, tilt = self._calculate_aim(
                        l, t, r, b,
                        frame_w=frame.shape[1],
                        frame_h=frame.shape[0],
                    )
                    self._send_esp32_command(self._build_servo_cmd(pan, tilt))
                    if mode == "PATROL":
                        self._send_esp32_command(self._build_move_cmd(0.0, 0.0))
                    now = time.time()
                    if now - self.last_unknown_save > UNKNOWN_SAVE_DEBOUNCE:
                        self.last_unknown_save = now
                        self._save_unknown_face(frame, t, r, b, l)
                    if not was_threat:
                        if self.settings.get("tts_threat", True):
                            self._speak(
                                "Intruder detected. Owner has been notified.")
                        if self.settings.get("strobe_on_threat", True):
                            self._trigger_strobe()
                        self.log_event(
                            f"[THREAT] detected — conf {confidence:.2f}")

                color = (0, 0, 255) if is_threat else (0, 255, 0)
                cv2.rectangle(annotated, (l, t), (r, b), color, 2)
                cv2.putText(
                    annotated, f"{label} ({confidence:.2f})",
                    (l, max(0, t - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1,
                )

            with self.frame_lock:
                self.latest_annotated_frame = annotated

    def _save_unknown_face(self, frame, t, r, b, l):
        try:
            crop = frame[t:b, l:r]
            if crop.size == 0:
                return
            path = os.path.join(UNK_DIR, f"{int(time.time() * 1000)}.jpg")
            self._cv2.imwrite(path, crop)
        except Exception as e:
            log.debug("Unknown face save: %s", e)

    def _serial_esp32_loop(self):
        import serial
        import serial.tools.list_ports

        def _find_port():
            if ESP32_PORT:
                return ESP32_PORT
            for p in serial.tools.list_ports.comports():
                if "USB" in p.device or "ACM" in p.device:
                    return p.device
            return None

        while True:
            port = _find_port()
            if not port:
                time.sleep(2)
                continue
            try:
                ser = serial.Serial(port, ESP32_BAUD, timeout=1)
                with self.serial_esp32_lock:
                    self._serial_esp32 = ser
                self.log_event(f"ESP32-S3 connected on {port}")

                stop = threading.Event()

                def _poll(stop_evt):
                    while not stop_evt.is_set():
                        time.sleep(SENSOR_POLL_INTERVAL)
                        try:
                            self.esp32_command_queue.put_nowait(
                                {"cmd": "sensors"})
                        except Full:
                            pass

                def _write(stop_evt):
                    while not stop_evt.is_set():
                        try:
                            cmd = self.esp32_command_queue.get(timeout=1.0)
                            with self.serial_esp32_lock:
                                ser.write(
                                    (json.dumps(cmd) + "\n").encode())
                        except Empty:
                            pass
                        except Exception:
                            stop_evt.set()
                            break

                threading.Thread(target=_poll,  args=(stop,),
                                 daemon=True).start()
                threading.Thread(target=_write, args=(stop,),
                                 daemon=True).start()

                while not stop.is_set():
                    try:
                        raw  = ser.readline().decode(errors="replace").strip()
                        if not raw:
                            continue
                        data = json.loads(raw)

                        with self.sensor_lock:
                            self.sensor_data.update(data)

                        # Update battery dict from ESP32 battery fields
                        if "battery_voltage" in data:
                            v   = data["battery_voltage"]
                            i   = data.get("battery_current_ma", 0.0)
                            pct = data.get(
                                "battery_pct",
                                self._voltage_to_percent(v))
                            with self.battery_lock:
                                self.battery = {
                                    "voltage":    round(v, 3),
                                    "current_ma": round(i, 1),
                                    "percentage": pct,
                                    "charging":   i > 50,
                                    "danger":     v < 6.6,
                                }

                        self._handle_sensor_alerts(data)

                    except json.JSONDecodeError:
                        pass
                    except Exception:
                        stop.set()
                        break

                stop.set()
                ser.close()
            except Exception as e:
                log.warning("ESP32 serial error: %s", e)
            with self.serial_esp32_lock:
                self._serial_esp32 = None
            self.log_event("ESP32-S3 disconnected.")
            time.sleep(2)

    def _serial_ipst_loop(self):
        """
        IPST SE handles E-stop button and status indicator LEDs only.
        12V truck LEDs are controlled by IRLZ44N MOSFETs via ESP32 GPIO.
        """
        import serial
        import serial.tools.list_ports

        def _find_port():
            if IPST_PORT:
                return IPST_PORT
            for p in serial.tools.list_ports.comports():
                if (p.device != (ESP32_PORT or "")
                        and ("USB" in p.device or "ACM" in p.device)):
                    return p.device
            return None

        last_led_send = 0.0
        while True:
            port = _find_port()
            if not port:
                time.sleep(2)
                continue
            try:
                ser = serial.Serial(port, IPST_BAUD, timeout=1)
                with self.serial_ipst_lock:
                    self._serial_ipst = ser
                self.log_event(f"IPST SE connected on {port}")

                while True:
                    try:
                        raw = ser.readline().decode(errors="replace").strip()
                        if raw == "ESTOP":
                            self.log_event("[ESTOP] received — halting motors")
                            self._send_esp32_command(
                                self._build_move_cmd(0.0, 0.0))
                            self.set_mode("IDLE")
                    except Exception:
                        pass

                    now = time.time()
                    if now - last_led_send >= 1.0:
                        last_led_send = now
                        with self.state_lock:
                            mode, threat = (self.current_mode,
                                            self.current_threat)
                        if threat:
                            led = "LED:RED"
                        elif mode == "IDLE":
                            led = "LED:OFF"
                        elif mode == "GUARD":
                            led = "LED:GREEN"
                        else:
                            led = "LED:BLUE_BLINK"
                        try:
                            with self.serial_ipst_lock:
                                ser.write(f"{led}\n".encode())
                        except Exception:
                            break
                ser.close()
            except Exception as e:
                log.warning("IPST serial error: %s", e)
            with self.serial_ipst_lock:
                self._serial_ipst = None
            self.log_event("IPST SE disconnected.")
            time.sleep(2)

    def _battery_monitor_loop(self):
        """
        Watches self.battery (populated by ESP32 serial thread) and triggers
        low-battery TTS alerts. No hardware access — INA219 is on ESP32 I2C.
        """
        while True:
            time.sleep(10)
            with self.battery_lock:
                b = dict(self.battery)
            v = b.get("voltage", 0.0)
            if 0 < v < 6.6:
                now = time.time()
                if now - self.last_battery_warn > BATTERY_WARN_INTERVAL:
                    self.last_battery_warn = now
                    self._speak("Battery critically low.")
                    self.log_event("[WARN] Battery critical")

    def _bluetooth_manager(self):
        mac = self.settings.get("jbl_mac", "") or JBL_MAC
        if not mac:
            log.warning("JBL_MAC not set — Bluetooth audio skipped.")
            return
        log.info("BT manager sleeping %ds for stack init…", BT_INIT_SLEEP_SEC)
        time.sleep(BT_INIT_SLEEP_SEC)

        def _connect():
            try:
                subprocess.run(
                    ["bluetoothctl", "connect", mac],
                    timeout=10, capture_output=True,
                )
            except Exception as e:
                log.warning("BT connect failed: %s", e)

        def _start_silent_loop():
            try:
                if (self.silent_loop_proc
                        and self.silent_loop_proc.poll() is None):
                    self.silent_loop_proc.terminate()
            except Exception:
                pass
            silence = os.path.join(SOUNDS_DIR, "silence.wav")
            if not os.path.exists(silence):
                log.warning("silence.wav missing — JBL keepalive skipped.")
                return
            try:
                self.silent_loop_proc = subprocess.Popen(
                    ["aplay", "--loop", silence,
                     "-D", f"bluealsa:DEV={mac},PROFILE=a2dp"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            except Exception as e:
                log.warning("Silent loop failed: %s", e)

        _connect()
        _start_silent_loop()
        while True:
            time.sleep(BT_CHECK_INTERVAL)
            try:
                result = subprocess.run(
                    ["bluetoothctl", "info", mac],
                    capture_output=True, text=True, timeout=5,
                )
                if "Connected: yes" not in result.stdout:
                    self.log_event("BT disconnected — reconnecting…")
                    _connect()
                    _start_silent_loop()
                elif (self.silent_loop_proc
                      and self.silent_loop_proc.poll() is not None):
                    _start_silent_loop()
            except Exception as e:
                log.debug("BT check: %s", e)

    def _patrol_loop(self):
        while True:
            time.sleep(0.2)
            with self.state_lock:
                mode, threat = self.current_mode, self.current_threat
            if mode != "PATROL" or threat:
                continue
            with self.sensor_lock:
                tof_fl   = self.sensor_data.get("tof_fl",   9999)
                tof_fc   = self.sensor_data.get("tof_fc",   9999)
                tof_fr   = self.sensor_data.get("tof_fr",   9999)
                tof_down = self.sensor_data.get("tof_down",    0)

            if tof_down > DROP_MM:
                cmd = self._build_move_cmd(-0.5, -0.5)
            elif tof_fc < OBSTACLE_MM:
                cmd = (self._build_move_cmd(-0.5, 0.5)
                       if tof_fl > tof_fr else
                       self._build_move_cmd(0.5, -0.5))
            elif tof_fl < OBSTACLE_MM:
                cmd = self._build_move_cmd(0.4, -0.4)
            elif tof_fr < OBSTACLE_MM:
                cmd = self._build_move_cmd(-0.4, 0.4)
            else:
                speed = min(MOTOR_CAP, self.settings.get("speed", 0.8))
                cmd   = self._build_move_cmd(speed, speed)

            self._send_esp32_command(cmd)

    def _wake_word_loop(self):
        """
        Listens for 'Hey Turtao' using openWakeWord.
        Only runs Whisper AFTER wake word is detected — not continuously.
        Commands: patrol, guard, stand down, stop.
        Gracefully degrades if packages not installed.
        """
        try:
            from openwakeword.model import Model as OWWModel
            import pyaudio
            import whisper
        except ImportError as e:
            log.warning("Wake word / Whisper not available: %s", e)
            return

        try:
            oww_model     = OWWModel(
                wakeword_models=[], inference_framework="onnx")
            whisper_model = whisper.load_model("tiny")
            log.info("Wake word + Whisper loaded.")
        except Exception as e:
            log.warning("Wake word model load failed: %s", e)
            return

        RATE  = 16000
        CHUNK = int(RATE * 0.08)   # 80 ms

        pa = pyaudio.PyAudio()
        try:
            stream = pa.open(
                rate=RATE, channels=1,
                format=pyaudio.paInt16,
                input=True,
                frames_per_buffer=CHUNK,
            )
        except Exception as e:
            log.warning("Microphone open failed: %s", e)
            pa.terminate()
            return

        log.info("Wake word loop running.")
        while True:
            try:
                chunk = np.frombuffer(
                    stream.read(CHUNK, exception_on_overflow=False),
                    dtype=np.int16,
                )
                preds = oww_model.predict(chunk)
                if not any(v > 0.7 for v in preds.values()):
                    continue

                # Wake word confirmed — capture 4s for Whisper
                if self.settings.get("tts_wake", True):
                    self._speak("Yes?")
                self.log_event("Wake word detected — listening for command")

                frames = [
                    stream.read(CHUNK, exception_on_overflow=False)
                    for _ in range(int(RATE * 4 / CHUNK))
                ]
                audio_np = (
                    np.frombuffer(b"".join(frames), dtype=np.int16)
                    .astype(np.float32) / 32768.0
                )
                result     = whisper_model.transcribe(audio_np, language="en")
                transcript = result.get("text", "").lower().strip()
                self.log_event(f"Voice: \"{transcript}\"")
                self._parse_voice_command(transcript)

            except Exception as e:
                log.debug("Wake word loop error: %s", e)

    def _parse_voice_command(self, transcript: str):
        if "patrol" in transcript:
            self.set_mode("PATROL")
            self._speak("Starting patrol.")
        elif "guard" in transcript:
            self.set_mode("GUARD")
            self._speak("Guard mode active.")
        elif "stand down" in transcript or "disarm" in transcript:
            self.set_mode("IDLE")
            self._speak("Standing down.")
        elif "stop" in transcript:
            self._send_esp32_command(self._build_move_cmd(0.0, 0.0))
            self._speak("Stopping.")

    # ── MJPEG generator ───────────────────────────────────────────────────────

    def _generate_mjpeg(self):
        cv2 = self._cv2
        while True:
            try:
                with self.frame_lock:
                    frame = (self.latest_annotated_frame
                             if self.latest_annotated_frame is not None
                             else self.latest_frame)
                if frame is None:
                    time.sleep(0.033)
                    continue
                _, buf = cv2.imencode(
                    ".jpg", frame,
                    [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY],
                )
                yield (
                    b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                    + buf.tobytes()
                    + b"\r\n"
                )
                time.sleep(0.033)
            except GeneratorExit:
                break
            except Exception:
                time.sleep(0.033)

    # ── shared enroll / promote / delete (Flask + GUI both call these) ────────

    def _do_enroll_capture(self):
        """Thread-safe. Returns result dict. Called by Flask route and GUI."""
        session = self.enrollment_session
        if session is None:
            return {"success": False, "reason": "no active session"}
        cv2 = self._cv2
        fr  = self._fr
        if cv2 is None or fr is None:
            return {"success": False,
                    "reason": "opencv or face_recognition not available"}

        name = session["name"]
        pose = session["pose"]

        quality_frames          = []
        best_frame, best_score  = None, -1.0

        for _ in range(ENROLL_SAMPLE_FRAMES):
            with self.frame_lock:
                frame = self.latest_frame
            if frame is None:
                time.sleep(0.05)
                continue
            rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            small = cv2.resize(rgb, (0, 0), fx=FACE_SCALE, fy=FACE_SCALE)
            locs  = fr.face_locations(small, model="hog")
            if not locs:
                time.sleep(0.05)
                continue
            scale_back = 1.0 / FACE_SCALE
            locs_full  = [
                (int(t * scale_back), int(r * scale_back),
                 int(b * scale_back), int(l * scale_back))
                for (t, r, b, l) in locs
            ]
            encs = fr.face_encodings(small, locs)
            if not encs:
                time.sleep(0.05)
                continue
            t, r, b, l = locs_full[0]
            if self._quality_gate_frame(frame, (t, r, b, l)):
                quality_frames.append(encs[0])
                score = cv2.Laplacian(
                    cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)[t:b, l:r],
                    cv2.CV_64F,
                ).var()
                if score > best_score:
                    best_score = score
                    best_frame = frame[t:b, l:r].copy()
            time.sleep(0.05)

        if len(quality_frames) < ENROLL_MIN_QUALITY:
            return {"success": False, "reason": "blur/size/light", "retry": True}

        if self.enrollment_session is None:
            return {"success": False, "reason": "session cancelled"}

        mean_enc  = np.mean(quality_frames, axis=0)
        mean_enc /= np.linalg.norm(mean_enc)
        np.save(os.path.join(EMB_DIR, f"{name}_{pose:03d}.npy"), mean_enc)

        if best_frame is not None:
            cv2.imwrite(
                os.path.join(
                    IMG_DIR, f"{name}_{pose:03d}_{int(time.time())}.jpg"),
                best_frame,
            )

        self.enrollment_session["pose"] += 1
        next_pose = self.enrollment_session["pose"]

        if next_pose >= 5:
            profiles       = self._load_profiles()
            profiles[name] = {
                "enrolled": datetime.now(timezone.utc).isoformat()}
            self._save_profiles(profiles)
            self._load_faces()
            self.log_event(f"[OK] Enrolled: {name}")
            self.enrollment_session = None
            return {"success": True, "complete": True}

        self.log_event(f"Pose {next_pose}/5 captured for {name}")
        return {
            "success":   True,
            "complete":  False,
            "pose":      next_pose,
            "pose_name": POSES[next_pose],
        }

    def _do_promote(self, uid: str, name: str):
        if not uid or not name:
            return False, "missing fields"
        path = os.path.join(UNK_DIR, f"{uid}.jpg")
        if not os.path.exists(path):
            return False, "unknown not found"
        cv2 = self._cv2
        fr  = self._fr
        img = cv2.imread(path)
        if img is None:
            return False, "failed to read image file"
        rgb  = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        locs = fr.face_locations(rgb, model="hog")
        encs = fr.face_encodings(rgb, locs)
        if not encs:
            return False, "no face found in image"
        enc = encs[0] / np.linalg.norm(encs[0])
        idx = len(sorted(glob.glob(os.path.join(EMB_DIR, f"{name}_*.npy"))))
        np.save(os.path.join(EMB_DIR, f"{name}_{idx:03d}.npy"), enc)
        import shutil
        shutil.copy(
            path, os.path.join(IMG_DIR, f"{name}_{idx:03d}_{uid}.jpg"))
        os.remove(path)
        profiles       = self._load_profiles()
        profiles[name] = {
            "enrolled": datetime.now(timezone.utc).isoformat()}
        self._save_profiles(profiles)
        self._load_faces()
        self.log_event(f"[OK] Promoted unknown {uid} to {name}")
        return True, ""

    def _do_delete_face(self, name: str):
        for p in (
            glob.glob(os.path.join(EMB_DIR, f"{name}_*.npy"))
            + glob.glob(os.path.join(IMG_DIR, f"{name}_*.jpg"))
        ):
            os.remove(p)
        profiles = self._load_profiles()
        profiles.pop(name, None)
        self._save_profiles(profiles)
        self._load_faces()
        self.log_event(f"[DEL] Deleted face: {name}")

    # ── Flask routes ──────────────────────────────────────────────────────────

    def _setup_routes(self):
        from flask import request, jsonify, send_file, Response, abort
        app = self.app

        @app.route("/api/health")
        def _health():
            with self.state_lock:
                mode = self.current_mode
            return jsonify({
                "status": "ok",
                "mode":   mode,
                "phase":  ROBOT_PHASE,
                "uptime_seconds": int(time.time() - self._start_time),
                "hw":     self.hw_status,
                "room":   self._current_room,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })

        @app.route("/api/alert")
        def _alert():
            with self.state_lock:
                threat, conf, mode, name = (
                    self.current_threat, self.current_confidence,
                    self.current_mode,   self.current_name,
                )
            return jsonify({
                "threat":     threat,
                "confidence": round(conf, 3),
                "state":      "THREAT" if threat else mode,
                "name":       name,
                "timestamp":  datetime.now(timezone.utc).isoformat(),
            })

        @app.route("/api/stream")
        def _stream():
            return Response(
                self._generate_mjpeg(),
                mimetype="multipart/x-mixed-replace; boundary=frame",
            )

        @app.route("/api/battery")
        def _battery():
            with self.battery_lock:
                return jsonify(dict(self.battery))

        @app.route("/api/environment")
        def _env():
            with self.sensor_lock:
                return jsonify(dict(self.sensor_data))

        @app.route("/api/hw")
        def _hw():
            return jsonify(self.hw_status)

        @app.route("/api/room")
        def _room():
            return jsonify({
                "room": self._current_room,
                "phone_present": self._phone_present,
            })

        @app.route("/api/faces")
        def _faces():
            profiles = self._load_profiles()
            return jsonify([
                {"name": n, "thumb_url": f"/api/faces/{n}/thumb"}
                for n in profiles
            ])

        @app.route("/api/faces/unknowns")
        def _unknowns():
            items = []
            for path in sorted(glob.glob(os.path.join(UNK_DIR, "*.jpg"))):
                uid = os.path.splitext(os.path.basename(path))[0]
                ts  = int(uid) / 1000 if uid.isdigit() else 0
                items.append({
                    "id": uid,
                    "timestamp": (
                        datetime.fromtimestamp(ts, timezone.utc).isoformat()
                        if ts else uid),
                    "image_url": f"/api/faces/unknowns/{uid}",
                })
            return jsonify(items)

        @app.route("/api/faces/<name>/thumb")
        def _thumb(name):
            files = sorted(glob.glob(os.path.join(IMG_DIR, f"{name}_*.jpg")))
            if not files:
                abort(404)
            return send_file(files[0], mimetype="image/jpeg")

        @app.route("/api/faces/unknowns/<uid>")
        def _unk_img(uid):
            path = os.path.join(UNK_DIR, f"{uid}.jpg")
            if not os.path.exists(path):
                abort(404)
            return send_file(path, mimetype="image/jpeg")

        @app.route("/api/faces/enroll/status")
        def _enroll_status():
            if self.enrollment_session is None:
                abort(404)
            return jsonify(self.enrollment_session)

        @app.route("/api/faces/enroll/start", methods=["POST"])
        def _enroll_start():
            data = request.get_json(force=True, silent=True) or {}
            name = data.get("name", "").strip()
            if not name:
                return jsonify({"error": "name required"}), 400
            self.enrollment_session = {"name": name, "pose": 0, "total": 5}
            return jsonify(
                {**self.enrollment_session, "pose_name": POSES[0]})

        @app.route("/api/faces/enroll/capture", methods=["POST"])
        def _enroll_capture():
            result = self._do_enroll_capture()
            code   = 200 if (result.get("success") or result.get("retry")) else 400
            return jsonify(result), code

        @app.route("/api/faces/promote", methods=["POST"])
        def _promote():
            data    = request.get_json(force=True, silent=True) or {}
            ok, msg = self._do_promote(
                data.get("unknown_id", ""),
                data.get("name", "").strip(),
            )
            return (jsonify({"ok": True}) if ok
                    else (jsonify({"error": msg}), 400))

        @app.route("/api/faces/<name>", methods=["DELETE"])
        def _delete_face(name):
            self._do_delete_face(name)
            return jsonify({"ok": True})

        @app.route("/api/settings", methods=["GET"])
        def _settings_get():
            return jsonify(self.settings)

        @app.route("/api/settings", methods=["POST"])
        def _settings_post():
            data = request.get_json(force=True, silent=True) or {}
            self.settings.update(data)
            self._save_settings()
            return jsonify({"ok": True})

        @app.route("/api/move", methods=["POST"])
        def _move():
            data = request.get_json(force=True, silent=True) or {}
            cmd  = self._build_move_cmd(
                float(data.get("ml", 0)),
                float(data.get("mr", 0)),
            )
            # bypass queue for responsiveness
            if self._serial_esp32:
                try:
                    with self.serial_esp32_lock:
                        self._serial_esp32.write(
                            (json.dumps(cmd) + "\n").encode())
                except Exception:
                    pass
            return jsonify({"ok": True})

        @app.route("/api/mode", methods=["POST"])
        def _mode():
            data = request.get_json(force=True, silent=True) or {}
            mode = data.get("mode", "IDLE").upper()
            if mode not in ("IDLE", "GUARD", "PATROL"):
                return jsonify({"error": "invalid mode"}), 400
            self.set_mode(mode)
            return jsonify({"ok": True, "mode": mode})

        @app.route("/api/led", methods=["POST"])
        def _led():
            data = request.get_json(force=True, silent=True) or {}
            led  = data.get("led", "white")   # "white" or "red"
            mode = data.get("mode", "off")     # "on", "off", "strobe"
            if mode == "strobe":
                self._trigger_strobe(data.get("duration_ms", STROBE_DEFAULT_MS))
            else:
                self._send_esp32_command(
                    {"cmd": f"led_{led}", "mode": mode,
                     "pwm": data.get("pwm", 255)})
            return jsonify({"ok": True})

        @app.route("/api/strobe", methods=["POST"])
        def _strobe():
            data = request.get_json(force=True, silent=True) or {}
            self._trigger_strobe(data.get("duration_ms", STROBE_DEFAULT_MS))
            return jsonify({"ok": True})

        @app.route("/api/room/calibrate", methods=["POST"])
        def _room_calibrate():
            """Store current WiFi RSSI as fingerprint for a named room."""
            data = request.get_json(force=True, silent=True) or {}
            room = data.get("room", "").strip()
            if not room:
                return jsonify({"error": "room name required"}), 400
            with self.sensor_lock:
                rssi = dict(self.sensor_data.get("wifi_rssi", {}))
            if not rssi:
                return jsonify({"error": "no wifi_rssi data from ESP32"}), 400
            fps = self.settings.setdefault("room_fingerprints", {})
            fps[room] = rssi
            self._save_settings()
            return jsonify({"ok": True, "room": room, "rssi": rssi})

    # ── SocketIO ──────────────────────────────────────────────────────────────

    def _setup_socketio(self):
        def _broadcaster():
            while True:
                time.sleep(2)
                with self.state_lock:
                    mode, threat, conf = (
                        self.current_mode,
                        self.current_threat,
                        self.current_confidence,
                    )
                with self.battery_lock:
                    pct = self.battery["percentage"]
                self.socketio.emit("status", {
                    "mode":        mode,
                    "threat":      threat,
                    "confidence":  round(conf, 3),
                    "battery_pct": pct,
                    "hw":          self.hw_status,
                    "room":        self._current_room,
                    "phone":       self._phone_present,
                    "connected":   True,
                    "timestamp":   datetime.now(timezone.utc).isoformat(),
                })
        self.socketio.start_background_task(_broadcaster)

    # ── start methods ─────────────────────────────────────────────────────────

    def start_threads(self):
        """Start all 8 background daemon threads."""
        threads = [
            (self._camera_loop,           "camera"),
            (self._face_recognition_loop, "face_recog"),
            (self._serial_esp32_loop,     "serial_esp32"),
            (self._serial_ipst_loop,      "serial_ipst"),
            (self._battery_monitor_loop,  "battery_monitor"),
            (self._bluetooth_manager,     "bluetooth"),
            (self._patrol_loop,           "patrol"),
            (self._wake_word_loop,        "wake_word"),
        ]
        for target, name in threads:
            threading.Thread(
                target=target, name=name, daemon=True).start()
        self.log_event(
            f"All 8 threads started — Phase {ROBOT_PHASE}.")

    def start_flask(self):
        """Blocking. Call in a daemon thread from gui.py, or directly for headless."""
        self.socketio.run(
            self.app,
            host="0.0.0.0",
            port=FLASK_PORT,
            use_reloader=False,
        )


# ── headless entry point (systemd production) ─────────────────────────────────
if __name__ == "__main__":
    core = TurtaoCore()
    core.start_threads()
    core.speak("Turtao online.")
    core.start_flask()   # blocks until process is killed
