"""
gui.py — Turtao Dev GUI (NEVER runs in production)
Dev-only tkinter window for testing on Pi 5 with monitor.
Imports TurtaoCore directly — no HTTP overhead for camera or state.

Usage:
    cd ~/turtao && source venv/bin/activate && python3 gui.py
"""

import tkinter as tk
from tkinter import ttk, messagebox, simpledialog
import threading
import time
import glob
import os

import cv2
from PIL import Image, ImageTk

from core import (
    TurtaoCore, POSES, UNK_DIR, FLASK_PORT, ROBOT_PHASE
)

# ── colour palette ────────────────────────────────────────────────────────────
BG      = "#0f0f1a"
BG2     = "#1a1a2e"
BG3     = "#22223a"
ACCENT  = "#4e6ef2"
GREEN   = "#22c55e"
RED     = "#ef4444"
YELLOW  = "#facc15"
GREY    = "#6b7280"
FG      = "#e2e8f0"
FG_DIM  = "#94a3b8"

# Liberation Sans is guaranteed on Raspberry Pi OS Bookworm
FONT_SM  = ("Liberation Sans",  9)
FONT_MD  = ("Liberation Sans", 11)
FONT_LG  = ("Liberation Sans", 14, "bold")

POLL_MS      = 400   # GUI refresh interval
FACE_REFRESH = 10    # refresh face list every N polls (~4 s)


class TurtaoGUI:
    """Main dev window. Receives a live TurtaoCore instance."""

    def __init__(self, root: tk.Tk, core: TurtaoCore):
        self.root  = root
        self.core  = core
        self._running     = True
        self._event_idx   = 0
        self._poll_count  = 0
        self._enrolling   = False
        self._enroll_busy = False

        root.title(f"Turtao Dev — Phase {ROBOT_PHASE}  |  port {FLASK_PORT}")
        root.configure(bg=BG)
        root.geometry("1280x780")
        root.minsize(1100, 700)

        self._build_layout()
        self._refresh_faces_list()
        self._refresh_unknowns()
        self._schedule_poll()

    # ─────────────────────────────── LAYOUT ──────────────────────────────────

    def _build_layout(self):
        # ── top bar ───────────────────────────────────────────────────────
        top = tk.Frame(self.root, bg=BG, height=50)
        top.pack(fill=tk.X, padx=12, pady=(10, 0))

        tk.Label(top, text="TURTAO",
                 font=("Liberation Sans", 18, "bold"),
                 bg=BG, fg=ACCENT).pack(side=tk.LEFT)

        tk.Label(top, text=f"Phase {ROBOT_PHASE}",
                 font=FONT_SM, bg=BG, fg=FG_DIM).pack(
            side=tk.LEFT, padx=(6, 20))

        # mode buttons
        mode_frame = tk.Frame(top, bg=BG)
        mode_frame.pack(side=tk.LEFT)
        self._mode_btns = {}
        for mode, color in [("IDLE", GREY), ("GUARD", GREEN), ("PATROL", ACCENT)]:
            b = tk.Button(
                mode_frame, text=mode, bg=color, fg="white",
                font=("Liberation Sans", 10, "bold"),
                width=8, relief=tk.FLAT, activebackground=color,
                cursor="hand2",
                command=lambda m=mode: self.core.set_mode(m),
            )
            b.pack(side=tk.LEFT, padx=3)
            self._mode_btns[mode] = b

        # state badge
        self._state_var = tk.StringVar(value="IDLE")
        self._state_lbl = tk.Label(
            top, textvariable=self._state_var,
            bg=BG3, fg=FG,
            font=("Liberation Sans", 13, "bold"),
            padx=14, pady=4, relief=tk.FLAT,
        )
        self._state_lbl.pack(side=tk.LEFT, padx=10)

        # hw status pills — right side
        hw_frame = tk.Frame(top, bg=BG)
        hw_frame.pack(side=tk.RIGHT)
        # camera, esp32, ipst, battery, bluetooth
        self._hw_labels = {}
        for key, display in [
            ("camera",    "CAM"),
            ("esp32",     "ESP32"),
            ("ipst",      "IPST"),
            ("battery",   "BATT"),
            ("bluetooth", "BT"),
        ]:
            lbl = tk.Label(
                hw_frame, text=display,
                bg=GREY, fg="white",
                font=("Liberation Sans", 8, "bold"),
                padx=6, pady=2, relief=tk.FLAT,
            )
            lbl.pack(side=tk.LEFT, padx=2)
            self._hw_labels[key] = lbl

        # battery text
        self._batt_var = tk.StringVar(value="Batt: --")
        tk.Label(top, textvariable=self._batt_var,
                 bg=BG, fg=FG_DIM, font=FONT_MD).pack(
            side=tk.RIGHT, padx=12)

        # ── main area ─────────────────────────────────────────────────────
        main = tk.Frame(self.root, bg=BG)
        main.pack(fill=tk.BOTH, expand=True, padx=12, pady=8)

        # left: camera + motor controls
        left = tk.Frame(main, bg=BG)
        left.pack(side=tk.LEFT, fill=tk.Y)

        self._cam_label = tk.Label(left, bg="#000000")
        self._cam_label.pack()
        self._show_cam_placeholder()

        self._recognition_var = tk.StringVar(value="No face detected")
        self._recognition_lbl = tk.Label(
            left, textvariable=self._recognition_var,
            bg=BG2, fg=FG_DIM, font=FONT_MD,
            width=60, anchor="w", padx=8, pady=4,
        )
        self._recognition_lbl.pack(fill=tk.X, pady=(4, 0))

        motor_frame = tk.LabelFrame(
            left, text="Manual Drive  (requires ESP32)",
            bg=BG, fg=FG_DIM, font=FONT_SM,
        )
        motor_frame.pack(fill=tk.X, pady=6)
        self._build_motor_controls(motor_frame)

        # right: notebook
        right = tk.Frame(main, bg=BG)
        right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(12, 0))

        nb = ttk.Notebook(right)
        nb.pack(fill=tk.BOTH, expand=True)

        self._tab_enroll   = tk.Frame(nb, bg=BG2)
        self._tab_faces    = tk.Frame(nb, bg=BG2)
        self._tab_unknowns = tk.Frame(nb, bg=BG2)
        self._tab_sensors  = tk.Frame(nb, bg=BG2)
        self._tab_settings = tk.Frame(nb, bg=BG2)
        self._tab_log      = tk.Frame(nb, bg=BG2)

        nb.add(self._tab_enroll,   text="  Enroll  ")
        nb.add(self._tab_faces,    text="  Faces   ")
        nb.add(self._tab_unknowns, text=" Unknowns ")
        nb.add(self._tab_sensors,  text="  Sensors ")
        nb.add(self._tab_settings, text=" Settings ")
        nb.add(self._tab_log,      text="   Log    ")

        self._build_tab_enroll()
        self._build_tab_faces()
        self._build_tab_unknowns()
        self._build_tab_sensors()
        self._build_tab_settings()
        self._build_tab_log()

        style = ttk.Style()
        style.theme_use("default")
        style.configure("TNotebook",      background=BG)
        style.configure("TNotebook.Tab",  background=BG3,
                        foreground=FG_DIM, padding=[10, 4], font=FONT_SM)
        style.map("TNotebook.Tab",
                  background=[("selected", ACCENT)],
                  foreground=[("selected", "white")])

    # ── motor controls ────────────────────────────────────────────────────────

    def _build_motor_controls(self, parent):
        grid = tk.Frame(parent, bg=BG)
        grid.pack(pady=6)
        btn_cfg = dict(bg=BG3, fg=FG,
                       font=("Liberation Sans", 12), width=5,
                       relief=tk.FLAT, activebackground=ACCENT)

        def move(ml, mr):
            if not self.core.hw_status["esp32"]:
                self._toast("ESP32 not connected")
                return
            self.core.send_esp32(self.core._build_move_cmd(ml, mr))

        tk.Button(grid, text="^", **btn_cfg,
                  command=lambda: move(0.6, 0.6)
                  ).grid(row=0, column=1, padx=2, pady=2)
        tk.Button(grid, text="<", **btn_cfg,
                  command=lambda: move(-0.5, 0.5)
                  ).grid(row=1, column=0, padx=2, pady=2)
        tk.Button(grid, text="STOP", bg=RED, fg="white",
                  font=("Liberation Sans", 9, "bold"), width=5,
                  relief=tk.FLAT,
                  command=lambda: move(0.0, 0.0)
                  ).grid(row=1, column=1, padx=2, pady=2)
        tk.Button(grid, text=">", **btn_cfg,
                  command=lambda: move(0.5, -0.5)
                  ).grid(row=1, column=2, padx=2, pady=2)
        tk.Button(grid, text="v", **btn_cfg,
                  command=lambda: move(-0.6, -0.6)
                  ).grid(row=2, column=1, padx=2, pady=2)

        # Strobe test button
        tk.Button(
            parent, text="Test Strobe", bg="#7c3aed", fg="white",
            font=FONT_SM, relief=tk.FLAT, padx=8,
            command=lambda: self.core._trigger_strobe(1000),
        ).pack(pady=(0, 4))

    # ── enroll tab ────────────────────────────────────────────────────────────

    def _build_tab_enroll(self):
        p = self._tab_enroll
        tk.Label(p, text="Face Enrollment", font=FONT_LG,
                 bg=BG2, fg=FG).pack(pady=(16, 4))
        tk.Label(p, text="5 poses required for reliable recognition.",
                 font=FONT_SM, bg=BG2, fg=FG_DIM).pack()

        form = tk.Frame(p, bg=BG2)
        form.pack(pady=12)
        tk.Label(form, text="Name:", bg=BG2, fg=FG,
                 font=FONT_MD).grid(row=0, column=0, padx=8, pady=4, sticky="e")
        self._enroll_name = tk.Entry(
            form, width=20, bg=BG3, fg=FG,
            insertbackground=FG, font=FONT_MD, relief=tk.FLAT)
        self._enroll_name.grid(row=0, column=1, padx=8, pady=4)
        self._enroll_start_btn = tk.Button(
            form, text="Start", bg=ACCENT, fg="white",
            font=("Liberation Sans", 10, "bold"),
            relief=tk.FLAT, padx=10,
            command=self._start_enrollment)
        self._enroll_start_btn.grid(row=0, column=2, padx=8)

        # pose progress row
        pose_row = tk.Frame(p, bg=BG2)
        pose_row.pack(pady=6)
        self._pose_indicators = []
        for i in range(5):
            lbl = tk.Label(pose_row, text=f"Pose {i+1}",
                           bg=BG3, fg=FG_DIM, font=FONT_SM,
                           width=12, pady=4, relief=tk.FLAT)
            lbl.grid(row=0, column=i, padx=3)
            self._pose_indicators.append(lbl)

        self._enroll_instruction = tk.Label(
            p, text="Enter a name and press Start.",
            bg=BG2, fg=YELLOW,
            font=("Liberation Sans", 11, "italic"),
            wraplength=340)
        self._enroll_instruction.pack(pady=8)

        self._capture_btn = tk.Button(
            p, text="[ Capture Pose ]",
            bg=GREEN, fg="white",
            font=("Liberation Sans", 12, "bold"),
            relief=tk.FLAT, padx=16, pady=8,
            state=tk.DISABLED,
            command=self._capture_pose)
        self._capture_btn.pack(pady=4)

        self._enroll_result = tk.Label(
            p, text="", bg=BG2, fg=FG, font=FONT_MD, wraplength=340)
        self._enroll_result.pack(pady=6)

        tk.Button(p, text="Cancel", bg=BG3, fg=FG_DIM,
                  font=FONT_SM, relief=tk.FLAT,
                  command=self._cancel_enrollment).pack(pady=2)

    # ── faces tab ─────────────────────────────────────────────────────────────

    def _build_tab_faces(self):
        p = self._tab_faces
        tk.Label(p, text="Enrolled Faces", font=FONT_LG,
                 bg=BG2, fg=FG).pack(pady=(16, 4))

        self._faces_listbox = tk.Listbox(
            p, bg=BG3, fg=FG, font=FONT_MD,
            selectbackground=ACCENT, relief=tk.FLAT,
            height=12, width=30)
        self._faces_listbox.pack(padx=20, pady=6)

        btn_row = tk.Frame(p, bg=BG2)
        btn_row.pack()
        tk.Button(btn_row, text="Refresh", bg=BG3, fg=FG,
                  font=FONT_SM, relief=tk.FLAT, padx=10,
                  command=self._refresh_faces_list).pack(side=tk.LEFT, padx=4)
        tk.Button(btn_row, text="Delete Selected", bg=RED, fg="white",
                  font=FONT_SM, relief=tk.FLAT, padx=10,
                  command=self._delete_selected_face).pack(side=tk.LEFT, padx=4)

        self._face_count_var = tk.StringVar(value="")
        tk.Label(p, textvariable=self._face_count_var,
                 bg=BG2, fg=FG_DIM, font=FONT_SM).pack(pady=4)

    # ── unknowns tab ──────────────────────────────────────────────────────────

    def _build_tab_unknowns(self):
        p = self._tab_unknowns
        tk.Label(p, text="Unknown Detections", font=FONT_LG,
                 bg=BG2, fg=FG).pack(pady=(16, 4))
        tk.Label(p, text="Select an image and promote it to a named face.",
                 font=FONT_SM, bg=BG2, fg=FG_DIM).pack()

        content = tk.Frame(p, bg=BG2)
        content.pack(fill=tk.BOTH, expand=True, padx=12, pady=8)

        self._unk_listbox = tk.Listbox(
            content, bg=BG3, fg=FG, font=FONT_SM,
            selectbackground=ACCENT, relief=tk.FLAT,
            height=10, width=28)
        self._unk_listbox.pack(side=tk.LEFT, fill=tk.Y)
        self._unk_listbox.bind("<<ListboxSelect>>", self._on_unknown_select)

        right = tk.Frame(content, bg=BG2)
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=12)

        self._unk_preview = tk.Label(
            right, bg=BG3, text="Select an image",
            fg=FG_DIM, width=20, height=10)
        self._unk_preview.pack()

        tk.Button(right, text="Promote to Named Face",
                  bg=GREEN, fg="white",
                  font=("Liberation Sans", 10, "bold"),
                  relief=tk.FLAT, padx=10, pady=4,
                  command=self._promote_unknown).pack(pady=8)

        tk.Button(right, text="Delete", bg=RED, fg="white",
                  font=FONT_SM, relief=tk.FLAT, padx=10,
                  command=self._delete_unknown).pack()

        tk.Button(p, text="Refresh", bg=BG3, fg=FG_DIM,
                  font=FONT_SM, relief=tk.FLAT, padx=8,
                  command=self._refresh_unknowns).pack(pady=6)

        self._unk_files = []

    # ── sensors tab ───────────────────────────────────────────────────────────

    def _build_tab_sensors(self):
        p = self._tab_sensors
        tk.Label(p, text="Sensor Readings", font=FONT_LG,
                 bg=BG2, fg=FG).pack(pady=(16, 4))

        # Always-visible ESP32 connection warning at top (no pack/pack_forget)
        self._esp32_warn_var = tk.StringVar(
            value="[!] ESP32 not connected -- sensor readings unavailable.")
        self._esp32_warn_lbl = tk.Label(
            p, textvariable=self._esp32_warn_var,
            bg="#3a2a00", fg=YELLOW, font=FONT_SM,
            anchor="w", padx=8, pady=3)
        self._esp32_warn_lbl.pack(fill=tk.X, padx=16, pady=(0, 4))

        grid = tk.Frame(p, bg=BG2)
        grid.pack(padx=16, pady=4, fill=tk.X)

        SENSOR_ROWS = [
            ("temp_dht",  "Temp DHT (C)"),
            ("humid",     "Humidity (%)"),
            ("temp_bmp",  "Temp BMP (C)"),
            ("pressure",  "Pressure (hPa)"),
            ("gas_mq2",   "Gas MQ-2"),
            ("gas_mq135", "Gas MQ-135"),
            ("sound",     "Sound"),
            ("pir",       "PIR Motion"),
            ("tof_fl",    "ToF Front-L (mm)"),
            ("tof_fc",    "ToF Front-C (mm)"),
            ("tof_fr",    "ToF Front-R (mm)"),
            ("tof_down",  "ToF Down (mm)"),
            ("accel_x",   "Accel X"),
            ("accel_y",   "Accel Y"),
            ("accel_z",   "Accel Z"),
            ("gyro_x",    "Gyro X"),
            ("gyro_y",    "Gyro Y"),
            ("gyro_z",    "Gyro Z"),
        ]

        self._sensor_vars = {}
        for i, (key, label) in enumerate(SENSOR_ROWS):
            row, col = divmod(i, 3)
            tk.Label(grid, text=label + ":", bg=BG2, fg=FG_DIM,
                     font=FONT_SM, anchor="e", width=18).grid(
                row=row, column=col * 2, padx=(8, 2), pady=2, sticky="e")
            var = tk.StringVar(value="--")
            tk.Label(grid, textvariable=var, bg=BG2, fg=ACCENT,
                     font=FONT_SM, anchor="w", width=10).grid(
                row=row, column=col * 2 + 1, padx=(2, 12), pady=2, sticky="w")
            self._sensor_vars[key] = var

        tk.Frame(p, bg=BG3, height=1).pack(fill=tk.X, padx=16, pady=8)
        tk.Label(p, text="Battery  (INA219 on ESP32 I2C)",
                 font=("Liberation Sans", 11, "bold"),
                 bg=BG2, fg=FG).pack()
        self._batt_detail_var = tk.StringVar(value="No data from ESP32")
        tk.Label(p, textvariable=self._batt_detail_var,
                 bg=BG2, fg=YELLOW, font=FONT_MD).pack(pady=2)

        # Room display
        tk.Frame(p, bg=BG3, height=1).pack(fill=tk.X, padx=16, pady=4)
        room_row = tk.Frame(p, bg=BG2)
        room_row.pack()
        tk.Label(room_row, text="Estimated room:", bg=BG2, fg=FG_DIM,
                 font=FONT_SM).pack(side=tk.LEFT, padx=4)
        self._room_var = tk.StringVar(value="unknown")
        tk.Label(room_row, textvariable=self._room_var, bg=BG2,
                 fg=ACCENT, font=FONT_MD).pack(side=tk.LEFT)

    # ── settings tab ──────────────────────────────────────────────────────────

    def _build_tab_settings(self):
        p = self._tab_settings
        tk.Label(p, text="Settings", font=FONT_LG,
                 bg=BG2, fg=FG).pack(pady=(16, 8))

        form = tk.Frame(p, bg=BG2)
        form.pack(padx=20, fill=tk.X)
        self._settings_vars = {}

        def add_row(label, key, kind="check", **kw):
            fr = tk.Frame(form, bg=BG2)
            fr.pack(fill=tk.X, pady=3)
            tk.Label(fr, text=label, bg=BG2, fg=FG, font=FONT_MD,
                     width=30, anchor="w").pack(side=tk.LEFT)
            if kind == "check":
                var = tk.BooleanVar(
                    value=self.core.settings.get(key, True))
                tk.Checkbutton(fr, variable=var, bg=BG2,
                               activebackground=BG2,
                               command=self._save_settings).pack(side=tk.LEFT)
            elif kind == "scale":
                var = tk.DoubleVar(
                    value=self.core.settings.get(key, kw.get("default", 0.5)))
                tk.Scale(fr, variable=var,
                         from_=kw.get("from_", 0), to=kw.get("to", 1),
                         resolution=kw.get("res", 0.01),
                         orient=tk.HORIZONTAL, bg=BG2, fg=FG,
                         highlightthickness=0, length=160,
                         command=lambda _: self._save_settings()
                         ).pack(side=tk.LEFT)
            elif kind == "entry":
                var = tk.StringVar(
                    value=str(self.core.settings.get(key, "")))
                en = tk.Entry(fr, textvariable=var, width=22,
                              bg=BG3, fg=FG, insertbackground=FG,
                              font=FONT_MD, relief=tk.FLAT)
                en.pack(side=tk.LEFT)
                en.bind("<FocusOut>", lambda _: self._save_settings())
                en.bind("<Return>",   lambda _: self._save_settings())
            self._settings_vars[key] = var

        add_row("TTS enabled",               "tts_enabled")
        add_row("TTS on threat",             "tts_threat")
        add_row("TTS on gas alert",          "tts_gas")
        add_row("TTS on tamper",             "tts_tamper")
        add_row("TTS on patrol start",       "tts_patrol")
        add_row("TTS on wake word",          "tts_wake")
        add_row("Stealth mode (no TTS/LEDs)","stealth_mode")
        add_row("Strobe on threat",          "strobe_on_threat")
        add_row("BLE auto arm/disarm",       "ble_auto_disarm")
        add_row("Recognition tolerance",
                "tolerance", "scale", from_=0.3, to=0.9, res=0.01, default=0.52)
        add_row("Patrol speed",
                "speed", "scale", from_=0.1, to=0.8, res=0.05, default=0.8)
        add_row("JBL MAC address",           "jbl_mac", "entry")

        btn_row = tk.Frame(p, bg=BG2)
        btn_row.pack(pady=12)
        tk.Button(btn_row, text="Test TTS", bg=BG3, fg=FG,
                  font=FONT_SM, relief=tk.FLAT, padx=10,
                  command=lambda: self.core.speak(
                      "Turtao test. Systems nominal.")
                  ).pack(side=tk.LEFT, padx=6)
        tk.Button(btn_row, text="Test Strobe (1s)", bg="#7c3aed", fg="white",
                  font=FONT_SM, relief=tk.FLAT, padx=10,
                  command=lambda: self.core._trigger_strobe(1000)
                  ).pack(side=tk.LEFT, padx=6)

    # ── log tab ───────────────────────────────────────────────────────────────

    def _build_tab_log(self):
        p = self._tab_log
        tk.Label(p, text="Event Log", font=FONT_LG,
                 bg=BG2, fg=FG).pack(pady=(16, 4))

        frame = tk.Frame(p, bg=BG2)
        frame.pack(fill=tk.BOTH, expand=True, padx=12, pady=4)
        sb = tk.Scrollbar(frame)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        self._log_text = tk.Text(
            frame, bg=BG3, fg=FG,
            font=("Courier", 9),
            state=tk.DISABLED, relief=tk.FLAT,
            yscrollcommand=sb.set)
        self._log_text.pack(fill=tk.BOTH, expand=True)
        sb.config(command=self._log_text.yview)

        tk.Button(p, text="Clear", bg=BG3, fg=FG_DIM,
                  font=FONT_SM, relief=tk.FLAT, padx=8,
                  command=self._clear_log).pack(pady=4)

    # ─────────────────────────────── POLLING ─────────────────────────────────

    def _schedule_poll(self):
        if self._running:
            self._poll()
            self.root.after(POLL_MS, self._schedule_poll)

    def _poll(self):
        self._poll_count += 1
        self._update_camera()
        self._update_state_badge()
        self._update_hw_pills()
        self._update_battery()
        self._update_sensors()
        self._update_log()
        if self._poll_count % FACE_REFRESH == 0:
            self._refresh_faces_list()

    # ── camera ────────────────────────────────────────────────────────────────

    def _show_cam_placeholder(self):
        img   = Image.new("RGB", (640, 480), color=(13, 13, 26))
        photo = ImageTk.PhotoImage(img)
        self._cam_label.config(image=photo, text="")
        self._cam_label.image = photo

    def _update_camera(self):
        with self.core.frame_lock:
            ann = self.core.latest_annotated_frame
            raw = self.core.latest_frame
            frame = ann if ann is not None else raw
        if frame is None:
            return
        try:
            rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img   = Image.fromarray(rgb)
            if img.size != (640, 480):
                img = img.resize((640, 480))
            photo = ImageTk.PhotoImage(img)
            self._cam_label.config(image=photo)
            self._cam_label.image = photo
        except Exception:
            pass

    # ── state badge ───────────────────────────────────────────────────────────

    def _update_state_badge(self):
        with self.core.state_lock:
            mode   = self.core.current_mode
            threat = self.core.current_threat
            conf   = self.core.current_confidence
            name   = self.core.current_name

        if threat:
            badge_text = f"!! THREAT  {name} ({conf:.0%})"
            badge_bg   = RED
            recog_text = f"UNKNOWN  conf {conf:.2f}"
            recog_fg   = RED
        elif mode == "GUARD":
            badge_text = "[ GUARD ]"
            badge_bg   = GREEN
            recog_text = (f"OK  {name}  conf {conf:.2f}"
                          if name != "Unknown" else "Scanning...")
            recog_fg   = GREEN if name != "Unknown" else FG_DIM
        elif mode == "PATROL":
            badge_text = "> PATROL"
            badge_bg   = ACCENT
            recog_text = (f"OK  {name}  conf {conf:.2f}"
                          if name != "Unknown" else "Scanning...")
            recog_fg   = GREEN if name != "Unknown" else FG_DIM
        else:
            badge_text = "  IDLE"
            badge_bg   = GREY
            recog_text = "No face detected"
            recog_fg   = FG_DIM

        self._state_var.set(badge_text)
        self._state_lbl.config(bg=badge_bg)
        self._recognition_var.set(recog_text)
        self._recognition_lbl.config(fg=recog_fg)

        for m, btn in self._mode_btns.items():
            btn.config(relief=tk.SUNKEN if m == mode else tk.FLAT)

    # ── hw pills ──────────────────────────────────────────────────────────────

    def _update_hw_pills(self):
        hw = self.core.hw_status
        for key, lbl in self._hw_labels.items():
            ok = hw.get(key, False)
            lbl.config(bg=GREEN if ok else GREY,
                       fg="white" if ok else "#aaaaaa")

    # ── battery ───────────────────────────────────────────────────────────────

    def _update_battery(self):
        with self.core.battery_lock:
            b = dict(self.core.battery)
        v   = b.get("voltage", 0.0)
        pct = b.get("percentage", 0)
        if v == 0.0:
            self._batt_var.set("Batt: --")
            self._batt_detail_var.set("No data from ESP32")
            return
        danger = "  [LOW]" if b.get("danger") else ""
        chg    = "  Charging" if b.get("charging") else ""
        self._batt_var.set(f"Batt: {pct}%{danger}")
        self._batt_detail_var.set(
            f"{pct}%   {v:.2f} V   {b.get('current_ma', 0):.0f} mA{chg}")

    # ── sensors ───────────────────────────────────────────────────────────────

    def _update_sensors(self):
        if not self.core.hw_status["esp32"]:
            self._esp32_warn_var.set(
                "[!] ESP32 not connected -- sensor readings unavailable.")
            self._esp32_warn_lbl.config(bg="#3a2a00")
            for var in self._sensor_vars.values():
                var.set("--")
            return
        self._esp32_warn_var.set("")
        self._esp32_warn_lbl.config(bg=BG2)
        with self.core.sensor_lock:
            data = dict(self.core.sensor_data)
        for key, var in self._sensor_vars.items():
            val = data.get(key, "--")
            var.set(f"{val:.1f}" if isinstance(val, float) else str(val))
        self._room_var.set(self.core._current_room)

    # ── log ───────────────────────────────────────────────────────────────────

    def _update_log(self):
        events = self.core.get_events(since_index=self._event_idx)
        if not events:
            return
        self._event_idx += len(events)
        self._log_text.config(state=tk.NORMAL)
        for ts, msg in events:
            self._log_text.insert(tk.END, f"[{ts}] {msg}\n")
        self._log_text.see(tk.END)
        self._log_text.config(state=tk.DISABLED)

    def _clear_log(self):
        self._log_text.config(state=tk.NORMAL)
        self._log_text.delete("1.0", tk.END)
        self._log_text.config(state=tk.DISABLED)

    # ─────────────────────────────── ENROLLMENT ──────────────────────────────

    def _start_enrollment(self):
        name = self._enroll_name.get().strip()
        if not name:
            self._toast("Enter a name first.")
            return
        if not self.core.hw_status["camera"]:
            self._toast("No camera connected.")
            return
        self.core.enrollment_session = {"name": name, "pose": 0, "total": 5}
        self._enrolling = True
        self._enroll_name.config(state=tk.DISABLED)
        self._enroll_start_btn.config(state=tk.DISABLED)
        self._update_enroll_ui(0)
        self.core.log_event(f"Enrollment started for {name}")

    def _update_enroll_ui(self, pose: int):
        for i, lbl in enumerate(self._pose_indicators):
            if i < pose:
                lbl.config(bg=GREEN, fg="white", text=f"OK {i+1}")
            elif i == pose:
                lbl.config(bg=ACCENT, fg="white", text=f"-> {i+1}")
            else:
                lbl.config(bg=BG3, fg=FG_DIM, text=f"Pose {i+1}")
        if pose < 5:
            self._enroll_instruction.config(
                text=f"Pose {pose+1}/5:  {POSES[pose]}")
            self._capture_btn.config(state=tk.NORMAL)
        else:
            self._enroll_instruction.config(text="Enrollment complete.")
            self._capture_btn.config(state=tk.DISABLED)

    def _capture_pose(self):
        if self._enroll_busy:
            return
        self._enroll_busy = True
        self._capture_btn.config(state=tk.DISABLED, text="Capturing...")
        self._enroll_result.config(text="Hold still -- capturing...", fg=YELLOW)

        def _do():
            try:
                result = self.core._do_enroll_capture()
            except Exception as e:
                result = {"success": False, "reason": str(e)}
            finally:
                self._enroll_busy = False
            self.root.after(0, lambda: self._on_capture_result(result))

        threading.Thread(target=_do, daemon=True).start()

    def _on_capture_result(self, result: dict):
        self._capture_btn.config(text="[ Capture Pose ]")
        if not result.get("success"):
            reason = result.get("reason", "unknown error")
            self._enroll_result.config(
                text=f"Error: {reason}. Try again.", fg=RED)
            self._capture_btn.config(
                state=tk.NORMAL if self.core.enrollment_session
                else tk.DISABLED)
            return
        if result.get("complete"):
            self._enroll_result.config(
                text="Done! Enrollment complete.", fg=GREEN)
            for lbl in self._pose_indicators:
                lbl.config(bg=GREEN, fg="white")
            self._capture_btn.config(state=tk.DISABLED)
            self._enrolling = False
            self._enroll_name.delete(0, tk.END)
            self._enroll_name.config(state=tk.NORMAL)
            self._enroll_start_btn.config(state=tk.NORMAL)
            self._refresh_faces_list()
        else:
            next_pose = result.get("pose", 0)
            self._enroll_result.config(
                text=f"Pose {next_pose}/5 saved. Continue.", fg=GREEN)
            self._update_enroll_ui(next_pose)

    def _cancel_enrollment(self):
        self.core.enrollment_session = None
        self._enrolling   = False
        self._enroll_busy = False
        self._enroll_result.config(text="Cancelled.", fg=FG_DIM)
        self._capture_btn.config(state=tk.DISABLED,
                                  text="[ Capture Pose ]")
        self._enroll_name.config(state=tk.NORMAL)
        self._enroll_start_btn.config(state=tk.NORMAL)
        for i, lbl in enumerate(self._pose_indicators):
            lbl.config(bg=BG3, fg=FG_DIM, text=f"Pose {i+1}")
        self._enroll_instruction.config(
            text="Enter a name and press Start.")

    # ── faces ─────────────────────────────────────────────────────────────────

    def _refresh_faces_list(self):
        profiles = self.core._load_profiles()
        self._faces_listbox.delete(0, tk.END)
        for name in sorted(profiles):
            self._faces_listbox.insert(tk.END, name)
        n = len(profiles)
        self._face_count_var.set(
            f"{n} face{'s' if n != 1 else ''} enrolled")

    def _delete_selected_face(self):
        sel = self._faces_listbox.curselection()
        if not sel:
            self._toast("Select a face to delete.")
            return
        name = self._faces_listbox.get(sel[0])
        if not messagebox.askyesno(
                "Delete", f"Delete all data for '{name}'?"):
            return
        self.core._do_delete_face(name)
        self._refresh_faces_list()

    # ── unknowns ──────────────────────────────────────────────────────────────

    def _refresh_unknowns(self):
        files = sorted(glob.glob(os.path.join(UNK_DIR, "*.jpg")))
        self._unk_files = []
        self._unk_listbox.delete(0, tk.END)
        for path in files:
            uid   = os.path.splitext(os.path.basename(path))[0]
            ts    = int(uid) / 1000 if uid.isdigit() else 0
            label = (time.strftime("%H:%M:%S", time.localtime(ts))
                     if ts else uid)
            self._unk_listbox.insert(tk.END, label)
            self._unk_files.append((uid, path))
        self._unk_preview.config(image="", text="Select an image")
        self._unk_preview.image = None

    def _on_unknown_select(self, _event):
        sel = self._unk_listbox.curselection()
        if not sel:
            return
        _, path = self._unk_files[sel[0]]
        try:
            img   = Image.open(path).resize((160, 160))
            photo = ImageTk.PhotoImage(img)
            self._unk_preview.config(image=photo, text="")
            self._unk_preview.image = photo
        except Exception:
            pass

    def _promote_unknown(self):
        sel = self._unk_listbox.curselection()
        if not sel:
            self._toast("Select an image first.")
            return
        uid, _ = self._unk_files[sel[0]]
        name = simpledialog.askstring(
            "Promote", "Name for this face:", parent=self.root)
        if not name or not name.strip():
            return
        ok, msg = self.core._do_promote(uid, name.strip())
        if ok:
            self._toast(f"Promoted to '{name.strip()}'")
            self._refresh_unknowns()
            self._refresh_faces_list()
        else:
            self._toast(f"Failed: {msg}")

    def _delete_unknown(self):
        sel = self._unk_listbox.curselection()
        if not sel:
            self._toast("Select an image first.")
            return
        _, path = self._unk_files[sel[0]]
        if os.path.exists(path):
            os.remove(path)
        self._toast("Deleted.")
        self._refresh_unknowns()

    # ── settings ──────────────────────────────────────────────────────────────

    def _save_settings(self):
        for key, var in self._settings_vars.items():
            self.core.settings[key] = var.get()
        self.core._save_settings()

    # ── helpers ───────────────────────────────────────────────────────────────

    def _toast(self, msg: str):
        self.core.log_event(msg)

    def on_close(self):
        self._running = False
        self.root.destroy()


# ── dev entry point ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    import threading

    core = TurtaoCore()
    core.start_threads()

    flask_thread = threading.Thread(
        target=core.start_flask, name="flask", daemon=True)
    flask_thread.start()

    core.log_event(f"Flask API on port {FLASK_PORT}")

    root = tk.Tk()
    app  = TurtaoGUI(root, core)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()
