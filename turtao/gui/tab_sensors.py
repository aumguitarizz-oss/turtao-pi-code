import tkinter as tk
from tkinter import ttk


class SensorsTab:
    def __init__(self, parent, core) -> None:
        self.core = core
        self.frame = ttk.Frame(parent)
        self._vars: dict[str, tk.StringVar] = {}
        self._setup_ui()

    def _setup_ui(self) -> None:
        ttk.Label(self.frame, text="Sensor Data", font=("", 14, "bold")).pack(
            pady=(10, 5)
        )

        main = ttk.Frame(self.frame)
        main.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)

        left = ttk.LabelFrame(main, text="Environmental")
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)

        right = ttk.LabelFrame(main, text="System")
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)

        env_fields = [
            ("Temperature", "temp_c", "°C"),
            ("Humidity", "humidity_pct", "%"),
            ("Pressure", "pressure_hpa", "hPa"),
            ("Gas (MQ-2)", "gas_mq2", ""),
            ("Air Quality (MQ-135)", "air_quality_mq135", ""),
            ("Sound Level", "sound_level", ""),
            ("Motion", "motion", ""),
        ]

        sys_fields = [
            ("Battery", "battery_pct", "%"),
            ("Voltage", "voltage", "V"),
            ("Current", "current_ma", "mA"),
            ("Motor Controller", "motor_controller_ok", ""),
            ("Firmware", "firmware_version", ""),
            ("Pan", "pan", "°"),
            ("Tilt", "tilt", "°"),
            ("Heading", "heading", "°"),
            ("Connected", "connected", ""),
            ("Latency", "latency_ms", "ms"),
            ("Mode", "mode", ""),
            ("Threat Label", "threat_label", ""),
        ]

        tof_frame = ttk.LabelFrame(self.frame, text="ToF Distance (cm)")
        tof_frame.pack(fill=tk.X, padx=20, pady=5)

        tof_inner = ttk.Frame(tof_frame)
        tof_inner.pack(padx=10, pady=5)

        for label, key, unit in env_fields:
            self._add_row(left, label, key, unit)

        for label, key, unit in sys_fields:
            self._add_row(right, label, key, unit)

        tof_labels = ["Front-Left", "Front-Center", "Front-Right", "Down"]
        for i, lbl in enumerate(tof_labels):
            key = f"tof_{i}"
            self._vars[key] = tk.StringVar(value="0")
            ttk.Label(tof_inner, text=f"{lbl}:").grid(row=0, column=i * 2, padx=(5, 2))
            ttk.Label(tof_inner, textvariable=self._vars[key], width=6).grid(
                row=0, column=i * 2 + 1, padx=(0, 10)
            )

    def _add_row(
        self, parent: ttk.Frame, label: str, key: str, unit: str
    ) -> None:
        row_frame = ttk.Frame(parent)
        row_frame.pack(fill=tk.X, padx=10, pady=2)

        ttk.Label(row_frame, text=f"{label}:", width=20, anchor="w").pack(side=tk.LEFT)

        var = tk.StringVar(value="—")
        self._vars[key] = var
        lbl = ttk.Label(row_frame, textvariable=var, width=10, anchor="e")
        lbl.pack(side=tk.LEFT)

        if unit:
            ttk.Label(row_frame, text=unit, width=4, anchor="w").pack(side=tk.LEFT)

    def refresh(self) -> None:
        with self.core.state:
            s = self.core.state.sensor_data
            self._set("temp_c", f"{s.temp_c:.1f}")
            self._set("humidity_pct", str(s.humidity_pct))
            self._set("pressure_hpa", f"{s.pressure_hpa:.1f}")
            self._set("gas_mq2", str(s.gas_mq2))
            self._set("air_quality_mq135", str(s.air_quality_mq135))
            self._set("sound_level", f"{s.sound_level:.1f}")
            self._set("motion", "Yes" if s.motion else "No")
            self._set("battery_pct", f"{s.battery_pct:.1f}")
            self._set("voltage", f"{s.voltage:.2f}")
            self._set("current_ma", str(s.current_ma))
            self._set("motor_controller_ok", "OK" if s.motor_controller_ok else "FAIL")
            self._set("firmware_version", s.firmware_version)
            self._set("pan", str(self.core.state.pan))
            self._set("tilt", str(self.core.state.tilt))
            self._set("heading", str(self.core.state.heading))
            self._set("connected", "Yes" if self.core.state.connected else "No")
            self._set("latency_ms", str(self.core.state.latency_ms))
            self._set("mode", self.core.state.mode.value)
            self._set("threat_label", self.core.state.threat_label.value)

            tof = s.tof_cm
            for i in range(4):
                self._set(f"tof_{i}", str(tof[i]) if i < len(tof) else "0")

    def _set(self, key: str, value: str) -> None:
        if key in self._vars:
            self._vars[key].set(value)
