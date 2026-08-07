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
            ("Temp (DHT22)", "temp_dht", "°C"),
            ("Humidity", "humidity", "%"),
            ("Gas (MQ-2)", "gas_mq2", "V"),
        ]

        sys_fields = [
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

        self._vars["tof_front"] = tk.StringVar(value="0")
        ttk.Label(tof_inner, text="Front:").grid(row=0, column=0, padx=(5, 2))
        ttk.Label(tof_inner, textvariable=self._vars["tof_front"], width=6).grid(
            row=0, column=1, padx=(0, 10)
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
            self._set("temp_dht", "—" if s.temp_dht is None else f"{s.temp_dht:.1f}")
            self._set("humidity", "—" if s.humidity is None else f"{s.humidity:.1f}")
            self._set("gas_mq2", f"{s.gas_mq2:.2f}")
            self._set("heading", str(self.core.state.heading))
            self._set("connected", "Yes" if self.core.state.connected else "No")
            self._set("latency_ms", str(self.core.state.latency_ms))
            self._set("mode", self.core.state.mode.value)
            self._set("threat_label", self.core.state.threat_label.value)

            self._set("tof_front", "0" if s.tof_front is None else str(s.tof_front))

    def _set(self, key: str, value: str) -> None:
        if key in self._vars:
            self._vars[key].set(value)
