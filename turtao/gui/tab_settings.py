import tkinter as tk
from tkinter import ttk


class SettingsTab:
    def __init__(self, parent, core) -> None:
        self.core = core
        self.frame = ttk.Frame(parent)
        self._bool_vars: dict[str, tk.BooleanVar] = {}
        self._float_vars: dict[str, tk.DoubleVar] = {}
        self._slider_labels: dict[str, tk.StringVar] = {}
        self._setup_ui()

    def _setup_ui(self) -> None:
        ttk.Label(self.frame, text="Settings", font=("", 14, "bold")).pack(
            pady=(10, 5)
        )

        main = ttk.Frame(self.frame)
        main.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)

        toggles_frame = ttk.LabelFrame(main, text="Toggles")
        toggles_frame.pack(fill=tk.X, pady=5)

        bool_fields = [
            ("Anti-Spoof", "anti_spoof_enabled", "anti_spoof_enabled"),
            ("Safe Mode", "safe_mode", "safe_mode"),
            ("Auto Flashbang", "auto_flashbang", "auto_flashbang"),
            ("Stealth Mode", "stealth_mode", "stealth_mode"),
        ]

        self._bool_tts_vars: dict[str, tk.BooleanVar] = {}
        tts_frame = ttk.LabelFrame(main, text="TTS Event Toggles")
        tts_frame.pack(fill=tk.X, pady=5)

        tts_fields = [
            ("Threat", "threat_tts"),
            ("Low Battery", "low_battery_tts"),
            ("Intruder", "intruder_tts"),
        ]

        notif_frame = ttk.LabelFrame(main, text="Notification Toggles")
        notif_frame.pack(fill=tk.X, pady=5)

        notif_fields = [
            ("Threat", "notif_threat"),
            ("Gas Danger", "notif_gas"),
            ("Low Battery", "notif_low_battery"),
            ("Tamper", "notif_tamper"),
            ("Connection Lost", "notif_connection"),
        ]

        audio_frame = ttk.LabelFrame(main, text="Audio")
        audio_frame.pack(fill=tk.X, pady=5)

        audio_row = ttk.Frame(audio_frame)
        audio_row.pack(fill=tk.X, padx=10, pady=5)

        ttk.Button(
            audio_row, text="🔊 Test Speaker", command=self._test_speaker
        ).pack(side=tk.LEFT)

        self._speaker_status_var = tk.StringVar(value="")
        ttk.Label(audio_row, textvariable=self._speaker_status_var).pack(
            side=tk.LEFT, padx=10
        )

        sliders_frame = ttk.LabelFrame(main, text="Sliders")
        sliders_frame.pack(fill=tk.X, pady=5)

        slider_fields = [
            ("Intercom Volume", "intercom_volume", 0.0, 1.0),
            ("Face Tolerance", "face_tolerance", 0.0, 1.0),
            ("Speed", "speed", 0.0, 1.0),
        ]

        for label, attr, _ in bool_fields:
            var = tk.BooleanVar()
            self._bool_vars[attr] = var
            cb = ttk.Checkbutton(
                toggles_frame,
                text=label,
                variable=var,
                command=lambda a=attr: self._toggle_bool(a),
            )
            cb.pack(anchor=tk.W, padx=10, pady=2)

        for label, key in tts_fields:
            var = tk.BooleanVar()
            self._bool_tts_vars[key] = var
            cb = ttk.Checkbutton(
                tts_frame,
                text=label,
                variable=var,
                command=lambda k=key: self._toggle_tts(k),
            )
            cb.pack(anchor=tk.W, padx=10, pady=2)

        for label, key in notif_fields:
            var = tk.BooleanVar()
            self._bool_vars[f"notifications.{key}"] = var
            cb = ttk.Checkbutton(
                notif_frame,
                text=label,
                variable=var,
                command=lambda k=key: self._toggle_notif(k),
            )
            cb.pack(anchor=tk.W, padx=10, pady=2)

        for label, attr, low, high in slider_fields:
            row = ttk.Frame(sliders_frame)
            row.pack(fill=tk.X, padx=10, pady=5)

            ttk.Label(row, text=f"{label}:", width=20, anchor="w").pack(side=tk.LEFT)

            var = tk.DoubleVar()
            self._float_vars[attr] = var

            lbl_var = tk.StringVar(value="—")
            self._slider_labels[attr] = lbl_var
            ttk.Label(row, textvariable=lbl_var, width=6).pack(side=tk.RIGHT)

            slider = ttk.Scale(
                row,
                from_=low,
                to=high,
                orient=tk.HORIZONTAL,
                variable=var,
                command=lambda v, a=attr: self._slider_changed(a),
            )
            slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)

    def _toggle_bool(self, attr: str) -> None:
        val = self._bool_vars[attr].get()
        setattr(self.core.settings, attr, val)
        try:
            from turtao.config import save_settings
            save_settings(self.core.settings)
        except Exception:
            pass

    def _toggle_tts(self, key: str) -> None:
        mapping = {
            "threat_tts": "threat",
            "low_battery_tts": "low_battery",
            "intruder_tts": "intruder",
        }
        attr = mapping.get(key, key)
        val = self._bool_tts_vars[key].get()
        setattr(self.core.settings.tts_event_toggles, attr, val)
        try:
            from turtao.config import save_settings
            save_settings(self.core.settings)
        except Exception:
            pass

    def _toggle_notif(self, key: str) -> None:
        mapping = {
            "notif_threat": "threat",
            "notif_gas": "gas_danger",
            "notif_low_battery": "low_battery",
            "notif_tamper": "tamper",
            "notif_connection": "connection_lost",
        }
        val = self._bool_vars[f"notifications.{key}"].get()
        real_key = mapping.get(key, key)
        setattr(self.core.settings.notifications, real_key, val)
        try:
            from turtao.config import save_settings
            save_settings(self.core.settings)
        except Exception:
            pass

    def _slider_changed(self, attr: str) -> None:
        val = round(self._float_vars[attr].get(), 3)
        setattr(self.core.settings, attr, val)
        self._slider_labels[attr].set(f"{val:.2f}")
        try:
            from turtao.config import save_settings
            save_settings(self.core.settings)
        except Exception:
            pass

    def _test_speaker(self) -> None:
        try:
            self.core.tts.speak("Speaker test. One, two, three.")
            self._speaker_status_var.set("🔊 Playing...")
        except Exception as e:
            self._speaker_status_var.set(f"⚠️ Failed: {e}")
            return
        self.frame.after(3000, lambda: self._speaker_status_var.set(""))

    def refresh(self) -> None:
        s = self.core.settings
        mapping = {
            "notif_threat": "threat",
            "notif_gas": "gas_danger",
            "notif_low_battery": "low_battery",
            "notif_tamper": "tamper",
            "notif_connection": "connection_lost",
        }
        for attr in self._bool_vars:
            if attr.startswith("notifications."):
                key = attr.split(".", 1)[1]
                real_key = mapping.get(key, key)
                val = getattr(s.notifications, real_key, False)
                self._bool_vars[attr].set(val)
            else:
                val = getattr(s, attr, False)
                self._bool_vars[attr].set(val)

        for k, var in self._bool_tts_vars.items():
            mapping = {
                "threat_tts": "threat",
                "low_battery_tts": "low_battery",
                "intruder_tts": "intruder",
            }
            attr = mapping.get(k, k)
            var.set(getattr(s.tts_event_toggles, attr, False))

        for attr in self._float_vars:
            val = getattr(s, attr, 0.0)
            self._float_vars[attr].set(val)
            self._slider_labels[attr].set(f"{val:.2f}")
