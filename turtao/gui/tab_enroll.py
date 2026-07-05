import tkinter as tk
from tkinter import ttk
import cv2
from PIL import Image, ImageTk


class EnrollTab:
    def __init__(self, parent, core) -> None:
        self.core = core
        self.frame = ttk.Frame(parent)
        self._setup_ui()

    def _setup_ui(self) -> None:
        row = 0

        ttk.Label(self.frame, text="Enrollment", font=("", 14, "bold")).grid(
            row=row, column=0, columnspan=2, pady=(10, 5)
        )
        row += 1

        ttk.Label(self.frame, text="Name:").grid(row=row, column=0, sticky="w", padx=10)
        self.name_var = tk.StringVar()
        self.name_entry = ttk.Entry(self.frame, textvariable=self.name_var, width=30)
        self.name_entry.grid(row=row, column=1, sticky="w", padx=10)
        row += 1

        btn_frame = ttk.Frame(self.frame)
        btn_frame.grid(row=row, column=0, columnspan=2, pady=5)
        self.start_btn = ttk.Button(
            btn_frame, text="Start Enrollment", command=self._start
        )
        self.start_btn.pack(side=tk.LEFT, padx=4)
        self.capture_btn = ttk.Button(
            btn_frame, text="Capture Pose", command=self._capture
        )
        self.capture_btn.pack(side=tk.LEFT, padx=4)
        self.cancel_btn = ttk.Button(
            btn_frame, text="Cancel", command=self._cancel
        )
        self.cancel_btn.pack(side=tk.LEFT, padx=4)
        row += 1

        self.status_var = tk.StringVar(value="Idle")
        self.status_label = ttk.Label(
            self.frame, textvariable=self.status_var, foreground="gray"
        )
        self.status_label.grid(row=row, column=0, columnspan=2, pady=5)
        row += 1

        self.image_label = ttk.Label(self.frame)
        self.image_label.grid(row=row, column=0, columnspan=2, pady=10)

    def _start(self) -> None:
        name = self.name_var.get().strip()
        if not name:
            self.status_var.set("Enter a name first")
            return
        result = self.core.enrollment.start_enrollment(name)
        self.status_var.set(result.get("message", ""))

    def _capture(self) -> None:
        with self.core.state:
            frame = self.core.state.latest_frame
        if frame is None:
            self.status_var.set("No frame available")
            return
        result = self.core.enrollment.capture_pose(frame)
        self.status_var.set(result.get("message", ""))

    def _cancel(self) -> None:
        result = self.core.enrollment.cancel_enrollment()
        self.status_var.set(result.get("message", ""))

    def refresh(self) -> None:
        status = self.core.enrollment.get_status()
        if status.get("status") == "active":
            self.start_btn.configure(state=tk.DISABLED)
            self.capture_btn.configure(state=tk.NORMAL)
            self.cancel_btn.configure(state=tk.NORMAL)
            self.name_entry.configure(state=tk.DISABLED)
            poses = status.get("pose", 1)
            total = status.get("total_poses", 5)
            cap = status.get("captured", 0)
            self.status_var.set(
                f"Enrolling '{status.get('name')}' — pose {poses}/{total} "
                f"({cap} frames captured)"
            )
        else:
            self.start_btn.configure(state=tk.NORMAL)
            self.capture_btn.configure(state=tk.DISABLED)
            self.cancel_btn.configure(state=tk.DISABLED)
            self.name_entry.configure(state=tk.NORMAL)
            if status.get("status") != "active":
                self.status_var.set("Idle")

        self._show_frame()

    def _show_frame(self) -> None:
        with self.core.state:
            frame = self.core.state.latest_annotated_frame
        if frame is not None:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w = rgb.shape[:2]
            max_w = 320
            if w > max_w:
                scale = max_w / w
                h, w = int(h * scale), max_w
                rgb = cv2.resize(rgb, (w, h))
            img = ImageTk.PhotoImage(Image.fromarray(rgb))
            self.image_label.configure(image=img)
            self.image_label.image = img
