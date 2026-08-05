import time
import tkinter as tk
from tkinter import ttk

import cv2
from PIL import Image, ImageTk


# Slow down the video refresh to every ~100ms (10fps preview) to reduce UI lag.
# The main poll loop (33ms) will still update status text quickly.
_FRAME_REFRESH_MS = 100
_LAST_FRAME_UPDATE = [0.0]


class EnrollTab:
    def __init__(self, parent, core) -> None:
        self.core = core
        self.frame = ttk.Frame(parent)
        self._last_action_time = 0.0
        self._last_action_message = ""
        self._last_status_hash: str = ""
        self._photo_image = None  # hold a reference to prevent GC
        self._setup_ui()

    def _setup_ui(self) -> None:
        self.frame.columnconfigure(0, weight=1)
        self.frame.rowconfigure(0, weight=1)
        container = ttk.Frame(self.frame)
        container.grid(row=0, column=0, pady=20)
        row = 0

        ttk.Label(container, text="Enrollment", font=("", 16, "bold")).grid(
            row=row, column=0, columnspan=2, pady=(10, 20)
        )
        row += 1

        ttk.Label(container, text="Name:").grid(row=row, column=0, sticky="e", padx=10)
        self.name_var = tk.StringVar()
        self.name_entry = ttk.Entry(container, textvariable=self.name_var, width=30)
        self.name_entry.grid(row=row, column=1, sticky="w", padx=10)
        row += 1

        btn_frame = ttk.Frame(container)
        btn_frame.grid(row=row, column=0, columnspan=2, pady=15)
        self.start_btn = ttk.Button(
            btn_frame, text="Start Enrollment", command=self._start
        )
        self.start_btn.pack(side=tk.LEFT, padx=4)
        # Option C: label updated to reflect 5-frame burst
        self.capture_btn = ttk.Button(
            btn_frame, text="Capture Pose (5 frames)", command=self._capture
        )
        self.capture_btn.pack(side=tk.LEFT, padx=4)
        self.cancel_btn = ttk.Button(
            btn_frame, text="Cancel", command=self._cancel
        )
        self.cancel_btn.pack(side=tk.LEFT, padx=4)
        row += 1

        # Status line (e.g. "Enrolling 'Alice' — pose 2/5")
        self.status_var = tk.StringVar(value="Idle")
        self.status_label = ttk.Label(
            container, textvariable=self.status_var,
            foreground="gray", font=("", 11), wraplength=500
        )
        self.status_label.grid(row=row, column=0, columnspan=2, pady=(5, 0))
        row += 1

        # Option B: big bold action prompt ("Turn your head slightly LEFT")
        self.action_var = tk.StringVar(value="")
        self.action_label = ttk.Label(
            container, textvariable=self.action_var,
            font=("", 13, "bold"), foreground="#61AFEF", wraplength=500
        )
        self.action_label.grid(row=row, column=0, columnspan=2, pady=(2, 0))
        row += 1

        # Option B: smaller tip line ("About 20–30° — don't go too far")
        self.tip_var = tk.StringVar(value="")
        self.tip_label = ttk.Label(
            container, textvariable=self.tip_var,
            font=("", 10), foreground="#ABB2BF", wraplength=500
        )
        self.tip_label.grid(row=row, column=0, columnspan=2, pady=(0, 5))
        row += 1

        # Option B: quality warning (shown when a capture fails)
        self.quality_var = tk.StringVar(value="")
        self.quality_label = ttk.Label(
            container, textvariable=self.quality_var,
            font=("", 10, "italic"), foreground="#E06C75", wraplength=500
        )
        self.quality_label.grid(row=row, column=0, columnspan=2, pady=(0, 8))
        row += 1

        # Option D: show how many outlier frames DBSCAN rejected
        self.outlier_var = tk.StringVar(value="")
        self.outlier_label = ttk.Label(
            container, textvariable=self.outlier_var,
            font=("", 9), foreground="#56B6C2", wraplength=500
        )
        self.outlier_label.grid(row=row, column=0, columnspan=2, pady=(0, 4))
        row += 1

        # Progress bar (hidden until enrollment is active)
        self.progress_var = tk.IntVar(value=0)
        self.progress = ttk.Progressbar(
            container, variable=self.progress_var,
            maximum=5, length=300, mode="determinate"
        )
        self.progress.grid(row=row, column=0, columnspan=2, pady=(0, 8))
        row += 1

        self.image_label = ttk.Label(container)
        self.image_label.grid(row=row, column=0, columnspan=2, pady=10)

    # ------------------------------------------------------------------
    # Button callbacks
    # ------------------------------------------------------------------

    def _start(self) -> None:
        name = self.name_var.get().strip()
        if not name:
            self._flash_status("Enter a name first", error=True)
            return
        result = self.core.enrollment.start_enrollment(name)
        msg = result.get("message", "")
        if result.get("status") == "error":
            self._flash_status(msg, error=True)
        else:
            self._flash_status(msg)
            # Show first pose guide immediately
            guide = result.get("guidance", {})
            self._apply_guidance(guide)

    def _capture(self) -> None:
        with self.core.state:
            frame = self.core.state.latest_frame
        if frame is None:
            self._flash_status("No frame available", error=True)
            return
        # Disable button during burst to prevent double-clicks
        self.capture_btn.configure(state=tk.DISABLED)
        self.frame.update_idletasks()

        result = self.core.enrollment.capture_pose(lambda: self.core.state.latest_frame)
        status = result.get("status", "")
        msg = result.get("message", "")
        guide = result.get("guidance", {})

        if status == "retry":
            # Option B: show quality warning
            quality_hint = result.get("guidance") if isinstance(result.get("guidance"), str) else msg
            self._flash_status(f"Retry: {msg}", error=True)
            self.quality_var.set(quality_hint if isinstance(quality_hint, str) else "")
            self.outlier_var.set("")
        elif status == "next_pose":
            self._flash_status(msg)
            self.quality_var.set("")
            self._apply_guidance(guide)
            # Option D: show outlier count if any were discarded
            n_out = result.get("outliers_discarded", 0)
            self.outlier_var.set(
                f"🔬 DBSCAN filtered {n_out} noisy frame(s) from this pose" if n_out else ""
            )
        elif status == "complete":
            total_out = result.get("total_outliers_discarded", 0)
            suffix = f" ({total_out} noisy frame(s) filtered across all poses)" if total_out else ""
            self._flash_status(f"✓ {msg}{suffix}")
            self.quality_var.set("")
            self.action_var.set("")
            self.tip_var.set("")
            self.outlier_var.set("")
        else:
            self._flash_status(msg)
            self.quality_var.set("")
            self.outlier_var.set("")
            if isinstance(guide, dict):
                self._apply_guidance(guide)

        # Re-enable after result
        self.capture_btn.configure(state=tk.NORMAL)

    def _cancel(self) -> None:
        result = self.core.enrollment.cancel_enrollment()
        self._flash_status(result.get("message", ""))
        self.action_var.set("")
        self.tip_var.set("")
        self.quality_var.set("")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _flash_status(self, msg: str, error: bool = False) -> None:
        self._last_action_time = time.time()
        self._last_action_message = msg
        self.status_var.set(msg)
        self.status_label.configure(foreground="#E06C75" if error else "#98C379")

    def _apply_guidance(self, guide) -> None:
        """Option B: Update action + tip labels from a guidance dict."""
        if isinstance(guide, dict):
            self.action_var.set(f"➤  {guide.get('action', '')}")
            self.tip_var.set(guide.get("tip", ""))
        elif isinstance(guide, str):
            self.action_var.set(guide)
            self.tip_var.set("")

    # ------------------------------------------------------------------
    # Refresh (called every 33ms by AppWindow._poll)
    # ------------------------------------------------------------------

    def refresh(self) -> None:
        status = self.core.enrollment.get_status()
        is_active = status.get("status") == "active"
        poses = status.get("pose", 1)
        total = status.get("total_poses", 5)
        cap = status.get("captured", 0)
        quality_issue = status.get("quality_issue", "")

        # Build a quick hash to avoid unnecessary widget updates
        state_hash = f"{is_active}|{poses}|{cap}|{quality_issue}"
        if state_hash != self._last_status_hash:
            self._last_status_hash = state_hash
            self._update_widgets(is_active, status, poses, total, cap, quality_issue)

        # Throttle frame preview to reduce CPU / flickering
        now = time.time()
        if now - _LAST_FRAME_UPDATE[0] >= _FRAME_REFRESH_MS / 1000.0:
            _LAST_FRAME_UPDATE[0] = now
            self._show_frame(is_active)

    def _update_widgets(self, is_active, status, poses, total, cap, quality_issue) -> None:
        if is_active:
            self.start_btn.configure(state=tk.DISABLED)
            self.capture_btn.configure(state=tk.NORMAL)
            self.cancel_btn.configure(state=tk.NORMAL)
            self.name_entry.configure(state=tk.DISABLED)
            self.progress_var.set(poses - 1)

            # Only overwrite status text if no flash is showing
            if time.time() - self._last_action_time >= 3.0:
                self.status_var.set(
                    f"Enrolling '{status.get('name')}' — pose {poses}/{total}"
                    f"  ({cap} frames captured)"
                )
                self.status_label.configure(foreground="gray")

            # Show persistent guidance from state (don't stomp on button result)
            if time.time() - self._last_action_time >= 1.5:
                guide = status.get("guidance", {})
                self._apply_guidance(guide)

            if quality_issue:
                from turtao.vision.enrollment import QUALITY_GUIDANCE
                self.quality_var.set(QUALITY_GUIDANCE.get(quality_issue, quality_issue))
            else:
                if time.time() - self._last_action_time >= 3.0:
                    self.quality_var.set("")
        else:
            self.start_btn.configure(state=tk.NORMAL)
            self.capture_btn.configure(state=tk.DISABLED)
            self.cancel_btn.configure(state=tk.DISABLED)
            self.name_entry.configure(state=tk.NORMAL)
            self.progress_var.set(0)

            if time.time() - self._last_action_time >= 3.0:
                self.status_var.set("Idle")
                self.status_label.configure(foreground="gray")
                self.action_var.set("")
                self.tip_var.set("")
                self.quality_var.set("")

    def _show_frame(self, is_enrolling: bool) -> None:
        with self.core.state:
            frame = self.core.state.latest_frame

        if frame is None:
            return

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]

        if is_enrolling:
            # Draw a face-align oval guide
            cx, cy = w // 2, h // 2
            axes = (int(w * 0.15), int(h * 0.30))
            cv2.ellipse(rgb, (cx, cy), axes, 0, 0, 360, (97, 175, 239), 2)

            # Option B: "Align face here" text inside oval
            cv2.putText(
                rgb, "Align face here",
                (cx - 58, cy - axes[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (97, 175, 239), 1,
            )

            # Option B: draw a simple face-centering crosshair
            half = 10
            cv2.line(rgb, (cx - half, cy), (cx + half, cy), (97, 175, 239), 1)
            cv2.line(rgb, (cx, cy - half), (cx, cy + half), (97, 175, 239), 1)

        max_w = 480  # smaller preview = less CPU
        if w > max_w:
            scale = max_w / w
            h, w = int(h * scale), max_w
            rgb = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_LINEAR)

        pil_img = Image.fromarray(rgb)
        self._photo_image = ImageTk.PhotoImage(pil_img)
        self.image_label.configure(image=self._photo_image)
