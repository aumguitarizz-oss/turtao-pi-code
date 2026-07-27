import tkinter as tk
from tkinter import ttk

import cv2
from PIL import Image, ImageTk

from turtao.state import Mode


class RecognitionTab:
    def __init__(self, parent, core) -> None:
        self.core = core
        self.frame = ttk.Frame(parent)
        self._setup_ui()

    def _setup_ui(self) -> None:
        self.frame.columnconfigure(0, weight=1)
        self.frame.rowconfigure(0, weight=1)
        container = ttk.Frame(self.frame)
        container.grid(row=0, column=0, pady=20)
        row = 0

        ttk.Label(container, text="Live Recognition", font=("", 16, "bold")).grid(
            row=row, column=0, pady=(10, 20)
        )
        row += 1

        self.recog_var = tk.BooleanVar(value=False)
        self.recog_btn = ttk.Checkbutton(
            container, text="Enable Face Recognition (GUARD Mode)", variable=self.recog_var, command=self._toggle_recog, style="Switch.TCheckbutton"
        )
        self.recog_btn.grid(row=row, column=0, pady=5)
        row += 1

        opt_frame = ttk.Frame(container)
        opt_frame.grid(row=row, column=0, pady=5)

        self.yolo_var = tk.BooleanVar(value=True)
        self.yolo_btn = ttk.Checkbutton(opt_frame, text="YOLO Tracker", variable=self.yolo_var, command=self._toggle_opts)
        self.yolo_btn.pack(side=tk.LEFT, padx=10)

        self.mp_var = tk.BooleanVar(value=False)
        self.mp_btn = ttk.Checkbutton(opt_frame, text="MediaPipe Pose", variable=self.mp_var, command=self._toggle_opts)
        self.mp_btn.pack(side=tk.LEFT, padx=10)
        row += 1

        self.image_label = ttk.Label(container)
        self.image_label.grid(row=row, column=0, pady=10)

    def _toggle_recog(self) -> None:
        if self.recog_var.get():
            self.core.set_mode(Mode.GUARD)
        else:
            self.core.set_mode(Mode.IDLE)

    def _toggle_opts(self) -> None:
        with self.core.state:
            self.core.state.show_yolo = self.yolo_var.get()
            self.core.state.show_mediapipe = self.mp_var.get()

    def refresh(self) -> None:
        with self.core.state:
            current_mode = self.core.state.mode
        self.recog_var.set(current_mode == Mode.GUARD)
        self._show_frame()

    def _show_frame(self) -> None:
        with self.core.state:
            frame = self.core.state.latest_frame
            threat_label = self.core.state.threat_label
            threat_box = self.core.state.threat_state.box
            persons = getattr(self.core.state, "latest_persons", [])
            show_yolo = getattr(self.core.state, "show_yolo", True)
            show_mp = getattr(self.core.state, "show_mediapipe", False)

        if frame is not None:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            if show_yolo:
                for p in persons:
                    px1, py1, px2, py2 = p["bbox"]
                    label = f"{p.get('class_name', 'Person')} ID:{p.get('tracker_id', '?')}"
                    cv2.rectangle(rgb, (px1, py1), (px2, py2), (255, 165, 0), 2)
                    cv2.putText(rgb, label, (px1, py1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 165, 0), 2)

            if threat_box is not None:
                l, t, r, b = threat_box
                with self.core.state:
                    name = getattr(self.core.state.threat_state, "name", "")
                
                color = (0, 255, 0) if threat_label.value == "SAFE" else (255, 0, 0)
                cv2.rectangle(rgb, (l, t), (r, b), color, 2)
                
                label_text = name if name else threat_label.value
                cv2.putText(rgb, label_text, (l, max(0, t - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            if show_mp:
                # Read 33 dynamic landmarks from state
                with self.core.state:
                    pose_lms = getattr(self.core.state, "pose_landmarks", [])
                
                # Draw MediaPipe Pose connections
                if pose_lms and len(pose_lms) >= 33:
                    pose_connections = [
                        # Torso
                        (11, 12), (11, 23), (12, 24), (23, 24),
                        # Left Arm
                        (11, 13), (13, 15),
                        # Right Arm
                        (12, 14), (14, 16),
                        # Left Leg
                        (23, 25), (25, 27),
                        # Right Leg
                        (24, 26), (26, 28),
                        # Shoulders to head connection
                        (0, 11), (0, 12)
                    ]
                    for start_idx, end_idx in pose_connections:
                        if start_idx < len(pose_lms) and end_idx < len(pose_lms):
                            pt1 = pose_lms[start_idx]
                            pt2 = pose_lms[end_idx]
                            cv2.line(rgb, pt1, pt2, (0, 255, 255), 2)
                    
                    # Only draw joint dots on core nodes (nose, shoulders, elbows, wrists, hips, knees, ankles)
                    core_joints = {0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28}
                    for idx, pt in enumerate(pose_lms):
                        if idx in core_joints:
                            cv2.circle(rgb, pt, 4, (0, 0, 255), -1)

            h, w = rgb.shape[:2]
            max_w = 640
            if w > max_w:
                scale = max_w / w
                h, w = int(h * scale), max_w
                rgb = cv2.resize(rgb, (w, h))
            img = ImageTk.PhotoImage(Image.fromarray(rgb))
            self.image_label.configure(image=img)
            self.image_label.image = img
