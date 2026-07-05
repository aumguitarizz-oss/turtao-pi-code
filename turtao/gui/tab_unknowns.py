import tkinter as tk
from tkinter import ttk, simpledialog
from pathlib import Path
from PIL import Image, ImageTk
import cv2


class UnknownsTab:
    def __init__(self, parent, core) -> None:
        self.core = core
        self.frame = ttk.Frame(parent)
        self._canvas_frame = ttk.Frame(self.frame)
        self._canvas_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self._canvas = tk.Canvas(self._canvas_frame, highlightthickness=0)
        self._scrollbar = ttk.Scrollbar(
            self._canvas_frame, orient=tk.VERTICAL, command=self._canvas.yview
        )
        self._scrollable = ttk.Frame(self._canvas)

        self._scrollable.bind(
            "<Configure>",
            lambda e: self._canvas.configure(scrollregion=self._canvas.bbox("all")),
        )
        self._canvas.create_window((0, 0), window=self._scrollable, anchor="nw")
        self._canvas.configure(yscrollcommand=self._scrollbar.set)

        self._canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self._scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self._thumbnails: list[ImageTk.PhotoImage] = []

    def refresh(self) -> None:
        for w in self._scrollable.winfo_children():
            w.destroy()
        self._thumbnails.clear()

        unknowns_dir = Path("face_data/unknowns")
        if not unknowns_dir.is_dir():
            ttk.Label(self._scrollable, text="No unknown faces captured", foreground="gray").pack(
                pady=20
            )
            return

        jpg_files = sorted(unknowns_dir.glob("*.jpg"))
        if not jpg_files:
            ttk.Label(self._scrollable, text="No unknown faces captured", foreground="gray").pack(
                pady=20
            )
            return

        for fpath in jpg_files:
            card = ttk.Frame(self._scrollable, relief=tk.RIDGE, borderwidth=1)
            card.pack(fill=tk.X, pady=4, padx=5)

            inner = ttk.Frame(card)
            inner.pack(fill=tk.X, padx=5, pady=5)

            img = None
            try:
                cv_img = cv2.imread(str(fpath))
                if cv_img is not None:
                    rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
                    h, w = rgb.shape[:2]
                    max_h = 80
                    if h > max_h:
                        scale = max_h / h
                        h, w = int(h * scale), int(w * scale)
                        rgb = cv2.resize(rgb, (w, h))
                    pil_img = Image.fromarray(rgb)
                    img = ImageTk.PhotoImage(pil_img)
            except Exception:
                pass

            if img is not None:
                thumb_label = ttk.Label(inner, image=img)
                thumb_label.pack(side=tk.LEFT, padx=(0, 10))
                self._thumbnails.append(img)
            else:
                ttk.Label(inner, text="(no image)", foreground="gray").pack(
                    side=tk.LEFT, padx=(0, 10)
                )

            ttk.Label(inner, text=fpath.stem.replace("unknown_", "")).pack(
                side=tk.LEFT, padx=10
            )

            promote_btn = ttk.Button(
                inner,
                text="Promote",
                command=lambda p=fpath: self._promote(p),
            )
            promote_btn.pack(side=tk.RIGHT, padx=2)

            delete_btn = ttk.Button(
                inner,
                text="Delete",
                command=lambda p=fpath: self._delete(p),
            )
            delete_btn.pack(side=tk.RIGHT, padx=2)

    def _promote(self, fpath: Path) -> None:
        name = simpledialog.askstring("Promote", "Enter name for this face:", parent=self.frame)
        if not name:
            return
        name = name.strip()
        if not name:
            return
        try:
            img = cv2.imread(str(fpath))
            if img is None:
                return
            result = self.core.enrollment.start_enrollment(name)
            if result.get("status") == "error":
                return
            result = self.core.enrollment.capture_pose(img)
        except Exception:
            pass

    def _delete(self, fpath: Path) -> None:
        try:
            fpath.unlink(missing_ok=True)
        except OSError:
            pass
