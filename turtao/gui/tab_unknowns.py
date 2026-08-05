import tkinter as tk
from pathlib import Path
from tkinter import simpledialog, ttk

import cv2
from PIL import Image, ImageTk


class UnknownsTab:
    def __init__(self, parent, core) -> None:
        self.core = core
        self.frame = ttk.Frame(parent)

        top_bar = ttk.Frame(self.frame)
        top_bar.pack(fill=tk.X, padx=10, pady=(8, 0))
        ttk.Label(top_bar, text="Unknown Faces", font=("", 14, "bold")).pack(side=tk.LEFT)
        self._count_var = tk.StringVar(value="")
        ttk.Label(top_bar, textvariable=self._count_var, foreground="gray").pack(
            side=tk.LEFT, padx=8
        )

        self._canvas_frame = ttk.Frame(self.frame)
        self._canvas_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=8)

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
        # Track the set of (path, mtime) tuples — only rebuild when this changes
        self._last_state: frozenset = frozenset()

    def refresh(self) -> None:
        unknowns_dir = Path("face_data/unknowns")
        jpg_files: list[Path] = []
        if unknowns_dir.is_dir():
            try:
                jpg_files = sorted(unknowns_dir.glob("*.jpg"))
            except OSError:
                pass

        # Build a stable fingerprint of what's on disk
        current_state: dict[str, float] = {}
        for f in jpg_files:
            try:
                current_state[str(f)] = f.stat().st_mtime
            except OSError:
                pass

        fingerprint = frozenset(current_state.items())
        if fingerprint == self._last_state:
            return  # nothing changed — skip all widget work

        self._last_state = fingerprint
        self._rebuild(jpg_files)

    def _rebuild(self, jpg_files: list[Path]) -> None:
        """Tear down and rebuild the scrollable list only when content changes."""
        for w in self._scrollable.winfo_children():
            w.destroy()
        self._thumbnails.clear()

        count = len(jpg_files)
        self._count_var.set(f"({count} captured)" if count else "")

        if not jpg_files:
            ttk.Label(
                self._scrollable,
                text="No unknown faces captured yet",
                foreground="gray",
            ).pack(pady=20)
            return

        for fpath in jpg_files:
            card = ttk.Frame(self._scrollable, relief=tk.RIDGE, borderwidth=1)
            card.pack(fill=tk.X, pady=4, padx=5)

            inner = ttk.Frame(card)
            inner.pack(fill=tk.X, padx=5, pady=5)

            img = self._load_thumb(fpath)
            if img is not None:
                thumb_label = ttk.Label(inner, image=img)
                thumb_label.pack(side=tk.LEFT, padx=(0, 10))
                self._thumbnails.append(img)
            else:
                ttk.Label(inner, text="(no image)", foreground="gray").pack(
                    side=tk.LEFT, padx=(0, 10)
                )

            # Show timestamp from filename
            stem = fpath.stem.replace("unknown_", "")
            display = stem[:4] + "-" + stem[4:6] + "-" + stem[6:8] + "  " + stem[9:11] + ":" + stem[11:13] + ":" + stem[13:15] if len(stem) >= 15 else stem
            ttk.Label(inner, text=display).pack(side=tk.LEFT, padx=10)

            promote_btn = ttk.Button(
                inner,
                text="Promote →",
                command=lambda p=fpath: self._promote(p),
            )
            promote_btn.pack(side=tk.RIGHT, padx=2)

            delete_btn = ttk.Button(
                inner,
                text="Delete",
                command=lambda p=fpath: self._delete(p),
            )
            delete_btn.pack(side=tk.RIGHT, padx=2)

    @staticmethod
    def _load_thumb(fpath: Path) -> ImageTk.PhotoImage | None:
        try:
            cv_img = cv2.imread(str(fpath))
            if cv_img is None:
                return None
            rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
            h, w = rgb.shape[:2]
            max_h = 80
            if h > max_h:
                scale = max_h / h
                rgb = cv2.resize(rgb, (int(w * scale), max_h), interpolation=cv2.INTER_LINEAR)
            return ImageTk.PhotoImage(Image.fromarray(rgb))
        except Exception:
            return None

    def _promote(self, fpath: Path) -> None:
        name = simpledialog.askstring(
            "Promote Unknown", "Enter name for this face:", parent=self.frame
        )
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
            self.core.enrollment.capture_pose(lambda: img)
            # Invalidate so the list rebuilds next poll
            self._last_state = frozenset()
        except Exception:
            pass

    def _delete(self, fpath: Path) -> None:
        try:
            fpath.unlink(missing_ok=True)
            # Invalidate so the list rebuilds next poll
            self._last_state = frozenset()
        except OSError:
            pass
