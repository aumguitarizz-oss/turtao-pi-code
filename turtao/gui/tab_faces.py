import json
import tkinter as tk
from pathlib import Path
from tkinter import ttk

import numpy as np
from PIL import Image, ImageTk


class FacesTab:
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
        self._last_profiles_mtime: float = -1.0

    def refresh(self) -> None:
        profiles_path = Path("face_data/profiles.json")
        embeddings_dir = Path("face_data/embeddings")

        mtime = 0.0
        if profiles_path.exists():
            try:
                mtime = profiles_path.stat().st_mtime
            except OSError:
                pass

        if mtime == self._last_profiles_mtime:
            return

        self._last_profiles_mtime = mtime

        for w in self._scrollable.winfo_children():
            w.destroy()
        self._thumbnails.clear()

        if mtime == 0.0:
            ttk.Label(self._scrollable, text="No enrolled faces", foreground="gray").pack(
                pady=20
            )
            return

        try:
            profiles = json.loads(profiles_path.read_text())
        except (json.JSONDecodeError, OSError):
            ttk.Label(self._scrollable, text="Error reading profiles", foreground="red").pack(
                pady=20
            )
            return

        if not profiles:
            ttk.Label(self._scrollable, text="No enrolled faces", foreground="gray").pack(
                pady=20
            )
            return

        for profile in profiles:
            card = ttk.LabelFrame(self._scrollable, text=profile.get("name", "?"))
            card.pack(fill=tk.X, pady=4, padx=5)

            inner = ttk.Frame(card)
            inner.pack(fill=tk.X, padx=5, pady=5)

            thumb_img = self._load_thumb(profile, embeddings_dir)
            if thumb_img is not None:
                thumb_label = ttk.Label(inner, image=thumb_img)
                thumb_label.pack(side=tk.LEFT, padx=(0, 10))
                self._thumbnails.append(thumb_img)
            else:
                ttk.Label(inner, text="(no thumbnail)", foreground="gray").pack(
                    side=tk.LEFT, padx=(0, 10)
                )

            info = profile.get("embeddings", [])
            ttk.Label(
                inner, text=f"{len(info)} pose(s)"
            ).pack(side=tk.LEFT, padx=10)

            name = profile.get("name", "")
            delete_btn = ttk.Button(
                inner,
                text="Delete",
                command=lambda n=name: self._delete_face(n),
            )
            delete_btn.pack(side=tk.RIGHT)

    def _load_thumb(self, profile: dict, embeddings_dir: Path) -> ImageTk.PhotoImage | None:
        emb_files = profile.get("embeddings", [])
        if not emb_files:
            return None
        npy_path = embeddings_dir / emb_files[0]
        if not npy_path.exists():
            return None
        arr = np.load(str(npy_path))
        size = int(np.sqrt(len(arr)))
        if size * size == len(arr):
            arr_2d = arr[: size * size].reshape(size, size)
            arr_2d = ((arr_2d - arr_2d.min()) / (arr_2d.max() - arr_2d.min() + 1e-10) * 255).astype(
                np.uint8
            )
            img = Image.fromarray(arr_2d, mode="L").resize((64, 64))
        else:
            img = Image.new("RGB", (64, 64), color=100)
        return ImageTk.PhotoImage(img)

    def _delete_face(self, name: str) -> None:
        try:
            profiles_path = Path("face_data/profiles.json")
            if not profiles_path.exists():
                return
            profiles = json.loads(profiles_path.read_text())
            profiles = [p for p in profiles if p.get("name") != name]
            profiles_path.write_text(json.dumps(profiles, indent=2))
        except (json.JSONDecodeError, OSError):
            pass
