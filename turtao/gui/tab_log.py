import logging
import tkinter as tk
from collections import deque
from tkinter import ttk


class LogTab:
    def __init__(self, parent, core) -> None:
        self.core = core
        self.frame = ttk.Frame(parent)
        self._buffer: deque[str] = deque(maxlen=500)
        self._setup_ui()

    def _setup_ui(self) -> None:
        ttk.Label(self.frame, text="Recent Logs", font=("", 14, "bold")).pack(
            pady=(10, 5)
        )

        self._text = tk.Text(self.frame, state=tk.DISABLED, wrap=tk.WORD)
        self._text.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        self._scrollbar = ttk.Scrollbar(self._text, command=self._text.yview)
        self._text.configure(yscrollcommand=self._scrollbar.set)
        self._scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        btn_frame = ttk.Frame(self.frame)
        btn_frame.pack(fill=tk.X, padx=10, pady=(0, 10))

        ttk.Button(btn_frame, text="Clear", command=self._clear).pack(side=tk.LEFT)

    def _clear(self) -> None:
        self._buffer.clear()
        self._text.configure(state=tk.NORMAL)
        self._text.delete(1.0, tk.END)
        self._text.configure(state=tk.DISABLED)

    def refresh(self) -> None:
        new_lines: list[str] = []
        handler = _CaptureHandler.instance
        if handler is not None:
            new_lines = handler.drain()

        if not new_lines:
            return

        self._text.configure(state=tk.NORMAL)
        for line in new_lines:
            self._text.insert(tk.END, line)
        self._text.see(tk.END)
        self._text.configure(state=tk.DISABLED)


class _CaptureHandler(logging.Handler):
    instance: "_CaptureHandler | None" = None

    def __init__(self, maxlen: int = 500) -> None:
        super().__init__()
        self._buffer: deque[str] = deque(maxlen=maxlen)
        self.setFormatter(
            logging.Formatter("%(asctime)s [%(threadName)s] %(levelname)s %(name)s: %(message)s")
        )
        _CaptureHandler.instance = self

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record) + "\n"
            self._buffer.append(msg)
        except Exception:
            self.handleError(record)

    def drain(self) -> list[str]:
        items = list(self._buffer)
        self._buffer.clear()
        return items


_capture_handler = _CaptureHandler()
logging.getLogger().addHandler(_capture_handler)
