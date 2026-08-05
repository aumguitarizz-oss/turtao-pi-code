import tkinter as tk
from tkinter import ttk

from turtao.gui.tab_enroll import EnrollTab
from turtao.gui.tab_faces import FacesTab
from turtao.gui.tab_log import LogTab
from turtao.gui.tab_recognition import RecognitionTab
from turtao.gui.tab_sensors import SensorsTab
from turtao.gui.tab_settings import SettingsTab
from turtao.gui.tab_unknowns import UnknownsTab

# Fast poll: lightweight status / text updates (33ms ≈ 30fps)
FAST_POLL_MS = 33
# Slow poll: expensive redraws (scrollable lists, file-system scans)
SLOW_POLL_MS = 500
# Even slower: settings / log
VERY_SLOW_POLL_MS = 1000

_slow_tick = 0
_very_slow_tick = 0


class AppWindow:
    def __init__(self, core) -> None:
        self.core = core

        self.root = tk.Tk()
        self.root.title("Turtao Pi — Development GUI")
        self.root.geometry("960x680")
        try:
            import sv_ttk
            sv_ttk.set_theme("dark")
        except ImportError:
            pass

        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        self.tab_enroll = EnrollTab(self.notebook, core)
        self.tab_faces = FacesTab(self.notebook, core)
        self.tab_unknowns = UnknownsTab(self.notebook, core)
        self.tab_sensors = SensorsTab(self.notebook, core)
        self.tab_settings = SettingsTab(self.notebook, core)
        self.tab_log = LogTab(self.notebook, core)
        self.tab_recog = RecognitionTab(self.notebook, core)

        self.notebook.add(self.tab_recog.frame, text="Recognition")
        self.notebook.add(self.tab_enroll.frame, text="Enroll")
        self.notebook.add(self.tab_faces.frame, text="Faces")
        self.notebook.add(self.tab_unknowns.frame, text="Unknowns")
        self.notebook.add(self.tab_sensors.frame, text="Sensors")
        self.notebook.add(self.tab_settings.frame, text="Settings")
        self.notebook.add(self.tab_log.frame, text="Log")

        self.root.after(FAST_POLL_MS, self._fast_poll)
        self.root.after(SLOW_POLL_MS, self._slow_poll)
        self.root.after(VERY_SLOW_POLL_MS, self._very_slow_poll)

    def _fast_poll(self) -> None:
        """High-frequency: live video frames + enrollment status text only."""
        try:
            # Only refresh the *visible* tab's video to reduce CPU
            current_tab = self.notebook.index(self.notebook.select())
            if current_tab == 0:  # Recognition
                self.tab_recog.refresh()
            elif current_tab == 1:  # Enroll
                self.tab_enroll.refresh()
            elif current_tab == 4:  # Sensors
                self.tab_sensors.refresh()
        except Exception:
            pass
        self.root.after(FAST_POLL_MS, self._fast_poll)

    def _slow_poll(self) -> None:
        """
        Medium-frequency: refresh list-heavy tabs only when they are visible
        and only call the ones that are currently shown.
        FacesTab and UnknownsTab already internally debounce via mtime checks —
        but we still avoid calling them at 30fps which was causing flicker.
        """
        try:
            current_tab = self.notebook.index(self.notebook.select())
            if current_tab == 2:  # Faces
                self.tab_faces.refresh()
            elif current_tab == 3:  # Unknowns
                self.tab_unknowns.refresh()
        except Exception:
            pass
        self.root.after(SLOW_POLL_MS, self._slow_poll)

    def _very_slow_poll(self) -> None:
        """Low-frequency: settings + log (rarely change)."""
        try:
            current_tab = self.notebook.index(self.notebook.select())
            if current_tab == 5:  # Settings
                self.tab_settings.refresh()
            elif current_tab == 6:  # Log
                self.tab_log.refresh()
        except Exception:
            pass
        self.root.after(VERY_SLOW_POLL_MS, self._very_slow_poll)

    def run(self) -> None:
        self.root.mainloop()
