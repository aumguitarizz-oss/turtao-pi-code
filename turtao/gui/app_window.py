import tkinter as tk
from tkinter import ttk
import threading

from turtao.gui.tab_enroll import EnrollTab
from turtao.gui.tab_faces import FacesTab
from turtao.gui.tab_unknowns import UnknownsTab
from turtao.gui.tab_sensors import SensorsTab
from turtao.gui.tab_settings import SettingsTab
from turtao.gui.tab_log import LogTab


POLL_MS = 400


class AppWindow:
    def __init__(self, core) -> None:
        self.core = core

        self.root = tk.Tk()
        self.root.title("Turtao Pi — Development GUI")
        self.root.geometry("960x680")

        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        self.tab_enroll = EnrollTab(self.notebook, core)
        self.tab_faces = FacesTab(self.notebook, core)
        self.tab_unknowns = UnknownsTab(self.notebook, core)
        self.tab_sensors = SensorsTab(self.notebook, core)
        self.tab_settings = SettingsTab(self.notebook, core)
        self.tab_log = LogTab(self.notebook, core)

        self.notebook.add(self.tab_enroll.frame, text="Enroll")
        self.notebook.add(self.tab_faces.frame, text="Faces")
        self.notebook.add(self.tab_unknowns.frame, text="Unknowns")
        self.notebook.add(self.tab_sensors.frame, text="Sensors")
        self.notebook.add(self.tab_settings.frame, text="Settings")
        self.notebook.add(self.tab_log.frame, text="Log")

        self.root.after(POLL_MS, self._poll)

    def _poll(self) -> None:
        self.tab_enroll.refresh()
        self.tab_faces.refresh()
        self.tab_unknowns.refresh()
        self.tab_sensors.refresh()
        self.tab_settings.refresh()
        self.tab_log.refresh()
        self.root.after(POLL_MS, self._poll)

    def run(self) -> None:
        self.root.mainloop()
