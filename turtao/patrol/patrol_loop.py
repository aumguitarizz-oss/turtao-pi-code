from __future__ import annotations

import json
import logging
import time

from turtao.state import AppState, Mode, ThreatLabel

logger = logging.getLogger(__name__)

_SPEED = 0.8
_SAFE_MODE = False


def set_speed(speed: float) -> None:
    global _SPEED
    _SPEED = speed


def set_safe_mode(safe: bool) -> None:
    global _SAFE_MODE
    _SAFE_MODE = safe


def patrol_loop(state: AppState, serial_link: object) -> None:
    """Commands steady forward motion while patrolling; obstacle
    avoidance is no longer done here. With only a single front-facing
    ToF left on the stripped-down build there's nothing to steer with
    (the old left/right/down 4-sensor differential steering and cliff
    detection are impossible), and the ESP32 firmware's own bumperTick()
    already reacts to the wall in real time (reverse -> turn right ->
    resume) regardless of what's commanded here -- so the Pi just holds
    a constant forward command and lets the microcontroller handle it."""
    logger.info("Patrol loop started")
    while not state.stop_event.is_set():
        with state:
            mode = state.mode
            threat = state.threat_label
            safe_mode = _SAFE_MODE
            speed = _SPEED

        if mode == Mode.PATROL:
            if safe_mode:
                time.sleep(0.2)
                continue
            if threat == ThreatLabel.THREAT:
                serial_link.write(json.dumps({"cmd": "move", "ml": 0.0, "mr": 0.0}))
                time.sleep(0.2)
                continue
            serial_link.write(json.dumps({"cmd": "move", "ml": speed, "mr": speed}))
        time.sleep(0.2)
