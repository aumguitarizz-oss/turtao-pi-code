from __future__ import annotations

import json
import logging
import time

from turtao.state import AppState, Mode, ThreatLabel

logger = logging.getLogger(__name__)

OBSTACLE_MM = 200
DROP_MM = 400

_SPEED = 0.8
_SAFE_MODE = False


def set_speed(speed: float) -> None:
    global _SPEED
    _SPEED = speed


def set_safe_mode(safe: bool) -> None:
    global _SAFE_MODE
    _SAFE_MODE = safe


def patrol_step(tof_fl: int, tof_fc: int, tof_fr: int, tof_down: int, speed: float) -> dict:
    if tof_down > DROP_MM:
        return {"cmd": "move", "ml": -0.5, "mr": -0.5}
    if tof_fc < OBSTACLE_MM:
        if tof_fl > tof_fr:
            return {"cmd": "move", "ml": -0.5, "mr": 0.5}
        return {"cmd": "move", "ml": 0.5, "mr": -0.5}
    if tof_fl < OBSTACLE_MM:
        return {"cmd": "move", "ml": 0.4, "mr": -0.4}
    if tof_fr < OBSTACLE_MM:
        return {"cmd": "move", "ml": -0.4, "mr": 0.4}
    return {"cmd": "move", "ml": speed, "mr": speed}


def patrol_loop(state: AppState, serial_link: object) -> None:
    logger.info("Patrol loop started")
    while not state.stop_event.is_set():
        with state:
            mode = state.mode
            threat = state.threat_label
            tof_cm = list(state.sensor_data.tof_cm)
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
            if len(tof_cm) >= 4:
                cmd = patrol_step(
                    tof_cm[0] * 10,
                    tof_cm[1] * 10,
                    tof_cm[2] * 10,
                    tof_cm[3] * 10,
                    speed,
                )
                serial_link.write(json.dumps(cmd))
        time.sleep(0.2)
