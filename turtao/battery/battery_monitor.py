from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

VOLTAGE_TABLE: list[tuple[float, int]] = [
    (12.6, 100), (12.2, 90), (11.8, 75),
    (11.5, 60),  (11.1, 50), (10.8, 35),
    (10.5, 20),  (10.2, 10), (9.9, 0),
]


def voltage_to_percent(voltage: float) -> float:
    if voltage >= VOLTAGE_TABLE[0][0]:
        return 100.0
    if voltage <= VOLTAGE_TABLE[-1][0]:
        return 0.0
    for i in range(len(VOLTAGE_TABLE) - 1):
        v_high, p_high = VOLTAGE_TABLE[i]
        v_low, p_low = VOLTAGE_TABLE[i + 1]
        if v_low <= voltage <= v_high:
            fraction = (voltage - v_low) / (v_high - v_low)
            return round(p_low + fraction * (p_high - p_low), 1)
    return 0.0


def get_battery_state(
    voltage: float, percent: float, is_charging: bool
) -> dict[str, Any]:
    return {
        "voltage": round(voltage, 2),
        "current_ma": 0,
        "percent": round(percent, 1),
        "status": "charging" if is_charging else "discharging",
    }
