from unittest.mock import MagicMock

from turtao.patrol.patrol_loop import patrol_loop, set_safe_mode, set_speed
from turtao.state import AppState, Mode, ThreatLabel


class TestPatrolHelpers:
    def test_set_speed_changes_global(self):
        set_speed(0.5)
        from turtao.patrol.patrol_loop import _SPEED
        assert _SPEED == 0.5
        set_speed(0.8)

    def test_set_safe_mode_changes_global(self):
        set_safe_mode(True)
        from turtao.patrol.patrol_loop import _SAFE_MODE
        assert _SAFE_MODE is True
        set_safe_mode(False)


class TestPatrolLoop:
    """Obstacle avoidance moved onto the ESP32's own bumperTick() now that
    only a single front ToF is left (no more differential left/right
    steering or cliff detection). patrol_loop() just holds a steady
    forward command while patrolling."""

    def _run_one_iteration(self, state: AppState, serial_link) -> None:
        # patrol_loop() has no natural exit; stop it after the first
        # write by setting stop_event from inside the mocked write().
        serial_link.write.side_effect = lambda *a, **k: state.stop_event.set()
        patrol_loop(state, serial_link)

    def test_commands_steady_forward_while_patrolling(self):
        set_speed(0.6)
        set_safe_mode(False)
        state = AppState()
        state.mode = Mode.PATROL
        state.threat_label = ThreatLabel.IDLE
        serial_link = MagicMock()

        self._run_one_iteration(state, serial_link)

        sent = serial_link.write.call_args[0][0]
        assert '"cmd": "move"' in sent
        assert '"ml": 0.6' in sent
        assert '"mr": 0.6' in sent
        set_speed(0.8)

    def test_commands_zero_speed_when_threat_active(self):
        set_speed(0.8)
        set_safe_mode(False)
        state = AppState()
        state.mode = Mode.PATROL
        state.threat_label = ThreatLabel.THREAT
        serial_link = MagicMock()

        self._run_one_iteration(state, serial_link)

        sent = serial_link.write.call_args[0][0]
        assert '"ml": 0.0' in sent
        assert '"mr": 0.0' in sent

    def test_sends_nothing_while_safe_mode_is_on(self):
        set_speed(0.8)
        set_safe_mode(True)
        state = AppState()
        state.mode = Mode.PATROL

        # Safe mode never calls write(), so drive the loop off the
        # sleep instead of the write side effect used elsewhere.
        import turtao.patrol.patrol_loop as patrol_module
        original_sleep = patrol_module.time.sleep
        calls = {"n": 0}

        def fake_sleep(seconds):
            calls["n"] += 1
            if calls["n"] >= 1:
                state.stop_event.set()

        patrol_module.time.sleep = fake_sleep
        try:
            serial_link = MagicMock()
            patrol_loop(state, serial_link)
        finally:
            patrol_module.time.sleep = original_sleep

        serial_link.write.assert_not_called()
        set_safe_mode(False)
