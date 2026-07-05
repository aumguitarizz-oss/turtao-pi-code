import pytest
from turtao.patrol.patrol_loop import patrol_step, set_speed, set_safe_mode


class TestPatrolStep:
    def test_drop_detected_reverses(self):
        result = patrol_step(
            tof_fl=100, tof_fc=150, tof_fr=180, tof_down=450, speed=0.8
        )
        assert result == {"cmd": "move", "ml": -0.5, "mr": -0.5}

    def test_drop_at_boundary_does_not_trigger(self):
        result = patrol_step(
            tof_fl=100, tof_fc=100, tof_fr=100, tof_down=400, speed=0.8
        )
        assert result != {"cmd": "move", "ml": -0.5, "mr": -0.5}

    def test_drop_above_threshold_triggers(self):
        result = patrol_step(
            tof_fl=100, tof_fc=100, tof_fr=100, tof_down=401, speed=0.8
        )
        assert result == {"cmd": "move", "ml": -0.5, "mr": -0.5}

    def test_center_obstacle_turns_toward_clear_left(self):
        result = patrol_step(
            tof_fl=300, tof_fc=50, tof_fr=100, tof_down=100, speed=0.8
        )
        assert result == {"cmd": "move", "ml": -0.5, "mr": 0.5}

    def test_center_obstacle_turns_toward_clear_right(self):
        result = patrol_step(
            tof_fl=100, tof_fc=50, tof_fr=300, tof_down=100, speed=0.8
        )
        assert result == {"cmd": "move", "ml": 0.5, "mr": -0.5}

    def test_center_obstacle_equal_sides_turns_right(self):
        result = patrol_step(
            tof_fl=150, tof_fc=50, tof_fr=150, tof_down=100, speed=0.8
        )
        assert result == {"cmd": "move", "ml": 0.5, "mr": -0.5}

    def test_left_obstacle_slight_right(self):
        result = patrol_step(
            tof_fl=50, tof_fc=300, tof_fr=300, tof_down=100, speed=0.8
        )
        assert result == {"cmd": "move", "ml": 0.4, "mr": -0.4}

    def test_left_obstacle_at_boundary_does_not_trigger(self):
        result = patrol_step(
            tof_fl=200, tof_fc=300, tof_fr=300, tof_down=100, speed=0.8
        )
        assert result == {"cmd": "move", "ml": 0.8, "mr": 0.8}

    def test_right_obstacle_slight_left(self):
        result = patrol_step(
            tof_fl=300, tof_fc=300, tof_fr=50, tof_down=100, speed=0.8
        )
        assert result == {"cmd": "move", "ml": -0.4, "mr": 0.4}

    def test_right_obstacle_at_boundary_does_not_trigger(self):
        result = patrol_step(
            tof_fl=300, tof_fc=300, tof_fr=200, tof_down=100, speed=0.8
        )
        assert result == {"cmd": "move", "ml": 0.8, "mr": 0.8}

    def test_clear_forward_at_speed(self):
        result = patrol_step(
            tof_fl=300, tof_fc=300, tof_fr=300, tof_down=100, speed=0.8
        )
        assert result == {"cmd": "move", "ml": 0.8, "mr": 0.8}

    def test_forward_uses_provided_speed(self):
        result = patrol_step(
            tof_fl=300, tof_fc=300, tof_fr=300, tof_down=100, speed=0.5
        )
        assert result == {"cmd": "move", "ml": 0.5, "mr": 0.5}

    def test_forward_at_zero_speed(self):
        result = patrol_step(
            tof_fl=300, tof_fc=300, tof_fr=300, tof_down=100, speed=0.0
        )
        assert result == {"cmd": "move", "ml": 0.0, "mr": 0.0}


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
