import threading
import time
from turtao.state import AppState, Mode, ThreatLabel, SensorData


class TestStateLockContention:
    def test_multiple_readers_simultaneously(self, app_state: AppState):
        results = []
        errors = []

        def reader():
            for _ in range(50):
                try:
                    with app_state:
                        _ = app_state.mode
                        _ = app_state.heading
                        _ = app_state.sensor_data.temp_c
                    time.sleep(0.001)
                except Exception as e:
                    errors.append(e)

        threads = [threading.Thread(target=reader) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)

        assert not errors, f"Reader errors: {errors}"

    def test_writers_dont_conflict(self, app_state: AppState):
        errors = []

        def writer():
            for i in range(50):
                try:
                    with app_state:
                        app_state.heading = i % 360
                        app_state.mode = Mode.PATROL if i % 2 == 0 else Mode.IDLE
                    time.sleep(0.001)
                except Exception as e:
                    errors.append(e)

        threads = [threading.Thread(target=writer) for _ in (0, 1, 2)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)

        assert not errors, f"Writer errors: {errors}"
        assert app_state.heading in range(360)
        assert app_state.mode in (Mode.PATROL, Mode.IDLE)

    def test_read_and_write_race_does_not_corrupt(self, app_state: AppState):
        errors = []

        def writer():
            for i in range(100):
                try:
                    with app_state:
                        app_state.sensor_data.temp_c = float(i)
                        app_state.sensor_data.humidity_pct = i
                        app_state.sensor_data.pressure_hpa = float(i * 10)
                except Exception as e:
                    errors.append(e)

        def reader():
            for _ in range(100):
                try:
                    with app_state:
                        t = app_state.sensor_data.temp_c
                        h = app_state.sensor_data.humidity_pct
                        p = app_state.sensor_data.pressure_hpa
                    assert isinstance(t, float)
                    assert isinstance(h, int)
                    assert isinstance(p, float)
                except Exception as e:
                    errors.append(e)

        threads = [threading.Thread(target=writer) for _ in range(3)]
        threads += [threading.Thread(target=reader) for _ in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)

        assert not errors, f"Race errors: {errors}"

    def test_mode_set_and_read_consistency(self, app_state: AppState):
        results = []
        errors = []

        def set_mode():
            for m in (Mode.IDLE, Mode.GUARD, Mode.PATROL):
                try:
                    with app_state:
                        app_state.mode = m
                    time.sleep(0.005)
                except Exception as e:
                    errors.append(e)

        def get_mode():
            for _ in range(30):
                try:
                    with app_state:
                        results.append(app_state.mode)
                    time.sleep(0.003)
                except Exception as e:
                    errors.append(e)

        t1 = threading.Thread(target=set_mode)
        t2 = threading.Thread(target=get_mode)
        t1.start()
        t2.start()
        t1.join(timeout=5)
        t2.join(timeout=5)

        assert not errors, f"Consistency errors: {errors}"

    def test_stop_event_thread_safe(self, app_state: AppState):
        errors = []

        def waiter():
            try:
                app_state.stop_event.wait(1.0)
            except Exception as e:
                errors.append(e)

        def setter():
            time.sleep(0.1)
            try:
                app_state.stop_event.set()
            except Exception as e:
                errors.append(e)

        t1 = threading.Thread(target=waiter)
        t2 = threading.Thread(target=setter)
        t1.start()
        t2.start()
        t1.join(timeout=5)
        t2.join(timeout=5)

        assert not errors, f"Stop event errors: {errors}"
        assert app_state.stop_event.is_set()

    def test_stress_mixed_access(self, app_state: AppState):
        errors = []

        def mixed_work():
            for _ in range(100):
                try:
                    with app_state:
                        app_state.frame_counter += 1
                        app_state.heading = (app_state.heading + 1) % 360
                        if app_state.frame_counter % 10 == 0:
                            app_state.sensor_data.tof_cm = [
                                app_state.frame_counter % 100,
                                (app_state.frame_counter + 10) % 100,
                                (app_state.frame_counter + 20) % 100,
                                (app_state.frame_counter + 30) % 100,
                            ]
                        _ = len(app_state.sensor_data.tof_cm)
                except Exception as e:
                    errors.append(e)
                time.sleep(0.001)

        threads = [threading.Thread(target=mixed_work) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert not errors, f"Stress errors: {errors}"
