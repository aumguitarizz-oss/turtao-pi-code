#!/usr/bin/env python3
"""Turtao Pi Backend - Entry Point"""

import argparse
import logging
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Turtao Surveillance Robot")
    parser.add_argument("--gui", action="store_true", help="Open development GUI window")
    parser.add_argument("--mock", action="store_true", help="Use mock hardware (no ESP32/camera required)")
    args = parser.parse_args()

    from turtao.logging_config import configure_logging
    configure_logging(debug=args.gui)

    logger = logging.getLogger(__name__)

    from turtao.config import load_dotenv_config, load_settings
    config = load_dotenv_config()
    settings = load_settings()

    from turtao.state import AppState
    state = AppState()

    if args.mock:
        from turtao.hardware.mocks import MockSerialLink, MockCamera
        serial_link = MockSerialLink()
        camera = MockCamera()
        logger.info("Started with MOCK hardware (no ESP32, no camera)")
    else:
        serial_link = None
        camera = None

    from turtao.core import TurtaoCore
    core = TurtaoCore(config=config, settings=settings, state=state, serial_link=serial_link, camera=camera)

    if args.gui:
        import threading
        from turtao.api.app_factory import create_app
        app = create_app(
            state, settings, config, core.serial, core.camera,
            core.face_engine, core.enrollment, core.tts,
            core.bt_manager, core.tracker, core.antispoof,
        )
        flask_thread = threading.Thread(
            target=lambda: app.run(
                host="0.0.0.0", port=config.flask_port, debug=False, use_reloader=False
            ),
            daemon=True,
            name="flask",
        )
        flask_thread.start()

        core.start()
        from turtao.gui.app_window import AppWindow
        window = AppWindow(core)
        window.run()
    else:
        core.start()
        from turtao.api.app_factory import create_app
        app = create_app(
            state, settings, config, core.serial, core.camera,
            core.face_engine, core.enrollment, core.tts,
            core.bt_manager, core.tracker, core.antispoof,
        )
        app.run(
            host="0.0.0.0", port=config.flask_port, debug=False, use_reloader=False,
        )


if __name__ == "__main__":
    main()
