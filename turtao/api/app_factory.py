import logging

from flask import Flask
from flask_cors import CORS
from flask_sock import Sock

logger = logging.getLogger(__name__)


def create_app(state, settings, config, serial_link, camera, face_engine, enrollment, tts, bt_manager, tracker, antispoof) -> Flask:
    app = Flask(__name__)
    CORS(app, origins="*")
    sock = Sock(app)

    app.config["STATE"] = state
    app.config["SETTINGS"] = settings
    app.config["CONFIG"] = config
    app.config["SERIAL"] = serial_link
    app.config["CAMERA"] = camera
    app.config["FACE_ENGINE"] = face_engine
    app.config["ENROLLMENT"] = enrollment
    app.config["TTS"] = tts
    app.config["BT_MANAGER"] = bt_manager
    app.config["TRACKER"] = tracker
    app.config["ANTISPOOF"] = antispoof

    from turtao.api.errors import register_error_handlers
    register_error_handlers(app)

    from turtao.api import (
        routes_alert,
        routes_audio,
        routes_ble,
        routes_camera,
        routes_control,
        routes_environment,
        routes_faces,
        routes_misc,
        routes_mode,
        routes_settings,
    )

    audio_bp = routes_audio.audio_bp
    camera_bp = routes_camera.camera_bp
    alert_bp = routes_alert.alert_bp
    environment_bp = routes_environment.environment_bp
    mode_bp = routes_mode.mode_bp
    control_bp = routes_control.control_bp
    faces_bp = routes_faces.faces_bp
    settings_bp = routes_settings.settings_bp
    ble_bp = routes_ble.ble_bp
    misc_bp = routes_misc.misc_bp

    routes_camera.inject_deps(camera=camera)
    routes_alert.inject_deps(state=state)
    routes_environment.inject_deps(state=state)
    routes_mode.inject_deps(state=state, serial=serial_link)
    routes_control.inject_deps(state=state, serial=serial_link, settings=settings)
    routes_faces.inject_deps(face_engine=face_engine, enrollment=enrollment, config=config, state=state)
    routes_settings.inject_deps(settings=settings, tts=tts)
    routes_ble.inject_deps(state=state, settings=settings, serial=serial_link)
    routes_misc.inject_deps(state=state, serial=serial_link, settings=settings)
    routes_audio.inject_deps(tts=tts)

    app.register_blueprint(audio_bp)
    app.register_blueprint(camera_bp)
    app.register_blueprint(alert_bp)
    app.register_blueprint(environment_bp)
    app.register_blueprint(mode_bp)
    app.register_blueprint(control_bp)
    app.register_blueprint(faces_bp)
    app.register_blueprint(settings_bp)
    app.register_blueprint(ble_bp)
    app.register_blueprint(misc_bp)

    from turtao.api.ws_intercom import register_intercom_ws
    from turtao.api.ws_status import register_status_ws
    register_status_ws(sock, state)
    register_intercom_ws(sock)

    return app
