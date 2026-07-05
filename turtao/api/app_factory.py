from flask import Flask
from flask_cors import CORS
from flask_sock import Sock
import logging

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

    from turtao.api.routes_camera import camera_bp
    from turtao.api.routes_alert import alert_bp
    from turtao.api.routes_environment import environment_bp
    from turtao.api.routes_mode import mode_bp
    from turtao.api.routes_control import control_bp
    from turtao.api.routes_faces import faces_bp
    from turtao.api.routes_settings import settings_bp
    from turtao.api.routes_ble import ble_bp
    from turtao.api.routes_misc import misc_bp

    camera_bp.inject_deps(camera=camera)
    alert_bp.inject_deps(state=state)
    environment_bp.inject_deps(state=state)
    mode_bp.inject_deps(state=state, serial=serial_link)
    control_bp.inject_deps(state=state, serial=serial_link)
    faces_bp.inject_deps(face_engine=face_engine, enrollment=enrollment, config=config)
    settings_bp.inject_deps(settings=settings, tts=tts)
    ble_bp.inject_deps(bt_manager=bt_manager, settings=settings, serial=serial_link)
    misc_bp.inject_deps(state=state, serial=serial_link, settings=settings)

    app.register_blueprint(camera_bp)
    app.register_blueprint(alert_bp)
    app.register_blueprint(environment_bp)
    app.register_blueprint(mode_bp)
    app.register_blueprint(control_bp)
    app.register_blueprint(faces_bp)
    app.register_blueprint(settings_bp)
    app.register_blueprint(ble_bp)
    app.register_blueprint(misc_bp)

    from turtao.api.ws_status import register_status_ws
    from turtao.api.ws_intercom import register_intercom_ws
    register_status_ws(sock, state)
    register_intercom_ws(sock)

    return app
