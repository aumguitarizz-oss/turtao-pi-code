from flask import jsonify
import logging

logger = logging.getLogger(__name__)


class APIError(Exception):
    def __init__(self, message: str, code: str = "UNKNOWN_ERROR", status: int = 400):
        self.message = message
        self.code = code
        self.status = status


def error_response(message: str, code: str = "UNKNOWN_ERROR", status: int = 400):
    return jsonify({"error": code, "detail": message}), status


def register_error_handlers(app) -> None:
    from turtao.api.errors import APIError

    @app.errorhandler(APIError)
    def handle_api_error(e):
        logger.warning("APIError: %s (%s)", e.message, e.code)
        return error_response(e.message, e.code, e.status)

    @app.errorhandler(404)
    def not_found(e):
        return error_response("Not found", "NOT_FOUND", 404)

    @app.errorhandler(500)
    def internal_error(e):
        logger.exception("Internal server error")
        return error_response("Internal server error", "INTERNAL_ERROR", 500)
