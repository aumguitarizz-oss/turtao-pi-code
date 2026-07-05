import logging
from flask_sock import Sock

logger = logging.getLogger(__name__)


def register_intercom_ws(sock: Sock) -> None:
    @sock.route("/intercom")
    def intercom_ws(ws):
        logger.info("Intercom WS connected")
        try:
            while True:
                data = ws.receive(timeout=1)
                if data is None:
                    break
        except Exception:
            pass
        finally:
            logger.info("Intercom WS disconnected")
