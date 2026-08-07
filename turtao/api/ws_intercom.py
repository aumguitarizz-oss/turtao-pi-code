import logging

from flask_sock import Sock

from turtao.audio.intercom import IntercomBridge

logger = logging.getLogger(__name__)


def handle_intercom_client(ws, intercom_bridge: IntercomBridge) -> None:
    """Run the receive loop for one connected /intercom client.

    A receive timeout is expected during natural pauses in speech and
    must not be treated as a disconnect -- only an actual exception from
    receive() (e.g. ConnectionClosed) ends the loop. Same pattern as
    ws_status.py's handle_status_client.
    """
    logger.info("Intercom WS connected")
    intercom_bridge.start()
    try:
        while True:
            try:
                data = ws.receive(timeout=2)
            except Exception:
                break
            if isinstance(data, (bytes, bytearray)):
                intercom_bridge.feed_pcm(bytes(data))
    finally:
        intercom_bridge.stop()
        logger.info("Intercom WS disconnected")


def register_intercom_ws(sock: Sock, intercom_bridge: IntercomBridge) -> None:
    @sock.route("/intercom")
    def intercom_ws(ws):
        handle_intercom_client(ws, intercom_bridge)
