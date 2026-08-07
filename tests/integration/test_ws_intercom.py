from unittest.mock import MagicMock

from turtao.api.ws_intercom import handle_intercom_client


class TestIntercomWsHandler:
    def test_starts_bridge_on_connect_and_stops_on_disconnect(self):
        ws = MagicMock()
        ws.receive.side_effect = [Exception("closed")]
        bridge = MagicMock()

        handle_intercom_client(ws, bridge)

        bridge.start.assert_called_once()
        bridge.stop.assert_called_once()

    def test_receive_timeout_does_not_disconnect_only_exception_does(self):
        # Mirrors ws_status.py's handle_status_client test: a timeout
        # (flask-sock returns None) during a natural pause in speech must
        # not end the connection -- only an actual exception does. The
        # original stub had this backwards (`if data is None: break`),
        # which would have dropped the mic after any ~1s gap.
        ws = MagicMock()
        ws.receive.side_effect = [None, None, None, Exception("closed")]
        bridge = MagicMock()

        handle_intercom_client(ws, bridge)

        assert ws.receive.call_count == 4
        bridge.stop.assert_called_once()

    def test_binary_frames_are_fed_to_the_bridge(self):
        ws = MagicMock()
        ws.receive.side_effect = [b"\x01\x02", b"\x03\x04", Exception("closed")]
        bridge = MagicMock()

        handle_intercom_client(ws, bridge)

        assert bridge.feed_pcm.call_args_list == [
            ((b"\x01\x02",),),
            ((b"\x03\x04",),),
        ]

    def test_non_binary_frames_are_ignored(self):
        # Text frames (e.g. a stray control message) shouldn't be handed
        # to feed_pcm, which expects raw PCM bytes.
        ws = MagicMock()
        ws.receive.side_effect = ["not audio", Exception("closed")]
        bridge = MagicMock()

        handle_intercom_client(ws, bridge)

        bridge.feed_pcm.assert_not_called()

    def test_stops_bridge_even_if_start_never_received_any_audio(self):
        ws = MagicMock()
        ws.receive.side_effect = [Exception("closed")]
        bridge = MagicMock()

        handle_intercom_client(ws, bridge)

        bridge.feed_pcm.assert_not_called()
        bridge.stop.assert_called_once()
