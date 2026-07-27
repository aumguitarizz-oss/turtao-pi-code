import pytest
from unittest.mock import MagicMock


class TestCameraRoutes:
    def test_snapshot_returns_jpeg_bytes(self, client):
        from turtao.api import routes_camera
        cam = MagicMock()
        cam.get_latest_frame.return_value = b"fake_jpeg_bytes"
        routes_camera._deps["camera"] = cam
        resp = client.get("/api/snapshot")
        assert resp.status_code == 200
        assert resp.data == b"fake_jpeg_bytes"

    def test_snapshot_no_frame_returns_503(self, client):
        from turtao.api import routes_camera
        cam = MagicMock()
        cam.get_latest_frame.return_value = None
        routes_camera._deps["camera"] = cam
        resp = client.get("/api/snapshot")
        assert resp.status_code == 503

    def test_stream_returns_multipart_mjpeg(self, client, mock_state):
        import numpy as np
        mock_state.latest_frame = np.zeros((10, 10, 3), dtype=np.uint8)
        resp = client.get("/api/stream")
        assert resp.status_code == 200
        assert "multipart/x-mixed-replace" in resp.content_type
        # Consume one frame from the real stream to prove the generator
        # actually produces valid MJPEG framing, not just correct headers.
        chunk = next(resp.response)
        assert b"--frame" in chunk
        assert b"Content-Type: image/jpeg" in chunk
