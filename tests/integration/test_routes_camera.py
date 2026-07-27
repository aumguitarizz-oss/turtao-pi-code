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
