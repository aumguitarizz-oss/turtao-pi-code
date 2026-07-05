import pytest


class TestFacesRoutes:
    def test_list_faces_empty(self, client):
        resp = client.get("/api/faces")
        assert resp.status_code == 200
        assert resp.get_json() == []

    def test_list_faces_with_data(self, client, mock_face_engine):
        mock_face_engine._faces["alice"] = 5
        resp = client.get("/api/faces")
        assert resp.status_code == 200
        data = resp.get_json()
        assert len(data) == 1
        assert data[0]["name"] == "alice"
        assert data[0]["poses"] == 5
        assert "thumb_url" in data[0]

    def test_enrollment_flow(self, client, mock_enrollment, mock_face_engine):
        mock_face_engine._faces["alice"] = 5
        resp = client.get("/api/faces")
        assert resp.status_code == 200
        assert len(resp.get_json()) == 1

    def test_enroll_start(self, client, mock_enrollment):
        resp = client.post("/api/faces/enroll/start", json={"name": "bob"})
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["active"] is True
        assert data["poses_total"] == 5

    def test_enroll_start_missing_name(self, client):
        resp = client.post("/api/faces/enroll/start", json={})
        assert resp.status_code == 400
        assert resp.get_json()["error"] == "VALIDATION_ERROR"

    def test_enroll_start_empty_name(self, client):
        resp = client.post("/api/faces/enroll/start", json={"name": ""})
        assert resp.status_code == 400
        assert resp.get_json()["error"] == "VALIDATION_ERROR"

    def test_enroll_start_duplicate_fails(self, client, mock_enrollment, mock_face_engine):
        mock_face_engine._faces["bob"] = 5
        resp = client.post("/api/faces/enroll/start", json={"name": "bob"})
        assert resp.status_code == 409
        assert resp.get_json()["error"] == "FACE_EXISTS"

    def test_enroll_capture_during_active(self, client, mock_enrollment):
        client.post("/api/faces/enroll/start", json={"name": "charlie"})
        for _ in range(4):
            resp = client.post("/api/faces/enroll/capture")
            assert resp.status_code == 200
            data = resp.get_json()
            assert data["active"] is True

    def test_enroll_capture_completes(self, client, mock_enrollment):
        client.post("/api/faces/enroll/start", json={"name": "charlie"})
        for _ in range(5):
            client.post("/api/faces/enroll/capture")
        resp = client.get("/api/faces/enroll/status")
        assert resp.get_json()["active"] is False

    def test_enroll_capture_without_start_returns_error(self, client):
        resp = client.post("/api/faces/enroll/capture")
        assert resp.status_code == 409
        assert resp.get_json()["error"] == "NO_ENROLLMENT"

    def test_enroll_cancel(self, client, mock_enrollment):
        client.post("/api/faces/enroll/start", json={"name": "dave"})
        client.post("/api/faces/enroll/capture")
        resp = client.post("/api/faces/enroll/cancel")
        assert resp.status_code == 200
        assert resp.get_json() == {"ok": True}
        assert mock_enrollment.active is False

    def test_enroll_cancel_without_active_returns_error(self, client):
        resp = client.post("/api/faces/enroll/cancel")
        assert resp.status_code == 200
        assert resp.get_json() == {"ok": True}

    def test_enroll_status_idle(self, client):
        resp = client.get("/api/faces/enroll/status")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["active"] is False

    def test_enroll_status_active(self, client, mock_enrollment):
        client.post("/api/faces/enroll/start", json={"name": "eve"})
        resp = client.get("/api/faces/enroll/status")
        assert resp.status_code == 200
        assert resp.get_json()["active"] is True

    def test_promote_unknown(self, client, mock_face_engine):
        mock_face_engine._unknowns["unk1"] = {"first_seen": "now", "cluster_count": 3}
        resp = client.post("/api/faces/promote", json={"face_id": "unk1", "name": "frank"})
        assert resp.status_code == 200
        assert resp.get_json() == {"ok": True}

    def test_promote_missing_face_id(self, client):
        resp = client.post("/api/faces/promote", json={"face_id": "nonexistent", "name": "frank"})
        assert resp.status_code == 404

    def test_promote_missing_fields(self, client):
        resp = client.post("/api/faces/promote", json={"face_id": "x"})
        assert resp.status_code == 400

    def test_delete_face(self, client, mock_face_engine):
        mock_face_engine._faces["alice"] = 5
        resp = client.delete("/api/faces/alice")
        assert resp.status_code == 200
        assert resp.get_json() == {"ok": True}

    def test_delete_nonexistent_face(self, client):
        resp = client.delete("/api/faces/nobody")
        assert resp.status_code == 404

    def test_delete_unknown(self, client, mock_face_engine):
        mock_face_engine._unknowns["unk1"] = {"first_seen": "now", "cluster_count": 1}
        resp = client.delete("/api/faces/unknowns/unk1")
        assert resp.status_code == 200

    def test_list_unknowns(self, client, mock_face_engine):
        mock_face_engine._unknowns["unk1"] = {"first_seen": "2024-01-01", "cluster_count": 2}
        resp = client.get("/api/faces/unknowns")
        assert resp.status_code == 200
        data = resp.get_json()
        assert len(data) == 1
        assert data[0]["id"] == "unk1"
