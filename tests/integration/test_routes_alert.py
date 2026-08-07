from turtao.state import ThreatLabel


class TestAlertRoute:
    def test_recognized_owner_is_not_reported_as_a_threat(self, client, mock_state):
        # threat_state.active is True for both a SAFE (enrolled) match and a
        # THREAT (unknown) match — only threat_label distinguishes them.
        # Regression test for a real bug: a correctly-recognized, enrolled
        # owner was reported as an intruder because this route read .active
        # instead of the label.
        mock_state.threat_state.active = True
        mock_state.threat_label = ThreatLabel.SAFE
        resp = client.get("/api/alert")
        assert resp.status_code == 200
        assert resp.get_json()["threat"] is False

    def test_unknown_face_is_reported_as_a_threat(self, client, mock_state):
        mock_state.threat_state.active = True
        mock_state.threat_label = ThreatLabel.THREAT
        resp = client.get("/api/alert")
        assert resp.status_code == 200
        assert resp.get_json()["threat"] is True

    def test_idle_is_not_reported_as_a_threat(self, client, mock_state):
        mock_state.threat_state.active = False
        mock_state.threat_label = ThreatLabel.IDLE
        resp = client.get("/api/alert")
        assert resp.status_code == 200
        assert resp.get_json()["threat"] is False
