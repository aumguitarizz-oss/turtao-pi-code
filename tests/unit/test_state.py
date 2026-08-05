from turtao.state import FaceDetection, ThreatLabel, ThreatState


class TestFaceDetection:
    def test_construction(self):
        fd = FaceDetection(box=(1, 2, 3, 4), name="alice", label=ThreatLabel.SAFE, confidence=0.8)
        assert fd.box == (1, 2, 3, 4)
        assert fd.name == "alice"
        assert fd.label == ThreatLabel.SAFE
        assert fd.confidence == 0.8


class TestThreatStateFaces:
    def test_defaults_to_empty_list(self):
        ts = ThreatState()
        assert ts.faces == []

    def test_faces_is_independent_per_instance(self):
        ts1 = ThreatState()
        ts2 = ThreatState()
        fd = FaceDetection(box=(0, 0, 1, 1), name="x", label=ThreatLabel.THREAT, confidence=0.0)
        ts1.faces.append(fd)
        assert ts2.faces == []
