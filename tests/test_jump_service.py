import importlib
import sys
from types import ModuleType, SimpleNamespace


class FakeGate:
    def __init__(self, **_):
        self.phase = "SEARCHING"
        self.ready_values = []

    def update(self, ready, _timestamp):
        self.ready_values.append(ready)
        return SimpleNamespace(
            phase="SEARCHING",
            ready_progress=0.0,
            countdown_remaining_sec=0.0,
        )


class FakeExtractor:
    def process_bgr_frame(self, _frame, _frame_index, _timestamp):
        return SimpleNamespace(detected=True), SimpleNamespace(pose_landmarks=None)

    def close(self):
        pass


def test_double_jump_signal_detection_cannot_bypass_full_body_gate(monkeypatch):
    fake_cv2 = ModuleType("cv2")
    fake_cv2.IMREAD_COLOR = 1
    fake_cv2.imdecode = lambda *_args: SimpleNamespace(shape=(288, 512, 3))
    fake_numpy = ModuleType("numpy")
    fake_numpy.uint8 = object()
    fake_numpy.frombuffer = lambda *_args, **_kwargs: object()
    monkeypatch.setitem(sys.modules, "cv2", fake_cv2)
    monkeypatch.setitem(sys.modules, "numpy", fake_numpy)
    sys.modules.pop("app.jump_service", None)
    jump_service = importlib.import_module("app.jump_service")

    module = SimpleNamespace(
        EngineConfig=lambda: object(),
        PoseSignalExtractor=lambda _config: FakeExtractor(),
        RealtimeCounterEngine=lambda *_args, **_kwargs: object(),
        RealtimeStartGate=FakeGate,
        core_landmarks_visible=lambda *_args: False,
    )
    monkeypatch.setattr(jump_service.importlib, "import_module", lambda _name: module)

    session = jump_service.JumpCounterSession("double", countdown_seconds=5)
    result = session.process(b"frame", max_frame_bytes=1024)

    assert result.ready is False
    assert result.phase == "SEARCHING"
    assert session.gate.ready_values == [False]
