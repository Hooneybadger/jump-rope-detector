import sys
from types import ModuleType, SimpleNamespace


class FinishLineSession:
    """Counting has started and the clock sits just under the target time."""

    def __init__(self, mode, countdown_seconds):
        self.mode = mode
        self.count = 0
        self.count_started_at = None
        self._elapsed = 0.0

    @property
    def elapsed(self):
        return self._elapsed

    def process(self, _payload, _max_frame_bytes):
        self.count_started_at = 1.0
        self.count = 37
        self._elapsed = 9.96
        return SimpleNamespace(
            count=self.count, phase="COUNTING", ready=True, framed=True, ready_progress=1.0,
            countdown=0.0, elapsed=self._elapsed, landmarks=[], processing_ms=4.0,
        )

    def close(self):
        pass


def fake_jump_service(monkeypatch, session_type):
    module = ModuleType("app.jump_service")
    module.ANALYSIS_FPS = {"basic": 20, "alternating": 20, "double": 20}
    module.JumpCounterSession = session_type
    monkeypatch.setitem(sys.modules, "app.jump_service", module)


def test_server_finishes_on_time_when_browser_stops_sending_frames(admin_client, monkeypatch):
    fake_jump_service(monkeypatch, FinishLineSession)
    with admin_client.websocket_connect(
        "/ws/count/basic?duration=10&countdown=3", headers={"origin": "http://testserver"}
    ) as socket:
        ready = socket.receive_json()
        assert ready["type"] == "ready" and ready["analysisFps"] == 20
        socket.send_bytes(b"frame")
        state = socket.receive_json()
        # Rounded elapsed already reads 10.0; the browser used to stop here and wait forever.
        assert state["elapsed"] == 10.0 and state["finished"] is False
        complete = socket.receive_json()

    assert complete["type"] == "complete"
    assert complete["count"] == 37
    assert complete["status"] == "completed"
    assert complete["reason"] == "time"
    assert complete["duration"] == 10
    record = admin_client.get(f"/api/workouts/{complete['workoutId']}").json()
    assert record["status"] == "completed" and record["count"] == 37


class NeverReadySession(FinishLineSession):
    def process(self, _payload, _max_frame_bytes):
        return SimpleNamespace(
            count=0, phase="SEARCHING", ready=False, framed=False, ready_progress=0.0,
            countdown=0.0, elapsed=0.0, landmarks=[], processing_ms=4.0,
        )


def test_every_mode_gets_a_final_result_when_stopped(admin_client, monkeypatch):
    fake_jump_service(monkeypatch, NeverReadySession)
    for mode in ("basic", "alternating", "double"):
        with admin_client.websocket_connect(
            f"/ws/count/{mode}?duration=60&countdown=3", headers={"origin": "http://testserver"}
        ) as socket:
            assert socket.receive_json()["type"] == "ready"
            socket.send_bytes(b"frame")
            assert socket.receive_json()["phase"] == "SEARCHING"
            socket.send_text('{"type": "stop"}')
            complete = socket.receive_json()
        assert complete["type"] == "complete"
        assert complete["mode"] == mode
        assert complete["status"] == "interrupted"
        assert complete["reason"] == "not_started"
