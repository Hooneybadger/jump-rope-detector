import importlib

import pytest

from app.pose_protocol import decode_pose


def packet():
    return {"type": "pose", "timestampMs": 1000, "inferenceMs": 4.5,
            "landmarks": [[0.5, 0.5, 0.0, 1.0, 1.0] for _ in range(33)]}


@pytest.mark.parametrize("field,value", [("landmarks", [0] * 32), ("timestampMs", float("nan")),
                                          ("inferenceMs", True), ("inferenceMs", -1), ("timestampMs", 10**400)])
def test_bad_pose_rejected(field, value):
    message = packet()
    message[field] = value
    with pytest.raises(ValueError):
        decode_pose(message)


def test_outside_image_and_missing_pose_preserved():
    message = packet()
    message["landmarks"][0][0] = -0.2
    _, _, result = decode_pose(message)
    assert result.pose_landmarks.landmark[0].x == -0.2
    message["landmarks"] = []
    assert decode_pose(message)[2].pose_landmarks is None


@pytest.mark.parametrize("bad", [float("inf"), None, "0.5", True, 11])
def test_invalid_landmark_rejected(bad):
    message = packet()
    message["landmarks"][0][0] = bad
    with pytest.raises(ValueError):
        decode_pose(message)


@pytest.mark.parametrize("mode", ["basic", "alternating", "double"])
def test_external_pose_preserves_signal_postprocessing(mode, monkeypatch):
    module = importlib.import_module(f"{mode}_jump.counter_engine")
    _, _, result = decode_pose(packet())
    monkeypatch.setattr(module.mp_pose, "Pose", lambda **_: pytest.fail("server must not create a model"))
    extractor = module.PoseSignalExtractor(module.EngineConfig(), initialize_pose=False)
    signal1, _ = extractor.process_pose_result(result, 0, 0.0)
    signal2, _ = extractor.process_pose_result(result, 1, 0.05)
    assert signal1.detected and signal2.detected
    if mode == "double":
        assert signal2.wrist_flow_ratio == 0.0
        assert signal2.ankle_flow_ratio == 0.0
        missing = decode_pose({**packet(), "landmarks": []})[2]
        bridged, _ = extractor.process_pose_result(missing, 2, 0.1)
        assert bridged.detected  # Existing missing-pose bridge survives the adapter.
    extractor.close()


def test_client_time_cannot_control_server_finish(monkeypatch):
    from app import jump_service
    importlib.reload(jump_service)
    clock = [100.0]
    monkeypatch.setattr(jump_service.time, "monotonic", lambda: clock[0])
    session = jump_service.JumpCounterSession("basic", countdown_seconds=3, input_backend="landmarks")
    assert session.extractor.pose is None
    session.process_landmarks(packet())
    with pytest.raises(ValueError, match="increase"):
        session.process_landmarks(packet())
    with pytest.raises(ValueError, match="drift"):
        session.process_landmarks({**packet(), "timestampMs": 100_000})
    session.count_started_at = 100.0
    clock[0] = 101.0
    assert session.elapsed == 1.0
    with pytest.raises(ValueError):
        session.process(b"jpeg", 1024)
    session.close()


@pytest.mark.parametrize("mode", ["basic", "alternating", "double"])
def test_browser_stream_roundtrip_and_stop(admin_client, mode):
    importlib.reload(importlib.import_module("app.jump_service"))
    with admin_client.websocket_connect(
        f"/ws/count/{mode}?duration=10&countdown=3&input=landmarks",
        headers={"origin": "http://testserver"},
    ) as socket:
        ready = socket.receive_json()
        assert ready["inputBackend"] == "landmarks"
        assert ready["analysisFps"] == 60
        socket.send_json(packet())
        state = socket.receive_json()
        assert state["type"] == "state"
        assert state["inferenceMs"] == 4.5
        assert len(state["landmarks"]) == 33
        socket.send_json({"type": "stop"})
        complete = socket.receive_json()
        assert complete["type"] == "complete" and complete["mode"] == mode


@pytest.mark.parametrize("bad", [[1], {"type": "pose"}, {"type": "pose", "landmarks": [0] * 33},
                                  {"type": "pose", "padding": "x" * 17_000}])
def test_browser_stream_rejects_malformed_messages(admin_client, bad):
    from starlette.websockets import WebSocketDisconnect
    importlib.reload(importlib.import_module("app.jump_service"))
    with admin_client.websocket_connect(
        "/ws/count/basic?duration=10&countdown=3&input=landmarks",
        headers={"origin": "http://testserver"},
    ) as socket:
        socket.receive_json()
        socket.send_json(bad)
        with pytest.raises(WebSocketDisconnect) as exc:
            socket.receive_json()
        assert exc.value.code == 1003


@pytest.mark.parametrize("query", ["duration=bad", "countdown=bad", "duration=10&input=unknown"])
def test_browser_stream_rejects_invalid_connection_options(admin_client, query):
    from starlette.websockets import WebSocketDisconnect
    with pytest.raises(WebSocketDisconnect) as exc:
        with admin_client.websocket_connect(f"/ws/count/basic?{query}", headers={"origin": "http://testserver"}):
            pass
    assert exc.value.code == 1008



def test_60hz_poses_keep_counter_signals_at_20hz():
    from app.pose_protocol import PoseSamplingClock
    clock = PoseSamplingClock()
    accepted = [i for i in range(60) if clock.accept(i / 60)]
    assert accepted == list(range(0, 60, 3))


def test_pose_sampling_clock_does_not_accumulate_timing_drift():
    from app.pose_protocol import PoseSamplingClock
    clock = PoseSamplingClock()
    timestamps = [i / 59 for i in range(590)]
    assert sum(clock.accept(t) for t in timestamps) == 200


def test_slow_first_gpu_inference_does_not_shift_capture_clock(monkeypatch):
    from app import jump_service
    importlib.reload(jump_service)
    clock = [100.0]
    monkeypatch.setattr(jump_service.time, "monotonic", lambda: clock[0])
    session = jump_service.JumpCounterSession("basic", input_backend="landmarks")
    clock[0] = 106.0
    session.process_landmarks({**packet(), "inferenceMs": 6000})
    clock[0] = 106.05
    session.process_landmarks({**packet(), "timestampMs": 7050})
    assert session.frame_index == 2
    session.close()
