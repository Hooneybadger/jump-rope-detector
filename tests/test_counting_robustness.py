import importlib
from types import SimpleNamespace

import pytest

from app.pose_protocol import decode_pose, plausible_pose
from app.rope_trip import RopeTripPolicy


def standing_packet(timestamp_ms=1000, left_ankle_y=0.9, right_ankle_y=0.9):
    points = [[0.5, 0.3, 0.0, 1.0, 1.0] for _ in range(33)]
    for index in (23, 24):  # hips
        points[index][1] = 0.55
    points[27][1], points[28][1] = left_ankle_y, right_ankle_y
    return {"type": "pose", "timestampMs": timestamp_ms, "inferenceMs": 4.0, "landmarks": points}


def test_plausible_pose_keeps_standing_and_drops_ankle_above_hips():
    assert plausible_pose(decode_pose(standing_packet())[2]).pose_landmarks is not None
    flipped = decode_pose(standing_packet(left_ankle_y=0.43))[2]
    assert plausible_pose(flipped).pose_landmarks is None
    missing = SimpleNamespace(pose_landmarks=None)
    assert plausible_pose(missing) is missing


def test_plausible_pose_drops_a_flat_torso_such_as_push_ups():
    packet = standing_packet(left_ankle_y=0.88, right_ankle_y=0.88)
    for index in (11, 12):  # shoulders level with the hips, one torso length to the side
        packet["landmarks"][index][:2] = [0.25, 0.83]
    for index in (23, 24):
        packet["landmarks"][index][:2] = [0.40, 0.85]
    assert plausible_pose(decode_pose(packet)[2]).pose_landmarks is None


def event(delta, running):
    return SimpleNamespace(count_delta=delta, running_count=running)


def jumps(policy, start, count, interval=0.4, running=0):
    shown = None
    for k in range(count):
        running += 1
        shown = policy.observe(event(1, running), start + k * interval)
    return shown, running


def test_rope_catch_deducts_once_when_jumping_resumes_after_a_pause():
    policy = RopeTripPolicy()
    shown, running = jumps(policy, 0.0, 10)  # last count at 3.6 s
    running -= 1
    assert policy.observe(event(-1, running), 4.2) == 10  # withheld, not shown yet
    assert policy.observe(event(-1, running - 1), 4.6) == 10  # second verdict in the same pause is ignored
    running = running - 1 + 1
    assert policy.observe(event(1, running), 6.5) == 10  # 10 jumps + restart jump - 1 catch
    assert policy.committed == 1


def test_immediate_continuation_or_no_restart_never_deducts():
    policy = RopeTripPolicy()
    _, running = jumps(policy, 0.0, 10)
    policy.observe(event(-1, running - 1), 4.0)
    assert policy.observe(event(1, running), 4.0) == 11  # 0.4 s after the last jump: a missed airtime, not a catch
    stop = RopeTripPolicy()
    _, running = jumps(stop, 0.0, 10)
    assert stop.observe(event(-1, running - 1), 4.2) == 10
    stop.expire(20.0)  # player stopped for good: the last jump stays counted
    assert stop.displayed(running - 1) == 10 and stop.committed == 0


@pytest.mark.parametrize("mode,expected", [("basic", 30), ("alternating", 60), ("double", 30)])
def test_counter_input_rate_per_mode(mode, expected, monkeypatch):
    from app import jump_service
    importlib.reload(jump_service)
    clock = [100.0]
    monkeypatch.setattr(jump_service.time, "monotonic", lambda: clock[0])
    session = jump_service.JumpCounterSession(mode, input_backend="landmarks")
    for k in range(60):  # one second of 60 fps browser poses
        clock[0] = 100.0 + k / 60
        session.process_landmarks(standing_packet(timestamp_ms=1000 + k * 1000 / 60))
    assert session.frame_index == expected
    session.close()


@pytest.mark.parametrize("hand_turns_per_jump", [1, 2])
def test_double_engine_hand_ratio_tracks_hand_turns_per_jump(hand_turns_per_jump):
    import math

    from double_jump.counter_engine import EngineConfig, RealtimeCounterEngine, SignalFrame

    engine = RealtimeCounterEngine(EngineConfig(classifier_model_path=None))
    jump_hz, rate = 2.2, 30.0
    for k in range(int(2.5 * rate)):
        t = k / rate
        hip = 0.55 - 0.03 * abs(math.sin(math.pi * jump_hz * t))  # one bounce per jump
        angle = 2 * math.pi * jump_hz * hand_turns_per_jump * t  # hands circle once or twice per jump
        hand = dict(left_wrist_x=0.40, left_wrist_y=0.55, right_wrist_x=0.60, right_wrist_y=0.55,
                    left_pinky_x=0.40 + 0.02 * math.cos(angle), left_pinky_y=0.55 + 0.02 * math.sin(angle),
                    left_index_x=0.40 + 0.03 * math.cos(angle), left_index_y=0.55 + 0.03 * math.sin(angle),
                    right_pinky_x=0.60 - 0.02 * math.cos(angle), right_pinky_y=0.55 + 0.02 * math.sin(angle),
                    right_index_x=0.60 - 0.03 * math.cos(angle), right_index_y=0.55 + 0.03 * math.sin(angle))
        engine._update_hand_samples(SignalFrame(frame_idx=k, time_sec=t, detected=True, left_hip_y=hip, right_hip_y=hip,
                                                left_foot_y=0.9, right_foot_y=0.9, leg_length=0.35, **hand))
    ratio, concentration = engine._hand_turn_ratio(2.5 - 1 / rate)
    assert abs(ratio - hand_turns_per_jump) < 0.25
    assert concentration >= EngineConfig().hand_ratio_min_concentration


def test_frame_height_reaches_double_engine_and_is_validated(monkeypatch):
    from app import jump_service
    importlib.reload(jump_service)
    clock = [100.0]
    monkeypatch.setattr(jump_service.time, "monotonic", lambda: clock[0])
    session = jump_service.JumpCounterSession("double", input_backend="landmarks")
    for k in range(2):
        clock[0] = 100.0 + k / 30
        session.process_landmarks({**standing_packet(timestamp_ms=1000 + k * 33), "frameHeight": 288})
    assert session.engine.frame_height_px == 288
    with pytest.raises(ValueError):
        session.process_landmarks({**standing_packet(timestamp_ms=2000), "frameHeight": "tall"})
    session.close()


def run_basic_engine(rhythm_adaptive, hz, hip_amp, foot_amp, noise=0.0, seconds=10.0):
    import math
    import random
    from dataclasses import replace

    import basic_jump.counter_engine as engine_module

    rnd = random.Random(0)
    engine = engine_module.RealtimeCounterEngine(replace(engine_module.EngineConfig(), rhythm_adaptive_enabled=rhythm_adaptive))
    count = 0
    for k in range(int(seconds * 30)):  # 30 Hz counter input, as in the app
        t = k / 30
        lift = max(0.0, math.sin(2 * math.pi * hz * t))
        hip, foot = 0.55 - hip_amp * lift + rnd.gauss(0, noise), 0.90 - foot_amp * lift + rnd.gauss(0, noise)
        signal = engine_module.SignalFrame(frame_idx=k, time_sec=t, detected=True, left_hip_y=hip, right_hip_y=hip,
                                           left_foot_y=foot, right_foot_y=foot, leg_length=0.35)
        if t < 2.0:
            engine.warmup(signal)
            continue
        event = engine.step(signal)
        if event is not None:
            count = event.running_count
    return count


def test_basic_rhythm_gating_keeps_fast_small_jumps_and_ignores_standing():
    expected = round(4.3 * 8)  # 8 s of counting at 4.3 jumps/s (about 258 per minute)
    assert run_basic_engine(False, 4.3, 0.02, 0.015) < 0.5 * expected  # fixed 0.3 s gap and amplitude floors
    assert run_basic_engine(True, 4.3, 0.02, 0.015) >= 0.8 * expected
    # the relaxation only lowers limits while a rhythm is present: ordinary jumping is unchanged
    assert run_basic_engine(True, 2.0, 0.04, 0.04) == run_basic_engine(False, 2.0, 0.04, 0.04) == 16
    assert run_basic_engine(True, 2.0, 0.0, 0.0, noise=0.002) == 0  # standing still with pose jitter
