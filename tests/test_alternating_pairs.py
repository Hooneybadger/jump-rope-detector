from types import SimpleNamespace
import time

import pytest

from alternating_jump.counter_engine import CounterEvent, EngineConfig, RealtimeCounterEngine, SignalFrame
from alternating_jump.pair_counter import AlternatingPairCounter, paired_label_times
from app.rope_trip import RopeTripPolicy
from tools.eval.evaluate_engines import count_labels


@pytest.mark.parametrize("feet,expected", [
    ([], 0), (["left"], 0), (["right"], 0),
    (["left", "right"], 1), (["right", "left"], 1),
    (["right", "left", "right"], 1),
    (["right", "left", "right", "left"], 2),
    (["right", "right", "left"], 1),
    (["left", "left", "left", "left"], 0),
])
def test_opposite_observed_steps_make_one_pair(feet, expected):
    pairs = AlternatingPairCounter()
    for i, side in enumerate(feet):
        pairs.add(side, i * 0.3)
    assert pairs.count == expected


def test_pause_drops_unfinished_pair_and_restarts_with_either_foot():
    pairs = AlternatingPairCounter()
    pairs.add("right", 0.0)
    assert not pairs.add("left", 2.0)
    assert pairs.add("right", 2.3)
    assert pairs.count == 1


def feed_step(engine, monkeypatch, side, time_sec, delta=1):
    def raw_step(signal):
        engine.last_decision = SimpleNamespace(side=side)
        engine.accepted_running_count += delta
        return CounterEvent(signal.frame_idx, signal.time_sec, engine.accepted_running_count, delta)
    monkeypatch.setattr(engine, "_step_event", raw_step)
    return engine.step(SignalFrame(frame_idx=round(time_sec * 60), time_sec=time_sec, detected=True))


def test_engine_emits_only_pairs_and_never_estimated_opposite_steps(monkeypatch):
    engine = RealtimeCounterEngine(EngineConfig())
    assert feed_step(engine, monkeypatch, "right", 0.0) is None
    assert feed_step(engine, monkeypatch, "right", 0.5, delta=2) is None
    event = feed_step(engine, monkeypatch, "left", 0.8)
    assert (event.running_count, event.count_delta) == (1, 1)
    assert event.trip_policy_applied
    assert RopeTripPolicy().observe(event, 0.8) == 1
    assert feed_step(engine, monkeypatch, "right", 1.1) is None
    engine.begin_count_phase()
    assert feed_step(engine, monkeypatch, "left", 1.4) is None
    assert feed_step(engine, monkeypatch, "right", 1.7).running_count == 1


def test_catch_correction_is_in_steps_before_pairing(monkeypatch):
    engine = RealtimeCounterEngine(EngineConfig())
    for i in range(6):
        feed_step(engine, monkeypatch, "right" if i % 2 == 0 else "left", i * 0.3)
    assert engine.pairs.count == 3
    assert feed_step(engine, monkeypatch, None, 1.9, delta=-1) is None
    assert engine.pairs.count == 3  # withheld until a confirmed restart
    event = feed_step(engine, monkeypatch, "right", 4.0)
    assert event.count_delta == -1 and event.running_count == 2
    assert feed_step(engine, monkeypatch, "left", 4.3).running_count == 3
    assert engine.step_trip_policy.committed == 1


def test_immediate_continuation_does_not_undo_a_completed_pair(monkeypatch):
    engine = RealtimeCounterEngine(EngineConfig())
    feed_step(engine, monkeypatch, "right", 0.0)
    feed_step(engine, monkeypatch, "left", 0.3)
    feed_step(engine, monkeypatch, None, 0.4, delta=-1)
    assert feed_step(engine, monkeypatch, "right", 0.6) is None
    assert feed_step(engine, monkeypatch, "left", 0.9).running_count == 2
    assert engine.step_trip_policy.committed == 0


def test_ground_truth_is_paired_independently_at_second_step():
    assert paired_label_times([0.0, 0.3, 0.6, 0.9, 1.2]) == [0.3, 0.9]
    assert paired_label_times([0.0, 2.0, 2.3]) == [2.3]
    record = dict(mode="alternating", labels_sec=[0.0, 0.3, 0.6, 0.9],
                  label_feet=["left", "left", "right", "left"])
    assert count_labels(record) == [0.6]
    assert count_labels(dict(mode="basic", labels_sec=[0.0, 0.3])) == [0.0, 0.3]


def pose_signal(frame_idx, side=None, airborne=False):
    left_foot, right_foot = {
        "left": (0.9, 0.78), "right": (0.78, 0.9),
        None: (0.85, 0.85) if airborne else (0.9, 0.9),
    }[side]
    return SignalFrame(
        frame_idx, frame_idx / 60, True,
        left_hip_x=0.45, right_hip_x=0.55, left_hip_y=0.5, right_hip_y=0.5,
        left_knee_y=0.7 if left_foot >= right_foot else 0.6,
        right_knee_y=0.7 if right_foot >= left_foot else 0.6,
        left_foot_y=left_foot, right_foot_y=right_foot, leg_length=0.4,
    )


@pytest.mark.parametrize("feet,counts", [
    (["right", "left", "right", "left", "right"], [0, 1, 1, 2, 2]),
    (["left", "right", "left", "right"], [0, 1, 1, 2]),
    (["right"] * 4, [0, 0, 0, 0]),
    ([None] * 4, [0, 0, 0, 0]),
])
def test_pose_steps_reach_live_session_as_completed_pairs(feet, counts):
    from app.jump_service import JumpCounterSession

    session = JumpCounterSession("alternating", input_backend="landmarks")
    try:
        session.visible = lambda *_: True
        session.gate = SimpleNamespace(update=lambda *_: SimpleNamespace(
            phase="COUNTING", ready_progress=1.0, countdown_remaining_sec=0.0,
        ))
        session.engine = RealtimeCounterEngine(session.config, enable_realtime_compensation=True)
        for i in range(30):
            session.engine.warmup(pose_signal(i))
        i = 30
        for side, expected in zip(feet, counts):
            for k in range(18):
                # Brief flight, then one supporting foot with the opposite knee lifted.
                signal = pose_signal(i, airborne=True) if k < 6 else pose_signal(i, side)
                result = session._advance(signal, SimpleNamespace(pose_landmarks=None),
                                          signal.time_sec, time.perf_counter())
                i += 1
            assert result.count == expected
    finally:
        session.close()
