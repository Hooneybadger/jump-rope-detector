from dataclasses import dataclass

import pytest

from app.signal_resampler import SignalResampler


@dataclass
class Signal:
    frame_idx: int
    time_sec: float
    detected: bool = True
    left_hip_y: float | None = None
    wrist_rotation_ratio: float = 0.0
    wrist_sync_ratio: float = 0.0


def test_low_rate_stream_is_expanded_to_engine_rate():
    resampler = SignalResampler(rate_hz=30.0)
    first = resampler.push(Signal(0, 0.0, left_hip_y=0.50))
    second = resampler.push(Signal(1, 0.1, left_hip_y=0.56, wrist_rotation_ratio=0.09, wrist_sync_ratio=0.8))

    assert [frame.frame_idx for frame in first] == [0]
    assert [frame.frame_idx for frame in second] == [1, 2, 3]
    assert [round(frame.left_hip_y, 3) for frame in second] == [0.52, 0.54, 0.56]
    assert [round(frame.wrist_rotation_ratio, 3) for frame in second] == [0.03, 0.03, 0.03]
    assert all(frame.wrist_sync_ratio == 0.8 for frame in second)
    assert second[-1].time_sec == pytest.approx(0.1)


def test_long_gaps_and_missing_pose_are_not_interpolated():
    resampler = SignalResampler(rate_hz=30.0, max_gap_sec=0.25)
    resampler.push(Signal(0, 0.0, left_hip_y=0.5))
    after_stall = resampler.push(Signal(1, 0.6, left_hip_y=0.7))
    missing = resampler.push(Signal(2, 0.7, detected=False))
    recovered = resampler.push(Signal(3, 0.8, left_hip_y=0.6))

    assert [frame.frame_idx for frame in after_stall] == [18]
    assert [frame.frame_idx for frame in missing] == [21]
    assert [frame.frame_idx for frame in recovered] == [24]


def test_frames_faster_than_engine_rate_do_not_reuse_indices():
    resampler = SignalResampler(rate_hz=30.0)
    resampler.push(Signal(0, 0.0, left_hip_y=0.5))

    assert resampler.push(Signal(1, 0.01, left_hip_y=0.5)) == []
    assert [frame.frame_idx for frame in resampler.push(Signal(2, 0.034, left_hip_y=0.5))] == [1]
