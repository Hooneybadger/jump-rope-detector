"""Validated browser pose protocol and legacy result adapter."""
from __future__ import annotations

import math
from types import SimpleNamespace

MAX_POSE_BYTES = 16_384


def finite_number(value, lower: float, upper: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("expected numeric pose value")
    try:
        value = float(value)
    except OverflowError as error:
        raise ValueError("pose value out of range") from error
    if not math.isfinite(value) or not lower <= value <= upper:
        raise ValueError("pose value out of range")
    return value


def decode_pose(message: dict):
    if not isinstance(message, dict) or message.get("type") != "pose":
        raise ValueError("expected pose message")
    timestamp_ms = finite_number(message.get("timestampMs"), 0, 1e12)
    inference_ms = finite_number(message.get("inferenceMs"), 0, 60_000)
    points = message.get("landmarks")
    if not isinstance(points, list) or len(points) not in (0, 33):
        raise ValueError("expected zero or 33 landmarks")
    landmarks = []
    for point in points:
        if not isinstance(point, list) or len(point) != 5:
            raise ValueError("expected x, y, z, visibility, presence")
        # Landmarks outside the image are valid MediaPipe outputs. Preserve them
        # for the existing full-body gate instead of clamping them into view.
        x, y, z = [finite_number(v, -10, 10) for v in point[:3]]
        visibility, presence = [finite_number(v, 0, 1) for v in point[3:]]
        landmarks.append(SimpleNamespace(x=x, y=y, z=z, visibility=visibility, presence=presence))
    result = SimpleNamespace(pose_landmarks=SimpleNamespace(landmark=landmarks) if landmarks else None)
    return timestamp_ms, inference_ms, result


class PoseSamplingClock:
    """Keep the counting input at its validated rate while inference runs faster."""

    def __init__(self, rate_hz: float = 20.0):
        self.rate_hz = rate_hz
        self.origin = None
        self.last_slot = -1

    def accept(self, timestamp: float) -> bool:
        if self.origin is None:
            self.origin = timestamp
        slot = math.floor((timestamp - self.origin) * self.rate_hz + 1e-7)
        if slot <= self.last_slot:
            return False
        self.last_slot = slot
        return True
