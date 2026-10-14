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


LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP, LEFT_ANKLE, RIGHT_ANKLE = 11, 12, 23, 24, 27, 28
# Vertical share of the shoulder->hip vector (cosine of the torso's lean from vertical, normalised image
# units). Skipping keeps the torso upright (lowest 0.67 over all labelled jumping, benchmark and external);
# push-ups and floor exercise lie flat (at most 0.51), so frames below this are treated as no pose.
UPRIGHT_MIN_COS = 0.55


def plausible_pose(result):
    """Drop poses with an ankle above the hips (image y grows downward) or a body that is not upright.

    MediaPipe occasionally returns such a flipped skeleton on the first frames or on
    overlays; a single one seeds the engines' floor/airborne state with an impossible
    height that does not recover for seconds. High-knee steps stay valid.
    A torso leaning more than about 57 degrees from vertical (UPRIGHT_MIN_COS) is also dropped, so floor
    exercises and deep bends between sets (push-ups, stretching, picking up the rope) are not counted as
    jumps. No frame inside a labelled jumping window (20 benchmark and 43 external clips) is dropped.
    """
    if not result.pose_landmarks:
        return result
    points = result.pose_landmarks.landmark
    hip_y = (points[LEFT_HIP].y + points[RIGHT_HIP].y) / 2
    if min(points[LEFT_ANKLE].y, points[RIGHT_ANKLE].y) <= hip_y:
        return SimpleNamespace(pose_landmarks=None)
    lean_dx = (points[LEFT_HIP].x + points[RIGHT_HIP].x - points[LEFT_SHOULDER].x - points[RIGHT_SHOULDER].x) / 2
    lean_dy = hip_y - (points[LEFT_SHOULDER].y + points[RIGHT_SHOULDER].y) / 2
    if lean_dy < UPRIGHT_MIN_COS * math.hypot(lean_dx, lean_dy):
        return SimpleNamespace(pose_landmarks=None)
    return result


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
