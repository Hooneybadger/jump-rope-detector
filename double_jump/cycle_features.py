from __future__ import annotations

import math
from typing import TYPE_CHECKING, Sequence

import numpy as np

if TYPE_CHECKING:
    from double_jump.counter_engine import SignalFrame


FEATURE_NAMES = [
    "left_shoulder_rel_x",
    "left_shoulder_rel_y",
    "right_shoulder_rel_x",
    "right_shoulder_rel_y",
    "left_elbow_rel_x",
    "left_elbow_rel_y",
    "right_elbow_rel_x",
    "right_elbow_rel_y",
    "left_wrist_rel_x",
    "left_wrist_rel_y",
    "right_wrist_rel_x",
    "right_wrist_rel_y",
    "left_knee_rel_x",
    "left_knee_rel_y",
    "right_knee_rel_x",
    "right_knee_rel_y",
    "left_ankle_rel_x",
    "left_ankle_rel_y",
    "right_ankle_rel_x",
    "right_ankle_rel_y",
    "left_forearm_angle",
    "right_forearm_angle",
    "left_upperarm_angle",
    "right_upperarm_angle",
    "left_elbow_angle",
    "right_elbow_angle",
    "torso_tilt",
    "ankle_gap_x",
    "foot_mean_rel_y",
    "wrist_rotation_ratio",
    "wrist_sync_ratio",
    "ankle_flow_ratio",
]


def _point(x: float | None, y: float | None) -> np.ndarray | None:
    if x is None or y is None:
        return None
    return np.array([x, y], dtype=np.float32)


def _mean_point(*points: np.ndarray | None) -> np.ndarray | None:
    valid = [point for point in points if point is not None]
    if not valid:
        return None
    return np.mean(np.stack(valid), axis=0)


def _vector_angle(a: np.ndarray | None, b: np.ndarray | None) -> float:
    if a is None or b is None:
        return 0.0
    vec = b - a
    if float(np.linalg.norm(vec)) <= 1e-6:
        return 0.0
    return math.atan2(float(vec[1]), float(vec[0]))


def _joint_angle(a: np.ndarray | None, b: np.ndarray | None, c: np.ndarray | None) -> float:
    if a is None or b is None or c is None:
        return 0.0
    ba = a - b
    bc = c - b
    norm_ba = float(np.linalg.norm(ba))
    norm_bc = float(np.linalg.norm(bc))
    if norm_ba <= 1e-6 or norm_bc <= 1e-6:
        return 0.0
    cosine = float(np.dot(ba, bc) / (norm_ba * norm_bc))
    cosine = max(-1.0, min(1.0, cosine))
    return math.acos(cosine)


def _relative_xy(point: np.ndarray | None, center: np.ndarray, scale: float) -> tuple[float, float]:
    if point is None:
        return 0.0, 0.0
    rel = (point - center) / max(scale, 1e-5)
    return float(rel[0]), float(rel[1])


def _frame_to_feature_vector(signal: SignalFrame) -> np.ndarray:
    left_shoulder = _point(signal.left_shoulder_x, signal.left_shoulder_y)
    right_shoulder = _point(signal.right_shoulder_x, signal.right_shoulder_y)
    left_elbow = _point(signal.left_elbow_x, signal.left_elbow_y)
    right_elbow = _point(signal.right_elbow_x, signal.right_elbow_y)
    left_wrist = _point(signal.left_wrist_x, signal.left_wrist_y)
    right_wrist = _point(signal.right_wrist_x, signal.right_wrist_y)
    left_hip = _point(signal.left_hip_x, signal.left_hip_y)
    right_hip = _point(signal.right_hip_x, signal.right_hip_y)
    left_knee = _point(signal.left_knee_x, signal.left_knee_y)
    right_knee = _point(signal.right_knee_x, signal.right_knee_y)
    left_ankle = _point(signal.left_ankle_x, signal.left_ankle_y)
    right_ankle = _point(signal.right_ankle_x, signal.right_ankle_y)

    pelvis_center = _mean_point(left_hip, right_hip)
    shoulder_center = _mean_point(left_shoulder, right_shoulder)
    if pelvis_center is None:
        pelvis_center = shoulder_center if shoulder_center is not None else np.zeros(2, dtype=np.float32)

    scale_candidates = [value for value in (signal.leg_length, signal.shoulder_width) if value is not None]
    scale = float(np.median(np.asarray(scale_candidates, dtype=np.float32))) if scale_candidates else 0.25
    scale = max(0.05, scale)

    foot_mean_rel_y = 0.0
    if signal.left_foot_y is not None and signal.right_foot_y is not None:
        foot_mean_rel_y = (((signal.left_foot_y + signal.right_foot_y) / 2.0) - float(pelvis_center[1])) / scale

    features = [
        *_relative_xy(left_shoulder, pelvis_center, scale),
        *_relative_xy(right_shoulder, pelvis_center, scale),
        *_relative_xy(left_elbow, pelvis_center, scale),
        *_relative_xy(right_elbow, pelvis_center, scale),
        *_relative_xy(left_wrist, pelvis_center, scale),
        *_relative_xy(right_wrist, pelvis_center, scale),
        *_relative_xy(left_knee, pelvis_center, scale),
        *_relative_xy(right_knee, pelvis_center, scale),
        *_relative_xy(left_ankle, pelvis_center, scale),
        *_relative_xy(right_ankle, pelvis_center, scale),
        _vector_angle(left_elbow, left_wrist),
        _vector_angle(right_elbow, right_wrist),
        _vector_angle(left_shoulder, left_elbow),
        _vector_angle(right_shoulder, right_elbow),
        _joint_angle(left_shoulder, left_elbow, left_wrist),
        _joint_angle(right_shoulder, right_elbow, right_wrist),
        _vector_angle(pelvis_center, shoulder_center),
        0.0 if left_ankle is None or right_ankle is None else float((right_ankle[0] - left_ankle[0]) / scale),
        float(foot_mean_rel_y),
        float(signal.wrist_rotation_ratio),
        float(signal.wrist_sync_ratio),
        float(signal.ankle_flow_ratio),
    ]
    return np.asarray(features, dtype=np.float32)


def build_cycle_feature_tensor(
    cycle_frames: Sequence[SignalFrame],
    target_frames: int = 32,
) -> np.ndarray:
    if target_frames <= 0:
        raise ValueError("target_frames must be positive")
    if not cycle_frames:
        return np.zeros((target_frames, len(FEATURE_NAMES)), dtype=np.float32)

    raw = np.stack([_frame_to_feature_vector(frame) for frame in cycle_frames], axis=0)
    if raw.shape[0] == target_frames:
        return raw
    if raw.shape[0] == 1:
        return np.repeat(raw, target_frames, axis=0)

    x_old = np.linspace(0.0, 1.0, raw.shape[0], dtype=np.float32)
    x_new = np.linspace(0.0, 1.0, target_frames, dtype=np.float32)
    resampled = np.zeros((target_frames, raw.shape[1]), dtype=np.float32)
    for feature_idx in range(raw.shape[1]):
        resampled[:, feature_idx] = np.interp(x_new, x_old, raw[:, feature_idx]).astype(np.float32)
    return resampled


def flatten_cycle_feature_tensor(tensor: np.ndarray) -> np.ndarray:
    return np.asarray(tensor, dtype=np.float32).reshape(-1)
