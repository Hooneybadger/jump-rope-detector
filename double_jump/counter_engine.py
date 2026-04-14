from __future__ import annotations

import json
import math
import xml.etree.ElementTree as ET
from collections import deque
from dataclasses import asdict, dataclass, field
from itertools import product
from pathlib import Path
from statistics import median

import cv2
import mediapipe as mp
import numpy as np

from double_jump.cycle_classifier import CycleClassifier, CyclePrediction


mp_pose = mp.solutions.pose

FOOT_LANDMARKS = {
    "left": (
        mp_pose.PoseLandmark.LEFT_ANKLE,
        mp_pose.PoseLandmark.LEFT_HEEL,
        mp_pose.PoseLandmark.LEFT_FOOT_INDEX,
    ),
    "right": (
        mp_pose.PoseLandmark.RIGHT_ANKLE,
        mp_pose.PoseLandmark.RIGHT_HEEL,
        mp_pose.PoseLandmark.RIGHT_FOOT_INDEX,
    ),
}
CORE_READY_LANDMARKS = (
    mp_pose.PoseLandmark.LEFT_SHOULDER,
    mp_pose.PoseLandmark.RIGHT_SHOULDER,
    mp_pose.PoseLandmark.LEFT_ELBOW,
    mp_pose.PoseLandmark.RIGHT_ELBOW,
    mp_pose.PoseLandmark.LEFT_WRIST,
    mp_pose.PoseLandmark.RIGHT_WRIST,
    mp_pose.PoseLandmark.LEFT_HIP,
    mp_pose.PoseLandmark.RIGHT_HIP,
    mp_pose.PoseLandmark.LEFT_KNEE,
    mp_pose.PoseLandmark.RIGHT_KNEE,
    mp_pose.PoseLandmark.LEFT_ANKLE,
    mp_pose.PoseLandmark.RIGHT_ANKLE,
    mp_pose.PoseLandmark.LEFT_HEEL,
    mp_pose.PoseLandmark.RIGHT_HEEL,
    mp_pose.PoseLandmark.LEFT_FOOT_INDEX,
    mp_pose.PoseLandmark.RIGHT_FOOT_INDEX,
)


@dataclass(frozen=True)
class EngineConfig:
    hip_visibility_threshold: float = 0.20
    foot_visibility_threshold: float = 0.15
    arm_visibility_threshold: float = 0.10
    wrist_visibility_threshold: float = 0.05
    ema_alpha_hip: float = 0.45
    ema_alpha_foot: float = 0.55
    fast_ema_alpha_hip: float = 0.25
    fast_ema_alpha_foot: float = 0.35
    baseline_alpha_hip: float = 0.02
    baseline_alpha_foot: float = 0.04
    floor_decay_ratio: float = 0.004
    contact_margin_ratio: float = 0.06
    symmetry_y_ratio: float = 0.16
    takeoff_height_ratio: float = 0.008
    takeoff_hip_ratio: float = 0.004
    min_airborne_frames: int = 1
    max_airborne_frames: int = 28
    landing_contact_frames: int = 1
    fast_mode_cadence_threshold: int = 7
    min_refractory_frames: int = 4
    fast_min_refractory_frames: int = 2
    motion_window_frames: int = 18
    recent_window_frames: int = 5
    min_hip_range_ratio: float = 0.040
    min_foot_range_ratio: float = 0.030
    min_recent_hip_range_ratio: float = 0.024
    max_foot_to_hip_ratio: float = 2.8
    guard_low_hip_range_ratio: float = 0.07
    guard_high_foot_range_ratio: float = 0.12
    guard_recent_hip_range_ratio: float = 0.03
    balanced_override_hip_range_ratio: float = 0.060
    balanced_override_foot_range_ratio: float = 0.055
    balanced_override_recent_hip_ratio: float = 0.020
    balanced_override_min_ratio: float = 0.55
    balanced_override_max_ratio: float = 1.60
    extended_override_hip_range_ratio: float = 0.050
    extended_override_foot_range_ratio: float = 0.090
    extended_override_recent_hip_ratio: float = 0.024
    extended_override_min_ratio: float = 1.50
    extended_override_max_ratio: float = 2.10
    extended_override_recent_to_hip_ratio: float = 0.42
    foot_floor_override_hip_range_ratio: float = 0.085
    foot_floor_override_foot_range_ratio: float = 0.025
    foot_floor_override_recent_hip_ratio: float = 0.050
    foot_floor_override_max_ratio: float = 0.34
    stale_tail_guard_hip_range_ratio: float = 0.18
    stale_tail_guard_recent_to_hip_ratio: float = 0.17
    min_jump_height_ratio: float = 0.018
    min_hip_lift_ratio: float = 0.0
    min_count_gap_frames: int = 7
    adaptive_gap_enabled: bool = True
    adaptive_gap_factor: float = 0.72
    adaptive_gap_history: int = 5
    adaptive_gap_min_intervals: int = 1
    adaptive_gap_floor_frames: int = 4
    adaptive_motion_hip_ratio: float = 0.05
    adaptive_motion_foot_ratio: float = 0.04
    adaptive_recent_hip_enabled: bool = True
    adaptive_recent_hip_floor: float = 0.022
    max_pose_missing_bridge_frames: int = 1
    pose_bridge_decay: float = 0.92
    pose_input_scale: float = 1.75
    optical_flow_winsize: int = 17
    wrist_roi_radius_ratio: float = 0.26
    forearm_roi_length_ratio: float = 0.42
    forearm_roi_thickness_ratio: float = 0.16
    wrist_bg_radius_ratio: float = 0.54
    wrist_bg_thickness_ratio: float = 0.22
    wrist_flow_quantile: float = 0.65
    wrist_flow_background_quantile: float = 0.50
    wrist_flow_baseline_alpha: float = 0.08
    wrist_flow_smoothing_alpha: float = 0.40
    wrist_contact_history_size: int = 45
    wrist_baseline_quantile: float = 0.40
    wrist_flow_active_ratio: float = 0.010
    wrist_flow_peak_ratio: float = 0.015
    wrist_flow_mean_ratio: float = 0.008
    min_wrist_flow_active_frames: int = 1
    wrist_rotation_peak_ratio: float = 0.010
    wrist_rotation_mean_ratio: float = 0.005
    wrist_sync_min_ratio: float = 0.10
    ankle_flow_peak_ratio: float = 0.010
    ankle_flow_active_ratio: float = 0.008
    min_rope_pass_hints: int = 0
    rope_pass_separation_frames: int = 2
    cycle_feature_frames: int = 32
    classifier_confidence_threshold: float = 0.45
    classifier_model_path: str | None = "double_jump/artifacts/cycle_classifier.json"
    rope_stuck_window_frames: int = 12
    rope_stuck_min_hold_frames: int = 3
    rope_stuck_contact_ratio: float = 0.70
    rope_stuck_attempt_flow_ratio: float = 0.010

    def to_dict(self) -> dict[str, float | int | bool]:
        return asdict(self)


@dataclass
class VideoMeta:
    stem: str
    path: str
    fps: float
    frame_count: int
    width: int
    height: int


@dataclass
class LabelEvent:
    frame_idx: int
    time_sec: float
    point_count: int
    source_indices: list[int]
    anomaly_tags: list[str]


@dataclass
class SignalFrame:
    frame_idx: int
    time_sec: float
    detected: bool
    left_shoulder_x: float | None = None
    left_shoulder_y: float | None = None
    right_shoulder_x: float | None = None
    right_shoulder_y: float | None = None
    left_elbow_x: float | None = None
    left_elbow_y: float | None = None
    right_elbow_x: float | None = None
    right_elbow_y: float | None = None
    left_wrist_x: float | None = None
    left_wrist_y: float | None = None
    right_wrist_x: float | None = None
    right_wrist_y: float | None = None
    left_hip_x: float | None = None
    left_ankle_x: float | None = None
    left_ankle_y: float | None = None
    right_hip_x: float | None = None
    right_ankle_x: float | None = None
    right_ankle_y: float | None = None
    left_hip_y: float | None = None
    right_hip_y: float | None = None
    left_knee_x: float | None = None
    left_knee_y: float | None = None
    right_knee_x: float | None = None
    right_knee_y: float | None = None
    left_foot_y: float | None = None
    right_foot_y: float | None = None
    leg_length: float | None = None
    shoulder_width: float | None = None
    left_wrist_flow_ratio: float = 0.0
    right_wrist_flow_ratio: float = 0.0
    wrist_flow_ratio: float = 0.0
    left_wrist_rotation_ratio: float = 0.0
    right_wrist_rotation_ratio: float = 0.0
    wrist_rotation_ratio: float = 0.0
    wrist_sync_ratio: float = 0.0
    ankle_flow_ratio: float = 0.0


@dataclass
class CounterEvent:
    frame_idx: int
    time_sec: float
    running_count: int
    count_delta: int = 1


@dataclass
class CounterDecision:
    frame_idx: int
    time_sec: float
    accepted: bool
    reason: str
    airtime_frames: int
    jump_height_ratio: float
    hip_lift_ratio: float
    wrist_flow_peak_ratio: float
    wrist_flow_mean_ratio: float
    wrist_flow_active_frames: int
    current_wrist_flow_ratio: float
    wrist_flow_baseline_ratio: float
    wrist_rotation_peak_ratio: float
    wrist_rotation_mean_ratio: float
    wrist_sync_peak_ratio: float
    rope_pass_hints: int
    classifier_label: str
    classifier_confidence: float
    classifier_source: str
    min_gap_frames: int
    cadence_locked: bool


@dataclass
class MonitorState:
    frame_idx: int = 0
    time_sec: float = 0.0
    detected: bool = False
    contact_gate: bool = False
    in_air: bool = False
    jump_height_ratio: float = 0.0
    hip_lift_ratio: float = 0.0
    wrist_flow_ratio: float = 0.0
    wrist_flow_baseline_ratio: float = 0.0
    wrist_flow_peak_ratio: float = 0.0
    wrist_flow_active_frames: int = 0
    wrist_rotation_ratio: float = 0.0
    wrist_sync_ratio: float = 0.0
    ankle_flow_ratio: float = 0.0
    rope_pass_hints: int = 0
    cycle_label: str = "-"
    cycle_confidence: float = 0.0
    cycle_source: str = "-"
    cadence_locked: bool = False


@dataclass
class JumpCycleEvidence:
    start_frame: int
    end_frame: int | None = None
    airborne_frames: int = 0
    max_jump_height_ratio: float = 0.0
    max_hip_lift_ratio: float = 0.0
    wrist_rotation_peak_ratio: float = 0.0
    wrist_rotation_sum_ratio: float = 0.0
    wrist_sync_peak_ratio: float = 0.0
    ankle_flow_peak_ratio: float = 0.0
    rope_pass_hints: int = 0
    rope_pass_frames: list[int] = field(default_factory=list)
    rotation_gate_active: bool = False
    counted: bool = False
    prediction: CyclePrediction | None = None


@dataclass(frozen=True)
class LabelWindowConfig:
    start_offset_frames: int = -15
    end_offset_frames: int = 3
    warmup_frames: int = 10

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


@dataclass
class VideoResult:
    stem: str
    gt_count: int
    predicted_count: int
    count_error: int
    exact_match: bool
    eval_start_frame: int
    eval_end_frame: int
    predicted_frames: list[int]


@dataclass
class StreamState:
    phase: str
    ready_progress: float
    countdown_remaining_sec: float
    count_started_at_sec: float | None


@dataclass
class _PendingCountCompensation:
    frame_idx: int
    time_sec: float
    observed_frames: int = 0
    contact_frames: int = 0
    airborne_seen: bool = False
    max_attempt_flow_ratio: float = 0.0


def probe_video(path: str | Path) -> VideoMeta:
    path = Path(path)
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Unable to open video: {path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 30.0)
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    capture.release()
    return VideoMeta(
        stem=path.stem,
        path=str(path),
        fps=fps,
        frame_count=frame_count,
        width=width,
        height=height,
    )


def _parse_label_points(keyframe: ET.Element) -> tuple[int, list[str]]:
    drawings = keyframe.find("Drawings")
    if drawings is None:
        return 0, ["missing_drawings"]
    point_count = 0
    for pencil in drawings.findall("Pencil"):
        point_list = pencil.find("PointList")
        if point_list is not None and point_list.findtext("Point"):
            point_count += 1
    anomaly_tags: list[str] = []
    if point_count != 2:
        anomaly_tags.append(f"point_count_{point_count}")
    return point_count, anomaly_tags


def _timestamp_mode(root: ET.Element) -> tuple[str, float]:
    avg_ts_per_frame = float(root.findtext("AverageTimeStampsPerFrame", "0"))
    return ("ms", avg_ts_per_frame) if avg_ts_per_frame < 100 else ("ticks", max(avg_ts_per_frame, 1.0))


def _timestamp_to_frame(timestamp_raw: int, mode: str, fps: float, ticks_per_frame: float) -> tuple[int, float]:
    if mode == "ms":
        time_sec = timestamp_raw / 1000.0
        return int(round(time_sec * fps)), time_sec
    frame_idx = int(round(timestamp_raw / ticks_per_frame))
    return frame_idx, frame_idx / fps


def parse_label_file(path: str | Path, fps: float = 30.0) -> list[LabelEvent]:
    path = Path(path)
    root = ET.parse(path).getroot()
    mode, ticks_per_frame = _timestamp_mode(root)
    keyframes = root.find("Keyframes")
    if keyframes is None:
        return []

    raw_events: list[LabelEvent] = []
    for index, keyframe in enumerate(keyframes.findall("Keyframe")):
        point_count, anomaly_tags = _parse_label_points(keyframe)
        frame_idx, time_sec = _timestamp_to_frame(
            int(keyframe.findtext("Timestamp", "0")),
            mode,
            fps,
            ticks_per_frame,
        )
        raw_events.append(
            LabelEvent(
                frame_idx=frame_idx,
                time_sec=time_sec,
                point_count=point_count,
                source_indices=[index],
                anomaly_tags=anomaly_tags,
            )
        )

    merged: list[LabelEvent] = []
    i = 0
    while i < len(raw_events):
        current = raw_events[i]
        if (
            current.point_count == 1
            and i + 1 < len(raw_events)
            and raw_events[i + 1].point_count == 1
            and abs(raw_events[i + 1].frame_idx - current.frame_idx) <= 1
            and abs(raw_events[i + 1].time_sec - current.time_sec) <= 0.020
        ):
            nxt = raw_events[i + 1]
            merged.append(
                LabelEvent(
                    frame_idx=int(round((current.frame_idx + nxt.frame_idx) / 2.0)),
                    time_sec=(current.time_sec + nxt.time_sec) / 2.0,
                    point_count=2,
                    source_indices=current.source_indices + nxt.source_indices,
                    anomaly_tags=current.anomaly_tags + nxt.anomaly_tags + ["merged_single_foot_pair"],
                )
            )
            i += 2
            continue
        merged.append(current)
        i += 1
    return merged


def load_ground_truth(label_dir: str | Path, video_dir: str | Path) -> dict[str, list[LabelEvent]]:
    label_dir = Path(label_dir)
    video_dir = Path(video_dir)
    ground_truth: dict[str, list[LabelEvent]] = {}
    for label_path in sorted(label_dir.glob("*.kva")):
        video_meta = probe_video(video_dir / f"{label_path.stem}.mp4")
        ground_truth[label_path.stem] = parse_label_file(label_path, fps=video_meta.fps)
    return ground_truth


def _pick_foot_y(lms, side: str, visibility_threshold: float) -> float | None:
    candidates: list[tuple[float, float]] = []
    for landmark_enum in FOOT_LANDMARKS[side]:
        landmark = lms[landmark_enum.value]
        candidates.append((float(landmark.visibility), float(landmark.y)))
    visible = [item for item in candidates if item[0] >= visibility_threshold]
    target = visible or candidates
    if not target:
        return None
    return sum(item[1] for item in target) / len(target)


def _visible_xy(lms, landmark_enum, threshold: float) -> tuple[float, float] | None:
    landmark = lms[landmark_enum.value]
    if float(landmark.visibility) < threshold:
        return None
    return float(landmark.x), float(landmark.y)


def _xy(lms, landmark_enum) -> tuple[float, float]:
    landmark = lms[landmark_enum.value]
    return float(landmark.x), float(landmark.y)


def pose_result_to_signal(
    result,
    frame_idx: int,
    timestamp_sec: float,
    config: EngineConfig,
) -> SignalFrame:
    if not result.pose_landmarks:
        return SignalFrame(frame_idx=frame_idx, time_sec=timestamp_sec, detected=False)

    lms = result.pose_landmarks.landmark
    left_hip = lms[mp_pose.PoseLandmark.LEFT_HIP.value]
    right_hip = lms[mp_pose.PoseLandmark.RIGHT_HIP.value]
    left_ankle = lms[mp_pose.PoseLandmark.LEFT_ANKLE.value]
    right_ankle = lms[mp_pose.PoseLandmark.RIGHT_ANKLE.value]
    left_knee = _xy(lms, mp_pose.PoseLandmark.LEFT_KNEE)
    right_knee = _xy(lms, mp_pose.PoseLandmark.RIGHT_KNEE)
    left_foot_y = _pick_foot_y(lms, "left", config.foot_visibility_threshold)
    right_foot_y = _pick_foot_y(lms, "right", config.foot_visibility_threshold)
    if left_foot_y is None or right_foot_y is None:
        return SignalFrame(frame_idx=frame_idx, time_sec=timestamp_sec, detected=False)

    left_shoulder = _visible_xy(lms, mp_pose.PoseLandmark.LEFT_SHOULDER, config.arm_visibility_threshold)
    right_shoulder = _visible_xy(lms, mp_pose.PoseLandmark.RIGHT_SHOULDER, config.arm_visibility_threshold)
    left_elbow = _visible_xy(lms, mp_pose.PoseLandmark.LEFT_ELBOW, config.arm_visibility_threshold)
    right_elbow = _visible_xy(lms, mp_pose.PoseLandmark.RIGHT_ELBOW, config.arm_visibility_threshold)
    left_wrist = _visible_xy(lms, mp_pose.PoseLandmark.LEFT_WRIST, config.wrist_visibility_threshold)
    right_wrist = _visible_xy(lms, mp_pose.PoseLandmark.RIGHT_WRIST, config.wrist_visibility_threshold)
    if left_elbow is None:
        landmark = lms[mp_pose.PoseLandmark.LEFT_ELBOW.value]
        if float(landmark.visibility) > 0.0:
            left_elbow = (float(landmark.x), float(landmark.y))
    if right_elbow is None:
        landmark = lms[mp_pose.PoseLandmark.RIGHT_ELBOW.value]
        if float(landmark.visibility) > 0.0:
            right_elbow = (float(landmark.x), float(landmark.y))
    if left_wrist is None:
        landmark = lms[mp_pose.PoseLandmark.LEFT_WRIST.value]
        if float(landmark.visibility) > 0.0:
            left_wrist = (float(landmark.x), float(landmark.y))
    if right_wrist is None:
        landmark = lms[mp_pose.PoseLandmark.RIGHT_WRIST.value]
        if float(landmark.visibility) > 0.0:
            right_wrist = (float(landmark.x), float(landmark.y))

    leg_length = max(
        0.05,
        (
            abs(float(left_hip.y) - float(left_ankle.y))
            + abs(float(right_hip.y) - float(right_ankle.y))
        )
        / 2.0,
    )
    shoulder_width = None
    if left_shoulder is not None and right_shoulder is not None:
        shoulder_width = max(
            0.02,
            math.hypot(
                left_shoulder[0] - right_shoulder[0],
                left_shoulder[1] - right_shoulder[1],
            ),
        )

    return SignalFrame(
        frame_idx=frame_idx,
        time_sec=timestamp_sec,
        detected=True,
        left_shoulder_x=None if left_shoulder is None else left_shoulder[0],
        left_shoulder_y=None if left_shoulder is None else left_shoulder[1],
        right_shoulder_x=None if right_shoulder is None else right_shoulder[0],
        right_shoulder_y=None if right_shoulder is None else right_shoulder[1],
        left_elbow_x=None if left_elbow is None else left_elbow[0],
        left_elbow_y=None if left_elbow is None else left_elbow[1],
        right_elbow_x=None if right_elbow is None else right_elbow[0],
        right_elbow_y=None if right_elbow is None else right_elbow[1],
        left_wrist_x=None if left_wrist is None else left_wrist[0],
        left_wrist_y=None if left_wrist is None else left_wrist[1],
        right_wrist_x=None if right_wrist is None else right_wrist[0],
        right_wrist_y=None if right_wrist is None else right_wrist[1],
        left_hip_x=float(left_hip.x),
        left_ankle_x=float(left_ankle.x),
        left_ankle_y=float(left_ankle.y),
        right_hip_x=float(right_hip.x),
        right_ankle_x=float(right_ankle.x),
        right_ankle_y=float(right_ankle.y),
        left_hip_y=float(left_hip.y),
        right_hip_y=float(right_hip.y),
        left_knee_x=left_knee[0],
        left_knee_y=left_knee[1],
        right_knee_x=right_knee[0],
        right_knee_y=right_knee[1],
        left_foot_y=left_foot_y,
        right_foot_y=right_foot_y,
        leg_length=leg_length,
        shoulder_width=shoulder_width,
    )


def _mean_pose_values(signal: SignalFrame) -> tuple[float, float, float]:
    assert signal.left_hip_y is not None
    assert signal.right_hip_y is not None
    assert signal.left_foot_y is not None
    assert signal.right_foot_y is not None
    assert signal.leg_length is not None
    mean_hip_y = (signal.left_hip_y + signal.right_hip_y) / 2.0
    mean_foot_y = (signal.left_foot_y + signal.right_foot_y) / 2.0
    return mean_hip_y, mean_foot_y, signal.leg_length


def _landmark_visibility_ratio(result, landmarks, visibility_threshold: float = 0.30) -> float:
    if not result.pose_landmarks:
        return 0.0
    lms = result.pose_landmarks.landmark
    visible_count = sum(
        1 for landmark in landmarks if float(lms[landmark.value].visibility) >= visibility_threshold
    )
    return visible_count / max(1, len(landmarks))


def core_landmarks_visible(
    result,
    visibility_threshold: float = 0.30,
    required_ratio: float = 0.80,
) -> bool:
    return _landmark_visibility_ratio(result, CORE_READY_LANDMARKS, visibility_threshold) >= required_ratio


class PoseSignalExtractor:
    def __init__(self, config: EngineConfig):
        self.config = config
        self.pose = mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.prev_signal: SignalFrame | None = None
        self.last_valid_signal: SignalFrame | None = None
        self.prev_valid_signal: SignalFrame | None = None
        self.missing_bridge_streak = 0

    def close(self) -> None:
        self.pose.close()

    @staticmethod
    def _wrap_angle(delta: float) -> float:
        while delta > math.pi:
            delta -= math.tau
        while delta < -math.pi:
            delta += math.tau
        return delta

    @staticmethod
    def _point(x: float | None, y: float | None) -> np.ndarray | None:
        if x is None or y is None:
            return None
        return np.array([x, y], dtype=np.float32)

    @staticmethod
    def _clamp_unit(value: float | None) -> float | None:
        if value is None:
            return None
        return max(0.0, min(1.0, value))

    @staticmethod
    def _clamp_positive(value: float | None) -> float | None:
        if value is None:
            return None
        return max(1e-5, value)

    def _body_center(self, signal: SignalFrame) -> np.ndarray | None:
        points = [
            self._point(signal.left_shoulder_x, signal.left_shoulder_y),
            self._point(signal.right_shoulder_x, signal.right_shoulder_y),
        ]
        valid_points = [point for point in points if point is not None]
        if not valid_points:
            return None
        return np.mean(np.stack(valid_points), axis=0)

    def _body_shift(self, prev_signal: SignalFrame, signal: SignalFrame) -> np.ndarray:
        prev_center = self._body_center(prev_signal)
        curr_center = self._body_center(signal)
        if prev_center is None or curr_center is None:
            return np.zeros(2, dtype=np.float32)
        return curr_center - prev_center

    def _scale_ratio(
        self,
        signal: SignalFrame,
        prev_signal: SignalFrame | None = None,
    ) -> float:
        leg_candidates = [value for value in (signal.leg_length, None if prev_signal is None else prev_signal.leg_length) if value is not None]
        shoulder_candidates = [
            value for value in (signal.shoulder_width, None if prev_signal is None else prev_signal.shoulder_width) if value is not None
        ]
        leg_scale = median(leg_candidates) if leg_candidates else 0.0
        shoulder_scale = median(shoulder_candidates) if shoulder_candidates else 0.0
        return max(0.04, leg_scale * 0.32, shoulder_scale * 0.70)

    def _wrist_kinematics(
        self,
        prev_signal: SignalFrame,
        signal: SignalFrame,
        wrist_attr: tuple[str, str],
        elbow_attr: tuple[str, str],
        body_shift: np.ndarray,
        scale_ratio: float,
    ) -> tuple[float, float]:
        prev_wrist = self._point(getattr(prev_signal, wrist_attr[0]), getattr(prev_signal, wrist_attr[1]))
        curr_wrist = self._point(getattr(signal, wrist_attr[0]), getattr(signal, wrist_attr[1]))
        prev_elbow = self._point(getattr(prev_signal, elbow_attr[0]), getattr(prev_signal, elbow_attr[1]))
        curr_elbow = self._point(getattr(signal, elbow_attr[0]), getattr(signal, elbow_attr[1]))
        if prev_wrist is None or curr_wrist is None or prev_elbow is None or curr_elbow is None:
            return 0.0, 0.0

        prev_forearm = prev_wrist - prev_elbow
        curr_forearm = curr_wrist - curr_elbow
        prev_len = float(np.linalg.norm(prev_forearm))
        curr_len = float(np.linalg.norm(curr_forearm))
        if prev_len < 1e-5 or curr_len < 1e-5:
            return 0.0, 0.0

        relative_delta = curr_forearm - prev_forearm
        wrist_disp = (curr_wrist - prev_wrist) - body_shift
        elbow_disp = (curr_elbow - prev_elbow) - body_shift
        joint_delta = wrist_disp - elbow_disp
        avg_forearm = (prev_forearm + curr_forearm) / 2.0
        avg_len = max(1e-5, float(np.linalg.norm(avg_forearm)))
        direction = avg_forearm / avg_len
        tangent = np.array([-direction[1], direction[0]], dtype=np.float32)
        tangential = abs(float(np.dot(relative_delta, tangent)))
        radial = abs(float(np.dot(relative_delta, direction)))
        angle_prev = math.atan2(float(prev_forearm[1]), float(prev_forearm[0]))
        angle_curr = math.atan2(float(curr_forearm[1]), float(curr_forearm[0]))
        angle_delta = abs(self._wrap_angle(angle_curr - angle_prev))
        angle_motion = angle_delta * ((prev_len + curr_len) / 2.0)
        joint_speed = float(np.linalg.norm(joint_delta))
        motion_ratio = max(joint_speed, tangential, angle_motion) / scale_ratio
        rotation_ratio = max(0.0, max(tangential, angle_motion) - (0.25 * radial)) / scale_ratio
        return motion_ratio, rotation_ratio

    def _ankle_motion_ratio(self, prev_signal: SignalFrame, signal: SignalFrame, body_shift: np.ndarray, scale_ratio: float) -> float:
        ankle_motion: list[float] = []
        for ankle_x_attr, foot_y_attr in (
            ("left_ankle_x", "left_foot_y"),
            ("right_ankle_x", "right_foot_y"),
        ):
            prev_x = getattr(prev_signal, ankle_x_attr)
            curr_x = getattr(signal, ankle_x_attr)
            prev_y = getattr(prev_signal, foot_y_attr)
            curr_y = getattr(signal, foot_y_attr)
            if prev_x is None or curr_x is None or prev_y is None or curr_y is None:
                continue
            prev_point = np.array([prev_x, prev_y], dtype=np.float32)
            curr_point = np.array([curr_x, curr_y], dtype=np.float32)
            disp = (curr_point - prev_point) - body_shift
            ankle_motion.append(float(np.linalg.norm(disp)))
        if not ankle_motion:
            return 0.0
        return max(ankle_motion) / scale_ratio

    def _measure_pose_kinematics(self, signal: SignalFrame) -> tuple[float, float, float, float, float]:
        prev_signal = self.prev_signal
        if prev_signal is None or not signal.detected or not prev_signal.detected:
            return 0.0, 0.0, 0.0, 0.0, 0.0
        body_shift = self._body_shift(prev_signal, signal)
        scale_ratio = self._scale_ratio(signal, prev_signal)
        left_ratio, left_rotation = self._wrist_kinematics(
            prev_signal,
            signal,
            ("left_wrist_x", "left_wrist_y"),
            ("left_elbow_x", "left_elbow_y"),
            body_shift,
            scale_ratio,
        )
        right_ratio, right_rotation = self._wrist_kinematics(
            prev_signal,
            signal,
            ("right_wrist_x", "right_wrist_y"),
            ("right_elbow_x", "right_elbow_y"),
            body_shift,
            scale_ratio,
        )
        ankle_ratio = self._ankle_motion_ratio(prev_signal, signal, body_shift, scale_ratio)
        return left_ratio, right_ratio, left_rotation, right_rotation, ankle_ratio

    def _bridge_value(self, attr: str, frame_delta: int, decay: float) -> float | None:
        last_valid = self.last_valid_signal
        if last_valid is None:
            return None
        current = getattr(last_valid, attr)
        if current is None:
            return None
        previous = None if self.prev_valid_signal is None else getattr(self.prev_valid_signal, attr)
        if previous is None:
            estimate = current
        else:
            estimate = current + ((current - previous) * frame_delta * decay)
        if attr.endswith("_x") or attr.endswith("_y"):
            return self._clamp_unit(estimate)
        if attr in {
            "left_foot_y",
            "right_foot_y",
            "left_hip_x",
            "left_hip_y",
            "right_hip_x",
            "right_hip_y",
            "left_knee_x",
            "left_knee_y",
            "right_knee_x",
            "right_knee_y",
            "left_ankle_x",
            "left_ankle_y",
            "right_ankle_x",
            "right_ankle_y",
        }:
            return self._clamp_unit(estimate)
        if attr in {"leg_length", "shoulder_width"}:
            return self._clamp_positive(estimate)
        return estimate

    def _bridge_missing_signal(self, frame_idx: int, timestamp_sec: float) -> SignalFrame | None:
        if self.last_valid_signal is None:
            return None
        frame_delta = frame_idx - self.last_valid_signal.frame_idx
        if frame_delta <= 0 or frame_delta > self.config.max_pose_missing_bridge_frames:
            return None
        decay = self.config.pose_bridge_decay ** max(0, frame_delta - 1)
        bridged = SignalFrame(
            frame_idx=frame_idx,
            time_sec=timestamp_sec,
            detected=True,
            left_shoulder_x=self._bridge_value("left_shoulder_x", frame_delta, decay),
            left_shoulder_y=self._bridge_value("left_shoulder_y", frame_delta, decay),
            right_shoulder_x=self._bridge_value("right_shoulder_x", frame_delta, decay),
            right_shoulder_y=self._bridge_value("right_shoulder_y", frame_delta, decay),
            left_elbow_x=self._bridge_value("left_elbow_x", frame_delta, decay),
            left_elbow_y=self._bridge_value("left_elbow_y", frame_delta, decay),
            right_elbow_x=self._bridge_value("right_elbow_x", frame_delta, decay),
            right_elbow_y=self._bridge_value("right_elbow_y", frame_delta, decay),
            left_wrist_x=self._bridge_value("left_wrist_x", frame_delta, decay),
            left_wrist_y=self._bridge_value("left_wrist_y", frame_delta, decay),
            right_wrist_x=self._bridge_value("right_wrist_x", frame_delta, decay),
            right_wrist_y=self._bridge_value("right_wrist_y", frame_delta, decay),
            left_hip_x=self._bridge_value("left_hip_x", frame_delta, decay),
            left_ankle_x=self._bridge_value("left_ankle_x", frame_delta, decay),
            left_ankle_y=self._bridge_value("left_ankle_y", frame_delta, decay),
            right_hip_x=self._bridge_value("right_hip_x", frame_delta, decay),
            right_ankle_x=self._bridge_value("right_ankle_x", frame_delta, decay),
            right_ankle_y=self._bridge_value("right_ankle_y", frame_delta, decay),
            left_hip_y=self._bridge_value("left_hip_y", frame_delta, decay),
            right_hip_y=self._bridge_value("right_hip_y", frame_delta, decay),
            left_knee_x=self._bridge_value("left_knee_x", frame_delta, decay),
            left_knee_y=self._bridge_value("left_knee_y", frame_delta, decay),
            right_knee_x=self._bridge_value("right_knee_x", frame_delta, decay),
            right_knee_y=self._bridge_value("right_knee_y", frame_delta, decay),
            left_foot_y=self._bridge_value("left_foot_y", frame_delta, decay),
            right_foot_y=self._bridge_value("right_foot_y", frame_delta, decay),
            leg_length=self._bridge_value("leg_length", frame_delta, decay),
            shoulder_width=self._bridge_value("shoulder_width", frame_delta, decay),
        )
        if (
            bridged.left_hip_y is None
            or bridged.right_hip_y is None
            or bridged.left_foot_y is None
            or bridged.right_foot_y is None
            or bridged.leg_length is None
        ):
            return None
        return bridged

    def process_bgr_frame(self, frame, frame_idx: int, timestamp_sec: float):
        pose_frame = frame
        if self.config.pose_input_scale > 1.0:
            pose_frame = cv2.resize(
                frame,
                None,
                fx=self.config.pose_input_scale,
                fy=self.config.pose_input_scale,
                interpolation=cv2.INTER_LINEAR,
            )
        rgb = cv2.cvtColor(pose_frame, cv2.COLOR_BGR2RGB)
        result = self.pose.process(rgb)
        signal = pose_result_to_signal(result, frame_idx, timestamp_sec, self.config)
        raw_detected = signal.detected
        if not raw_detected:
            self.missing_bridge_streak += 1
            bridged_signal = self._bridge_missing_signal(frame_idx, timestamp_sec)
            if bridged_signal is not None:
                signal = bridged_signal
            else:
                self.prev_signal = None
                return signal, result
        else:
            self.missing_bridge_streak = 0
        left_flow_ratio, right_flow_ratio, left_rotation, right_rotation, ankle_ratio = self._measure_pose_kinematics(signal)
        signal.left_wrist_flow_ratio = left_flow_ratio
        signal.right_wrist_flow_ratio = right_flow_ratio
        signal.wrist_flow_ratio = max(left_flow_ratio, right_flow_ratio)
        signal.left_wrist_rotation_ratio = left_rotation
        signal.right_wrist_rotation_ratio = right_rotation
        signal.wrist_rotation_ratio = max(left_rotation, right_rotation)
        lower_rotation = min(left_rotation, right_rotation)
        upper_rotation = max(left_rotation, right_rotation)
        signal.wrist_sync_ratio = 0.0 if upper_rotation <= 1e-6 else lower_rotation / upper_rotation
        signal.ankle_flow_ratio = ankle_ratio
        self.prev_signal = signal if signal.detected else None
        if raw_detected:
            self.prev_valid_signal = self.last_valid_signal
            self.last_valid_signal = signal
        return signal, result


def extract_signal_stream(video_path: str | Path, config: EngineConfig) -> tuple[VideoMeta, list[SignalFrame]]:
    video_meta = probe_video(video_path)
    capture = cv2.VideoCapture(str(video_path))
    extractor = PoseSignalExtractor(config)

    signals: list[SignalFrame] = []
    frame_idx = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            timestamp_sec = frame_idx / video_meta.fps if video_meta.fps > 0 else 0.0
            signal, _ = extractor.process_bgr_frame(frame, frame_idx, timestamp_sec)
            signals.append(signal)
            frame_idx += 1
    finally:
        extractor.close()
        capture.release()

    return video_meta, signals


class _AirborneStateMachine:
    def __init__(self, config: EngineConfig):
        self.config = config
        self.floor_y: float | None = None
        self.prev_hip: float | None = None
        self.prev_foot: float | None = None
        self.prev_hip_fast: float | None = None
        self.prev_foot_fast: float | None = None
        self.hip_baseline: float | None = None
        self.foot_baseline: float | None = None
        self.interval_history: deque[int] = deque(maxlen=5)
        self.last_takeoff_frame: int | None = None
        self.contact_gate = False
        self.in_air = False
        self.release_streak = 0
        self.contact_streak = 0
        self.airborne_frames = 0
        self.airborne_min_foot_motion: float | None = None
        self.airborne_min_hip_motion: float | None = None
        self.current_jump_height_ratio = 0.0
        self.current_hip_lift_ratio = 0.0

    @staticmethod
    def _ema(alpha: float, value: float, previous: float | None) -> float:
        if previous is None:
            return value
        return alpha * value + (1.0 - alpha) * previous

    def _update_filtered_signal(self, signal: SignalFrame) -> tuple[float, float, float, bool]:
        mean_hip_y_raw, mean_foot_y_raw, leg_length = _mean_pose_values(signal)
        mean_hip_y_std = self._ema(self.config.ema_alpha_hip, mean_hip_y_raw, self.prev_hip)
        mean_foot_y_std = self._ema(self.config.ema_alpha_foot, mean_foot_y_raw, self.prev_foot)
        mean_hip_y_fast = self._ema(self.config.fast_ema_alpha_hip, mean_hip_y_raw, self.prev_hip_fast)
        mean_foot_y_fast = self._ema(self.config.fast_ema_alpha_foot, mean_foot_y_raw, self.prev_foot_fast)
        fast_mode = len(self.interval_history) >= 3 and median(self.interval_history) <= self.config.fast_mode_cadence_threshold
        mean_hip_y = mean_hip_y_fast if fast_mode else mean_hip_y_std
        mean_foot_y = mean_foot_y_fast if fast_mode else mean_foot_y_std

        self.prev_hip = mean_hip_y_std
        self.prev_foot = mean_foot_y_std
        self.prev_hip_fast = mean_hip_y_fast
        self.prev_foot_fast = mean_foot_y_fast

        self.hip_baseline = self._ema(self.config.baseline_alpha_hip, mean_hip_y, self.hip_baseline)
        self.foot_baseline = self._ema(self.config.baseline_alpha_foot, mean_foot_y, self.foot_baseline)
        hip_motion = mean_hip_y - self.hip_baseline
        foot_motion = mean_foot_y - self.foot_baseline
        if self.floor_y is None:
            self.floor_y = foot_motion
        else:
            self.floor_y = max(foot_motion, self.floor_y - self.config.floor_decay_ratio * leg_length)
        return foot_motion, hip_motion, leg_length, fast_mode

    def advance(self, signal: SignalFrame) -> None:
        if not signal.detected:
            self.contact_gate = False
            return
        assert signal.left_foot_y is not None
        assert signal.right_foot_y is not None
        foot_motion, hip_motion, leg_length, _ = self._update_filtered_signal(signal)

        contact_threshold = self.floor_y - (self.config.contact_margin_ratio * leg_length)
        symmetry_y = abs(signal.left_foot_y - signal.right_foot_y)
        self.contact_gate = foot_motion >= contact_threshold and symmetry_y <= (self.config.symmetry_y_ratio * leg_length)
        release_height_ratio = max(0.0, (self.floor_y - foot_motion) / leg_length)
        hip_lift_ratio = max(0.0, -hip_motion / leg_length)
        takeoff_ready = (
            release_height_ratio >= self.config.takeoff_height_ratio
            or hip_lift_ratio >= self.config.takeoff_hip_ratio
        )

        if self.contact_gate:
            self.contact_streak += 1
            self.release_streak = 0
        else:
            self.release_streak += 1
            self.contact_streak = 0

        if not self.in_air:
            self.current_jump_height_ratio = 0.0
            self.current_hip_lift_ratio = 0.0
            if self.release_streak >= 1 and takeoff_ready:
                self.in_air = True
                self.airborne_frames = self.release_streak
                self.airborne_min_foot_motion = foot_motion
                self.airborne_min_hip_motion = hip_motion
                self.current_jump_height_ratio = release_height_ratio
                self.current_hip_lift_ratio = hip_lift_ratio
            return

        self.airborne_frames += 1
        self.airborne_min_foot_motion = (
            foot_motion if self.airborne_min_foot_motion is None else min(self.airborne_min_foot_motion, foot_motion)
        )
        self.airborne_min_hip_motion = (
            hip_motion if self.airborne_min_hip_motion is None else min(self.airborne_min_hip_motion, hip_motion)
        )
        self.current_jump_height_ratio = max(0.0, (self.floor_y - self.airborne_min_foot_motion) / leg_length)
        self.current_hip_lift_ratio = max(0.0, -self.airborne_min_hip_motion / leg_length)

        if self.contact_gate and self.contact_streak >= self.config.landing_contact_frames:
            self.in_air = False
            if self.last_takeoff_frame is not None and signal.frame_idx > self.last_takeoff_frame:
                self.interval_history.append(signal.frame_idx - self.last_takeoff_frame)
            self.last_takeoff_frame = signal.frame_idx
            self.airborne_frames = 0
            self.airborne_min_foot_motion = None
            self.airborne_min_hip_motion = None
            self.current_jump_height_ratio = 0.0
            self.current_hip_lift_ratio = 0.0

    def arm_for_counting(self) -> None:
        self.contact_gate = False
        self.in_air = False
        self.release_streak = 0
        self.contact_streak = 0
        self.airborne_frames = 0
        self.airborne_min_foot_motion = None
        self.airborne_min_hip_motion = None
        self.current_jump_height_ratio = 0.0
        self.current_hip_lift_ratio = 0.0
        self.last_takeoff_frame = None
        self.interval_history.clear()


class RealtimeCounterEngine:
    def __init__(self, config: EngineConfig, enable_realtime_compensation: bool = False):
        self.config = config
        self.enable_realtime_compensation = enable_realtime_compensation
        self.classifier = CycleClassifier(
            target_frames=config.cycle_feature_frames,
            model_path=config.classifier_model_path,
        )
        self.state_engine = _AirborneStateMachine(config)
        self.hip_history: deque[float] = deque(maxlen=config.motion_window_frames)
        self.foot_history: deque[float] = deque(maxlen=config.motion_window_frames)
        self.leg_history: deque[float] = deque(maxlen=config.motion_window_frames)
        self.recent_hip_history: deque[float] = deque(maxlen=config.recent_window_frames)
        self.accepted_interval_history: deque[int] = deque(maxlen=config.adaptive_gap_history)
        self.accepted_hip_history: deque[float] = deque(maxlen=config.adaptive_gap_history)
        self.accepted_foot_history: deque[float] = deque(maxlen=config.adaptive_gap_history)
        self.last_accepted_frame: int | None = None
        self.accepted_frame_history: deque[int] = deque()
        self.accepted_running_count = 0
        self.wrist_flow_baseline_ratio = 0.0
        self.wrist_flow_smoothed_ratio = 0.0
        self.contact_wrist_flow_history: deque[float] = deque(maxlen=config.wrist_contact_history_size)
        self.ankle_flow_baseline_ratio = 0.0
        self.ankle_flow_smoothed_ratio = 0.0
        self.contact_ankle_flow_history: deque[float] = deque(maxlen=config.wrist_contact_history_size)
        self.airborne_flow_peak_ratio = 0.0
        self.airborne_flow_sum_ratio = 0.0
        self.airborne_flow_active_frames = 0
        self.airborne_frame_count = 0
        self.current_cycle_frames: list[SignalFrame] = []
        self.jump_cycle: JumpCycleEvidence | None = None
        self.last_cycle_prediction: CyclePrediction | None = None
        self.last_decision: CounterDecision | None = None
        self.monitor = MonitorState()
        self.pending_compensation: _PendingCountCompensation | None = None
        self.current_attempt_flow_ratio = 0.0

    def _update_motion_history(self, signal: SignalFrame) -> None:
        if not signal.detected:
            return
        mean_hip, mean_foot, leg_length = _mean_pose_values(signal)
        self.hip_history.append(mean_hip)
        self.foot_history.append(mean_foot)
        self.leg_history.append(leg_length)
        self.recent_hip_history.append(mean_hip)

    def motion_metrics(self) -> dict[str, float]:
        if not self.hip_history or not self.foot_history or not self.leg_history:
            return {
                "hip_range_ratio": 0.0,
                "foot_range_ratio": 0.0,
                "recent_hip_range_ratio": 0.0,
                "foot_to_hip_ratio": 0.0,
            }
        leg_median = median(self.leg_history)
        hip_range_ratio = (max(self.hip_history) - min(self.hip_history)) / leg_median
        foot_range_ratio = (max(self.foot_history) - min(self.foot_history)) / leg_median
        foot_to_hip_ratio = foot_range_ratio / max(hip_range_ratio, 1e-6)
        recent_hip_range_ratio = 0.0
        if len(self.recent_hip_history) >= self.config.recent_window_frames:
            recent_hip_range_ratio = (max(self.recent_hip_history) - min(self.recent_hip_history)) / leg_median
        return {
            "hip_range_ratio": hip_range_ratio,
            "foot_range_ratio": foot_range_ratio,
            "recent_hip_range_ratio": recent_hip_range_ratio,
            "foot_to_hip_ratio": foot_to_hip_ratio,
        }

    def _cadence_locked(self) -> bool:
        if not self.config.adaptive_gap_enabled:
            return False
        if len(self.accepted_interval_history) < self.config.adaptive_gap_min_intervals:
            return False
        if not self.accepted_hip_history or not self.accepted_foot_history:
            return False
        return (
            median(self.accepted_hip_history) >= self.config.adaptive_motion_hip_ratio
            and median(self.accepted_foot_history) >= self.config.adaptive_motion_foot_ratio
        )

    def _effective_limits(self) -> tuple[int, float, bool]:
        min_gap_frames = self.config.min_count_gap_frames
        min_recent_hip_ratio = self.config.min_recent_hip_range_ratio
        cadence_locked = self._cadence_locked()
        if cadence_locked:
            raw_interval_median = median(self.accepted_interval_history)
            min_gap_frames = max(
                self.config.adaptive_gap_floor_frames,
                min(
                    self.config.min_count_gap_frames,
                    int(round(raw_interval_median * self.config.adaptive_gap_factor)),
                ),
            )
            if self.config.adaptive_recent_hip_enabled:
                min_recent_hip_ratio = max(
                    self.config.adaptive_recent_hip_floor,
                    self.config.min_recent_hip_range_ratio * 0.8,
                )
        return min_gap_frames, min_recent_hip_ratio, cadence_locked

    def _balanced_motion_override(self, metrics: dict[str, float], cadence_locked: bool) -> bool:
        if not cadence_locked:
            return False
        return (
            metrics["hip_range_ratio"] >= self.config.balanced_override_hip_range_ratio
            and metrics["foot_range_ratio"] >= self.config.balanced_override_foot_range_ratio
            and metrics["recent_hip_range_ratio"] >= self.config.balanced_override_recent_hip_ratio
            and self.config.balanced_override_min_ratio <= metrics["foot_to_hip_ratio"] <= self.config.balanced_override_max_ratio
        )

    def _extended_motion_override(self, metrics: dict[str, float], cadence_locked: bool) -> bool:
        if not cadence_locked:
            return False
        recent_to_hip_ratio = metrics["recent_hip_range_ratio"] / max(metrics["hip_range_ratio"], 1e-6)
        return (
            metrics["hip_range_ratio"] >= self.config.extended_override_hip_range_ratio
            and metrics["foot_range_ratio"] >= self.config.extended_override_foot_range_ratio
            and metrics["recent_hip_range_ratio"] >= self.config.extended_override_recent_hip_ratio
            and self.config.extended_override_min_ratio <= metrics["foot_to_hip_ratio"] <= self.config.extended_override_max_ratio
            and recent_to_hip_ratio >= self.config.extended_override_recent_to_hip_ratio
        )

    def _foot_floor_override(self, metrics: dict[str, float]) -> bool:
        return (
            metrics["hip_range_ratio"] >= self.config.foot_floor_override_hip_range_ratio
            and metrics["foot_range_ratio"] >= self.config.foot_floor_override_foot_range_ratio
            and metrics["recent_hip_range_ratio"] >= self.config.foot_floor_override_recent_hip_ratio
            and metrics["foot_to_hip_ratio"] <= self.config.foot_floor_override_max_ratio
        )

    def _stale_tail_reject(self, metrics: dict[str, float], cadence_locked: bool) -> bool:
        if not cadence_locked:
            return False
        if metrics["hip_range_ratio"] < self.config.stale_tail_guard_hip_range_ratio:
            return False
        recent_to_hip_ratio = metrics["recent_hip_range_ratio"] / max(metrics["hip_range_ratio"], 1e-6)
        return recent_to_hip_ratio < self.config.stale_tail_guard_recent_to_hip_ratio

    def _set_decision(
        self,
        signal: SignalFrame,
        accepted: bool,
        reason: str,
        airtime_frames: int,
        jump_height_ratio: float,
        hip_lift_ratio: float,
        wrist_flow_peak_ratio: float,
        wrist_flow_mean_ratio: float,
        wrist_flow_active_frames: int,
        current_wrist_flow_ratio: float,
        wrist_rotation_peak_ratio: float,
        wrist_rotation_mean_ratio: float,
        wrist_sync_peak_ratio: float,
        rope_pass_hints: int,
        classifier_label: str,
        classifier_confidence: float,
        classifier_source: str,
        min_gap_frames: int,
        cadence_locked: bool,
    ) -> None:
        self.last_decision = CounterDecision(
            frame_idx=signal.frame_idx,
            time_sec=signal.time_sec,
            accepted=accepted,
            reason=reason,
            airtime_frames=airtime_frames,
            jump_height_ratio=jump_height_ratio,
            hip_lift_ratio=hip_lift_ratio,
            wrist_flow_peak_ratio=wrist_flow_peak_ratio,
            wrist_flow_mean_ratio=wrist_flow_mean_ratio,
            wrist_flow_active_frames=wrist_flow_active_frames,
            current_wrist_flow_ratio=current_wrist_flow_ratio,
            wrist_flow_baseline_ratio=self.wrist_flow_baseline_ratio,
            wrist_rotation_peak_ratio=wrist_rotation_peak_ratio,
            wrist_rotation_mean_ratio=wrist_rotation_mean_ratio,
            wrist_sync_peak_ratio=wrist_sync_peak_ratio,
            rope_pass_hints=rope_pass_hints,
            classifier_label=classifier_label,
            classifier_confidence=classifier_confidence,
            classifier_source=classifier_source,
            min_gap_frames=min_gap_frames,
            cadence_locked=cadence_locked,
        )

    def _reset_airborne_flow(self) -> None:
        self.airborne_flow_peak_ratio = 0.0
        self.airborne_flow_sum_ratio = 0.0
        self.airborne_flow_active_frames = 0
        self.airborne_frame_count = 0
        self.current_cycle_frames = []
        self.jump_cycle = None

    def _update_wrist_flow(self, signal: SignalFrame, in_air: bool) -> float:
        raw_ratio = signal.wrist_flow_ratio
        alpha = self.config.wrist_flow_smoothing_alpha
        self.wrist_flow_smoothed_ratio = (
            raw_ratio
            if self.wrist_flow_smoothed_ratio <= 0.0
            else (alpha * raw_ratio) + ((1.0 - alpha) * self.wrist_flow_smoothed_ratio)
        )
        if not in_air:
            beta = self.config.wrist_flow_baseline_alpha
            ema_baseline = (
                self.wrist_flow_smoothed_ratio
                if self.wrist_flow_baseline_ratio <= 0.0
                else (beta * self.wrist_flow_smoothed_ratio) + ((1.0 - beta) * self.wrist_flow_baseline_ratio)
            )
            self.contact_wrist_flow_history.append(self.wrist_flow_smoothed_ratio)
            rolling_baseline = ema_baseline
            if self.contact_wrist_flow_history:
                q = max(0.0, min(0.95, self.config.wrist_baseline_quantile))
                rolling_baseline = float(np.quantile(np.asarray(self.contact_wrist_flow_history, dtype=np.float32), q))
            self.wrist_flow_baseline_ratio = (ema_baseline + rolling_baseline) / 2.0
        return max(0.0, self.wrist_flow_smoothed_ratio - self.wrist_flow_baseline_ratio)

    def _update_ankle_flow(self, signal: SignalFrame, in_air: bool) -> float:
        raw_ratio = signal.ankle_flow_ratio
        alpha = self.config.wrist_flow_smoothing_alpha
        self.ankle_flow_smoothed_ratio = (
            raw_ratio
            if self.ankle_flow_smoothed_ratio <= 0.0
            else (alpha * raw_ratio) + ((1.0 - alpha) * self.ankle_flow_smoothed_ratio)
        )
        if not in_air:
            beta = self.config.wrist_flow_baseline_alpha
            ema_baseline = (
                self.ankle_flow_smoothed_ratio
                if self.ankle_flow_baseline_ratio <= 0.0
                else (beta * self.ankle_flow_smoothed_ratio) + ((1.0 - beta) * self.ankle_flow_baseline_ratio)
            )
            self.contact_ankle_flow_history.append(self.ankle_flow_smoothed_ratio)
            rolling_baseline = ema_baseline
            if self.contact_ankle_flow_history:
                q = max(0.0, min(0.95, self.config.wrist_baseline_quantile))
                rolling_baseline = float(np.quantile(np.asarray(self.contact_ankle_flow_history, dtype=np.float32), q))
            self.ankle_flow_baseline_ratio = (ema_baseline + rolling_baseline) / 2.0
        return max(0.0, self.ankle_flow_smoothed_ratio - self.ankle_flow_baseline_ratio)

    def _start_jump_cycle(self, signal: SignalFrame) -> None:
        self.jump_cycle = JumpCycleEvidence(start_frame=signal.frame_idx)
        self.current_cycle_frames = [signal]

    def _append_cycle_frame(self, signal: SignalFrame) -> None:
        if self.jump_cycle is None:
            return
        if self.current_cycle_frames and self.current_cycle_frames[-1].frame_idx == signal.frame_idx:
            self.current_cycle_frames[-1] = signal
            return
        self.current_cycle_frames.append(signal)

    def _maybe_add_rope_pass_hint(
        self,
        signal: SignalFrame,
        wrist_flow_excess_ratio: float,
        ankle_flow_excess_ratio: float,
    ) -> None:
        if self.jump_cycle is None:
            return
        qualifies = (
            signal.wrist_sync_ratio >= self.config.wrist_sync_min_ratio
            and signal.wrist_rotation_ratio >= (self.config.wrist_rotation_peak_ratio * 0.75)
            and (
                wrist_flow_excess_ratio >= self.config.wrist_flow_active_ratio
                or ankle_flow_excess_ratio >= self.config.ankle_flow_active_ratio
            )
        )
        if not qualifies:
            self.jump_cycle.rotation_gate_active = False
            return
        if self.jump_cycle.rotation_gate_active:
            return
        separated = (
            not self.jump_cycle.rope_pass_frames
            or (signal.frame_idx - self.jump_cycle.rope_pass_frames[-1]) >= self.config.rope_pass_separation_frames
        )
        if not separated:
            return
        self.jump_cycle.rope_pass_hints += 1
        self.jump_cycle.rope_pass_frames.append(signal.frame_idx)
        self.jump_cycle.rotation_gate_active = True

    def _update_jump_cycle(
        self,
        signal: SignalFrame,
        wrist_flow_excess_ratio: float,
        ankle_flow_excess_ratio: float,
    ) -> None:
        if self.jump_cycle is None:
            return
        self.jump_cycle.airborne_frames += 1
        self.jump_cycle.max_jump_height_ratio = max(
            self.jump_cycle.max_jump_height_ratio,
            self.state_engine.current_jump_height_ratio,
        )
        self.jump_cycle.max_hip_lift_ratio = max(
            self.jump_cycle.max_hip_lift_ratio,
            self.state_engine.current_hip_lift_ratio,
        )
        self.jump_cycle.wrist_rotation_peak_ratio = max(
            self.jump_cycle.wrist_rotation_peak_ratio,
            signal.wrist_rotation_ratio,
        )
        self.jump_cycle.wrist_rotation_sum_ratio += signal.wrist_rotation_ratio
        self.jump_cycle.wrist_sync_peak_ratio = max(
            self.jump_cycle.wrist_sync_peak_ratio,
            signal.wrist_sync_ratio,
        )
        self.jump_cycle.ankle_flow_peak_ratio = max(
            self.jump_cycle.ankle_flow_peak_ratio,
            ankle_flow_excess_ratio,
        )
        self._maybe_add_rope_pass_hint(signal, wrist_flow_excess_ratio, ankle_flow_excess_ratio)

    def _finish_jump_cycle(self, signal: SignalFrame) -> JumpCycleEvidence | None:
        if self.jump_cycle is None:
            return None
        self.jump_cycle.end_frame = signal.frame_idx
        prediction = self.classifier.predict(self.current_cycle_frames)
        self.jump_cycle.prediction = prediction
        self.last_cycle_prediction = prediction
        return self.jump_cycle

    def _update_monitor(self, signal: SignalFrame, cadence_locked: bool) -> None:
        cycle_prediction = self.last_cycle_prediction
        self.monitor = MonitorState(
            frame_idx=signal.frame_idx,
            time_sec=signal.time_sec,
            detected=signal.detected,
            contact_gate=self.state_engine.contact_gate,
            in_air=self.state_engine.in_air,
            jump_height_ratio=self.state_engine.current_jump_height_ratio,
            hip_lift_ratio=self.state_engine.current_hip_lift_ratio,
            wrist_flow_ratio=self.wrist_flow_smoothed_ratio,
            wrist_flow_baseline_ratio=self.wrist_flow_baseline_ratio,
            wrist_flow_peak_ratio=self.airborne_flow_peak_ratio,
            wrist_flow_active_frames=self.airborne_flow_active_frames,
            wrist_rotation_ratio=signal.wrist_rotation_ratio,
            wrist_sync_ratio=signal.wrist_sync_ratio,
            ankle_flow_ratio=self.ankle_flow_smoothed_ratio,
            rope_pass_hints=0 if self.jump_cycle is None else self.jump_cycle.rope_pass_hints,
            cycle_label="-" if cycle_prediction is None else cycle_prediction.label,
            cycle_confidence=0.0 if cycle_prediction is None else cycle_prediction.confidence,
            cycle_source="-" if cycle_prediction is None else cycle_prediction.source,
            cadence_locked=cadence_locked,
        )

    def _clear_pending_compensation(self) -> None:
        self.pending_compensation = None

    def _start_pending_compensation(self, candidate: CounterEvent) -> None:
        if not self.enable_realtime_compensation:
            return
        self.pending_compensation = _PendingCountCompensation(
            frame_idx=candidate.frame_idx,
            time_sec=candidate.time_sec,
        )

    def _emit_compensation(self, signal: SignalFrame) -> CounterEvent | None:
        if self.accepted_running_count <= 0:
            self._clear_pending_compensation()
            return None
        had_previous_interval = len(self.accepted_frame_history) > 1
        self.accepted_running_count -= 1
        if self.accepted_frame_history:
            self.accepted_frame_history.pop()
        if self.accepted_hip_history:
            self.accepted_hip_history.pop()
        if self.accepted_foot_history:
            self.accepted_foot_history.pop()
        if had_previous_interval and self.accepted_interval_history:
            self.accepted_interval_history.pop()
        self.last_accepted_frame = None if not self.accepted_frame_history else self.accepted_frame_history[-1]
        self._clear_pending_compensation()
        return CounterEvent(
            frame_idx=signal.frame_idx,
            time_sec=signal.time_sec,
            running_count=self.accepted_running_count,
            count_delta=-1,
        )

    def _maybe_compensate_recent_count(self, signal: SignalFrame) -> CounterEvent | None:
        pending = self.pending_compensation
        if not self.enable_realtime_compensation or pending is None:
            return None
        elapsed_frames = signal.frame_idx - pending.frame_idx
        if elapsed_frames <= 0:
            return None
        if elapsed_frames > self.config.rope_stuck_window_frames:
            self._clear_pending_compensation()
            return None
        pending.observed_frames += 1
        if self.state_engine.contact_gate:
            pending.contact_frames += 1
        if self.state_engine.in_air:
            pending.airborne_seen = True
        pending.max_attempt_flow_ratio = max(
            pending.max_attempt_flow_ratio,
            self.current_attempt_flow_ratio,
        )
        if pending.airborne_seen:
            self._clear_pending_compensation()
            return None
        if pending.observed_frames < self.config.rope_stuck_min_hold_frames:
            return None
        contact_ratio = pending.contact_frames / max(1, pending.observed_frames)
        if (
            contact_ratio >= self.config.rope_stuck_contact_ratio
            and pending.max_attempt_flow_ratio >= self.config.rope_stuck_attempt_flow_ratio
        ):
            return self._emit_compensation(signal)
        return None

    def warmup(self, signal: SignalFrame) -> None:
        self.last_decision = None
        self._step_internal(signal, allow_count=False)
        self._clear_pending_compensation()

    def step(self, signal: SignalFrame) -> CounterEvent | None:
        self.last_decision = None
        event = self._step_internal(signal, allow_count=True)
        if event is not None:
            self._clear_pending_compensation()
            self._start_pending_compensation(event)
            return event
        return self._maybe_compensate_recent_count(signal)

    def arm_for_counting(self) -> None:
        self.state_engine.arm_for_counting()
        self.last_accepted_frame = None
        self.accepted_frame_history.clear()
        self.accepted_running_count = 0
        self.accepted_interval_history.clear()
        self.accepted_hip_history.clear()
        self.accepted_foot_history.clear()
        self.last_cycle_prediction = None
        self._reset_airborne_flow()
        self._clear_pending_compensation()

    def _step_internal(self, signal: SignalFrame, allow_count: bool) -> CounterEvent | None:
        if not signal.detected:
            self.monitor = MonitorState(frame_idx=signal.frame_idx, time_sec=signal.time_sec, detected=False)
            self._reset_airborne_flow()
            self.current_attempt_flow_ratio = 0.0
            return None

        self._update_motion_history(signal)
        metrics = self.motion_metrics()
        min_gap_frames, min_recent_hip_ratio, cadence_locked = self._effective_limits()

        was_in_air = self.state_engine.in_air
        self.state_engine.advance(signal)
        is_in_air = self.state_engine.in_air

        flow_excess_ratio = self._update_wrist_flow(signal, is_in_air)
        ankle_flow_excess_ratio = self._update_ankle_flow(signal, is_in_air)
        self.current_attempt_flow_ratio = max(flow_excess_ratio, ankle_flow_excess_ratio)
        completed_cycle: JumpCycleEvidence | None = None
        if not was_in_air and is_in_air:
            self._reset_airborne_flow()
            self._start_jump_cycle(signal)
        if is_in_air or (was_in_air and not is_in_air):
            self._append_cycle_frame(signal)
        if is_in_air:
            self.airborne_frame_count += 1
            self.airborne_flow_sum_ratio += flow_excess_ratio
            self.airborne_flow_peak_ratio = max(self.airborne_flow_peak_ratio, flow_excess_ratio)
            if flow_excess_ratio >= self.config.wrist_flow_active_ratio:
                self.airborne_flow_active_frames += 1
            self._update_jump_cycle(signal, flow_excess_ratio, ankle_flow_excess_ratio)
        elif was_in_air and not is_in_air:
            completed_cycle = self._finish_jump_cycle(signal)

        self._update_monitor(signal, cadence_locked)
        if not allow_count:
            if completed_cycle is not None:
                self._reset_airborne_flow()
            return None
        if completed_cycle is None:
            return None

        airtime_frames = completed_cycle.airborne_frames
        jump_height_ratio = completed_cycle.max_jump_height_ratio
        hip_lift_ratio = completed_cycle.max_hip_lift_ratio
        wrist_flow_peak_ratio = self.airborne_flow_peak_ratio
        wrist_flow_mean_ratio = self.airborne_flow_sum_ratio / max(1, self.airborne_frame_count)
        wrist_rotation_peak_ratio = completed_cycle.wrist_rotation_peak_ratio
        wrist_rotation_mean_ratio = completed_cycle.wrist_rotation_sum_ratio / max(1, completed_cycle.airborne_frames)
        wrist_sync_peak_ratio = completed_cycle.wrist_sync_peak_ratio
        rope_pass_hints = completed_cycle.rope_pass_hints
        prediction = completed_cycle.prediction
        classifier_label = "unknown" if prediction is None else prediction.label
        classifier_confidence = 0.0 if prediction is None else prediction.confidence
        classifier_source = "-" if prediction is None else prediction.source

        reject_reason: str | None = None
        if len(self.hip_history) < max(3, self.config.motion_window_frames // 2):
            reject_reason = "insufficient_window"
        elif self.last_accepted_frame is not None and (signal.frame_idx - self.last_accepted_frame) < min_gap_frames:
            reject_reason = "min_gap"
        elif airtime_frames < self.config.min_airborne_frames:
            reject_reason = "airtime_short"
        elif airtime_frames > self.config.max_airborne_frames:
            reject_reason = "airtime_long"
        elif jump_height_ratio < self.config.min_jump_height_ratio:
            reject_reason = "jump_height"
        elif self.config.min_hip_lift_ratio > 0.0 and hip_lift_ratio < self.config.min_hip_lift_ratio:
            reject_reason = "hip_lift"
        elif metrics["hip_range_ratio"] < self.config.min_hip_range_ratio:
            reject_reason = "hip_range"
        elif (
            metrics["foot_range_ratio"] < self.config.min_foot_range_ratio
            and not self._foot_floor_override(metrics)
        ):
            reject_reason = "foot_range"
        elif (
            metrics["recent_hip_range_ratio"] < min_recent_hip_ratio
            and not self._balanced_motion_override(metrics, cadence_locked)
            and not self._extended_motion_override(metrics, cadence_locked)
        ):
            reject_reason = "recent_hip_range"
        elif metrics["foot_to_hip_ratio"] > self.config.max_foot_to_hip_ratio:
            reject_reason = "foot_to_hip_ratio"
        elif (
            metrics["hip_range_ratio"] < self.config.guard_low_hip_range_ratio
            and metrics["foot_range_ratio"] >= self.config.guard_high_foot_range_ratio
            and metrics["recent_hip_range_ratio"] < self.config.guard_recent_hip_range_ratio
        ):
            reject_reason = "foot_dominant_low_hip"
        elif self._stale_tail_reject(metrics, cadence_locked):
            reject_reason = "stale_tail"
        elif classifier_label != "double_under":
            reject_reason = f"classifier_{classifier_label}"
        elif classifier_confidence < self.config.classifier_confidence_threshold:
            reject_reason = "classifier_confidence"

        if reject_reason is not None:
            self._set_decision(
                signal,
                False,
                reject_reason,
                airtime_frames,
                jump_height_ratio,
                hip_lift_ratio,
                wrist_flow_peak_ratio,
                wrist_flow_mean_ratio,
                self.airborne_flow_active_frames,
                flow_excess_ratio,
                wrist_rotation_peak_ratio,
                wrist_rotation_mean_ratio,
                wrist_sync_peak_ratio,
                rope_pass_hints,
                classifier_label,
                classifier_confidence,
                classifier_source,
                min_gap_frames,
                cadence_locked,
            )
            self._reset_airborne_flow()
            return None

        if self.last_accepted_frame is not None and signal.frame_idx > self.last_accepted_frame:
            self.accepted_interval_history.append(signal.frame_idx - self.last_accepted_frame)
        self.last_accepted_frame = signal.frame_idx
        self.accepted_frame_history.append(signal.frame_idx)
        self.accepted_running_count += 1
        self.accepted_hip_history.append(metrics["hip_range_ratio"])
        self.accepted_foot_history.append(metrics["foot_range_ratio"])
        completed_cycle.counted = True
        self._set_decision(
            signal,
            True,
            "accepted",
            airtime_frames,
            jump_height_ratio,
            hip_lift_ratio,
            wrist_flow_peak_ratio,
            wrist_flow_mean_ratio,
            self.airborne_flow_active_frames,
            flow_excess_ratio,
            wrist_rotation_peak_ratio,
            wrist_rotation_mean_ratio,
            wrist_sync_peak_ratio,
            rope_pass_hints,
            classifier_label,
            classifier_confidence,
            classifier_source,
            min_gap_frames,
            cadence_locked,
        )
        self._reset_airborne_flow()
        return CounterEvent(
            frame_idx=signal.frame_idx,
            time_sec=signal.time_sec,
            running_count=self.accepted_running_count,
            count_delta=1,
        )


class RealtimeStartGate:
    def __init__(
        self,
        ready_hold_seconds: float = 1.0,
        countdown_seconds: float = 3.0,
        ready_dropout_seconds: float = 0.35,
    ):
        self.ready_hold_seconds = ready_hold_seconds
        self.countdown_seconds = countdown_seconds
        self.ready_dropout_seconds = ready_dropout_seconds
        self.phase = "SEARCHING"
        self.ready_since_sec: float | None = None
        self.last_ready_sec: float | None = None
        self.countdown_end_sec: float | None = None
        self.count_started_at_sec: float | None = None

    def update(self, full_body_ready: bool, timestamp_sec: float) -> StreamState:
        if full_body_ready:
            self.last_ready_sec = timestamp_sec

        ready_active = full_body_ready or (
            self.last_ready_sec is not None
            and (timestamp_sec - self.last_ready_sec) <= self.ready_dropout_seconds
        )

        if self.phase == "SEARCHING":
            if ready_active:
                if self.ready_since_sec is None:
                    self.ready_since_sec = timestamp_sec
                if (timestamp_sec - self.ready_since_sec) >= self.ready_hold_seconds:
                    self.phase = "COUNTDOWN"
                    self.countdown_end_sec = timestamp_sec + self.countdown_seconds
            else:
                self.ready_since_sec = None
        elif self.phase == "COUNTDOWN":
            if not ready_active:
                self.phase = "SEARCHING"
                self.ready_since_sec = None
                self.countdown_end_sec = None
            elif self.countdown_end_sec is not None and timestamp_sec >= self.countdown_end_sec:
                self.phase = "COUNTING"
                self.count_started_at_sec = timestamp_sec
        return self.snapshot(timestamp_sec)

    def snapshot(self, timestamp_sec: float) -> StreamState:
        ready_progress = 0.0
        if self.phase == "SEARCHING" and self.ready_since_sec is not None:
            ready_progress = min(1.0, (timestamp_sec - self.ready_since_sec) / self.ready_hold_seconds)

        countdown_remaining_sec = 0.0
        if self.phase == "COUNTDOWN" and self.countdown_end_sec is not None:
            countdown_remaining_sec = max(0.0, self.countdown_end_sec - timestamp_sec)

        return StreamState(
            phase=self.phase,
            ready_progress=ready_progress,
            countdown_remaining_sec=countdown_remaining_sec,
            count_started_at_sec=self.count_started_at_sec,
        )

    def reset(self) -> None:
        self.phase = "SEARCHING"
        self.ready_since_sec = None
        self.last_ready_sec = None
        self.countdown_end_sec = None
        self.count_started_at_sec = None


def build_label_window(
    ground_truth_events: list[LabelEvent],
    window_config: LabelWindowConfig | None = None,
) -> tuple[int, int]:
    if not ground_truth_events:
        return 0, -1
    if window_config is None:
        return 0, ground_truth_events[-1].frame_idx
    start_frame = max(0, ground_truth_events[0].frame_idx + window_config.start_offset_frames)
    end_frame = max(start_frame, ground_truth_events[-1].frame_idx + window_config.end_offset_frames)
    return start_frame, end_frame


def run_counter_on_signals(
    signals: list[SignalFrame],
    config: EngineConfig,
    start_frame: int = 0,
    end_frame: int | None = None,
    warmup_frames: int = 0,
) -> list[CounterEvent]:
    engine = RealtimeCounterEngine(config)
    events: list[CounterEvent] = []
    effective_end_frame = signals[-1].frame_idx if signals and end_frame is None else end_frame
    if effective_end_frame is None:
        return events

    start_index = max(0, start_frame - max(warmup_frames, config.motion_window_frames))
    counting_armed = False
    for signal in signals[start_index:]:
        if signal.frame_idx > effective_end_frame:
            break
        if signal.frame_idx < start_frame:
            engine.warmup(signal)
            continue
        if not counting_armed:
            engine.arm_for_counting()
            counting_armed = True
        event = engine.step(signal)
        if event is not None:
            events.append(event)
    return events


def run_dataset(
    signal_cache: dict[str, tuple[VideoMeta, list[SignalFrame]]],
    ground_truth: dict[str, list[LabelEvent]],
    config: EngineConfig,
    window_config: LabelWindowConfig | None = None,
) -> list[VideoResult]:
    results: list[VideoResult] = []
    for stem in sorted(signal_cache):
        _, signals = signal_cache[stem]
        eval_start_frame, eval_end_frame = build_label_window(ground_truth[stem], window_config)
        warmup_frames = 0 if window_config is None else window_config.warmup_frames
        events = run_counter_on_signals(
            signals,
            config,
            start_frame=eval_start_frame,
            end_frame=eval_end_frame,
            warmup_frames=warmup_frames,
        )
        gt_count = len(ground_truth[stem])
        predicted_count = sum(event.count_delta for event in events)
        results.append(
            VideoResult(
                stem=stem,
                gt_count=gt_count,
                predicted_count=predicted_count,
                count_error=predicted_count - gt_count,
                exact_match=(predicted_count == gt_count),
                eval_start_frame=eval_start_frame,
                eval_end_frame=eval_end_frame,
                predicted_frames=[event.frame_idx for event in events],
            )
        )
    return results


def summarize_results(results: list[VideoResult]) -> dict[str, object]:
    exact_matches = sum(result.exact_match for result in results)
    total_gt_count = sum(result.gt_count for result in results)
    total_predicted_count = sum(result.predicted_count for result in results)
    total_abs_error = sum(abs(result.count_error) for result in results)
    return {
        "overall_count_accuracy": (1.0 - (total_abs_error / total_gt_count)) if total_gt_count else 0.0,
        "exact_video_count_accuracy": exact_matches / len(results) if results else 0.0,
        "videos": [asdict(result) for result in results],
        "total_gt_count": total_gt_count,
        "total_predicted_count": total_predicted_count,
        "signed_total_error": total_predicted_count - total_gt_count,
        "total_abs_error": total_abs_error,
    }


def default_search_configs(limit: int | None = None) -> list[EngineConfig]:
    configs = [
        EngineConfig(
            takeoff_height_ratio=values[0],
            min_jump_height_ratio=values[1],
            min_count_gap_frames=values[2],
            wrist_flow_peak_ratio=values[3],
            wrist_flow_mean_ratio=values[4],
            wrist_flow_active_ratio=values[5],
        )
        for values in product(
            [0.008, 0.010, 0.012],
            [0.016, 0.018, 0.020],
            [6, 7, 8],
            [0.008, 0.010, 0.012],
            [0.003, 0.004, 0.006],
            [0.003, 0.004, 0.006],
        )
    ]
    if limit is not None:
        return configs[:limit]
    return configs


def search_best_config(
    signal_cache: dict[str, tuple[VideoMeta, list[SignalFrame]]],
    ground_truth: dict[str, list[LabelEvent]],
    window_config: LabelWindowConfig | None = None,
    limit: int | None = None,
) -> tuple[EngineConfig, dict[str, object]]:
    best_config = EngineConfig()
    best_summary = summarize_results(run_dataset(signal_cache, ground_truth, best_config, window_config))
    for config in default_search_configs(limit):
        summary = summarize_results(run_dataset(signal_cache, ground_truth, config, window_config))
        if (
            summary["overall_count_accuracy"] > best_summary["overall_count_accuracy"]
            or (
                summary["overall_count_accuracy"] == best_summary["overall_count_accuracy"]
                and summary["total_abs_error"] < best_summary["total_abs_error"]
            )
            or (
                summary["overall_count_accuracy"] == best_summary["overall_count_accuracy"]
                and summary["total_abs_error"] == best_summary["total_abs_error"]
                and summary["exact_video_count_accuracy"] > best_summary["exact_video_count_accuracy"]
            )
        ):
            best_config = config
            best_summary = summary
    return best_config, best_summary


def save_summary(
    path: str | Path,
    config: EngineConfig,
    summary: dict[str, object],
    window_config: LabelWindowConfig | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": config.to_dict(),
        "label_window": None if window_config is None else window_config.to_dict(),
        "summary": summary,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
