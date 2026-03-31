from __future__ import annotations

import json
import math
import xml.etree.ElementTree as ET
from collections import deque
from dataclasses import asdict, dataclass
from itertools import product
from pathlib import Path
from statistics import median

import cv2
import mediapipe as mp
import numpy as np


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
)


@dataclass(frozen=True)
class EngineConfig:
    hip_visibility_threshold: float = 0.50
    foot_visibility_threshold: float = 0.35
    arm_visibility_threshold: float = 0.18
    wrist_visibility_threshold: float = 0.10
    foot_cutoff_hz: float = 5.0
    hip_cutoff_hz: float = 4.0
    wrist_speed_cutoff_hz: float = 7.0
    ground_baseline_alpha: float = 0.12
    floor_decay_ratio: float = 0.004
    contact_margin_ratio: float = 0.10
    symmetry_y_ratio: float = 0.18
    takeoff_height_ratio: float = 0.016
    takeoff_hip_ratio: float = 0.010
    min_count_gap_frames: int = 6
    min_airborne_frames: int = 2
    max_airborne_frames: int = 28
    min_jump_height_ratio: float = 0.018
    min_hip_lift_ratio: float = 0.010
    min_wrist_peak_count: int = 1
    min_wrist_peak_speed_ratio: float = 0.06
    min_wrist_energy_ratio: float = 0.035
    wrist_peak_refractory_frames: int = 2
    fft_window_frames: int = 36
    min_fft_peak_hz: float = 2.0
    min_fft_power_ratio: float = 0.08
    min_fft_wrist_to_jump_ratio: float = 0.60
    adaptive_gap_enabled: bool = True
    adaptive_gap_factor: float = 0.68
    adaptive_gap_history: int = 6
    adaptive_gap_min_intervals: int = 2
    adaptive_gap_floor_frames: int = 4
    search_wrist_peak_count: int = 1
    min_wrist_rotation_count: float = 0.35
    max_wrist_rotation_count: float = 5.00
    target_wrist_rotation_count: float = 1.50
    rotation_count_tolerance: float = 1.20
    min_wrist_rotation_balance: float = 0.18
    wrist_period_history_size: int = 12
    accepted_profile_history_size: int = 6
    cadence_similarity_min_ratio: float = 0.25
    ema_alpha_hip: float = 0.45
    ema_alpha_foot: float = 0.55
    fast_ema_alpha_hip: float = 0.25
    fast_ema_alpha_foot: float = 0.35
    baseline_alpha_hip: float = 0.02
    baseline_alpha_foot: float = 0.04
    descend_velocity_ratio: float = 0.006
    ascend_velocity_ratio: float = 0.004
    fast_descend_velocity_ratio: float = 0.002
    fast_ascend_velocity_ratio: float = 0.001
    min_refractory_frames: int = 4
    fast_min_refractory_frames: int = 2
    fast_mode_cadence_threshold: int = 7
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
    adaptive_motion_hip_ratio: float = 0.05
    adaptive_motion_foot_ratio: float = 0.04
    adaptive_recent_hip_enabled: bool = True
    adaptive_recent_hip_floor: float = 0.022
    contact_release_frames: int = 2
    landing_contact_frames: int = 2
    min_accept_score: int = 2
    strong_jump_score: int = 2
    wrist_cadence_min_hz: float = 2.2
    wrist_confidence_floor: float = 0.15

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
    left_hip_y: float | None = None
    right_hip_y: float | None = None
    left_foot_y: float | None = None
    right_foot_y: float | None = None
    leg_length: float | None = None


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
    wrist_peak_count: int
    wrist_peak_speed_ratio: float
    wrist_energy_ratio: float
    wrist_fft_peak_hz: float
    wrist_fft_power_ratio: float
    wrist_to_jump_ratio: float
    wrist_rotation_count: float
    wrist_rotation_balance: float
    wrist_rotation_cadence_hz: float
    wrist_period_frames: float
    wrist_period_confidence: float
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
    wrist_speed_ratio: float = 0.0
    wrist_peak_count: int = 0
    wrist_rotation_count: float = 0.0
    wrist_rotation_balance: float = 0.0
    wrist_period_frames: float = 0.0
    wrist_period_sec: float = 0.0
    wrist_period_confidence: float = 0.0
    wrist_cadence_hz: float = 0.0
    cadence_profile_ratio: float = 0.0


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
    if (
        float(left_hip.visibility) < config.hip_visibility_threshold
        or float(right_hip.visibility) < config.hip_visibility_threshold
    ):
        return SignalFrame(frame_idx=frame_idx, time_sec=timestamp_sec, detected=False)

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
        left_hip_y=float(left_hip.y),
        right_hip_y=float(right_hip.y),
        left_foot_y=left_foot_y,
        right_foot_y=right_foot_y,
        leg_length=leg_length,
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
            smooth_landmarks=False,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def close(self) -> None:
        self.pose.close()

    def process_bgr_frame(self, frame, frame_idx: int, timestamp_sec: float):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.pose.process(rgb)
        signal = pose_result_to_signal(result, frame_idx, timestamp_sec, self.config)
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


class _ButterworthLowPass:
    def __init__(self, cutoff_hz: float):
        self.cutoff_hz = cutoff_hz
        self.z1 = 0.0
        self.z2 = 0.0
        self.initialized = False
        self.last_fs: float | None = None
        self.coeffs: tuple[float, float, float, float, float] | None = None

    def _compute_coeffs(self, fs: float) -> tuple[float, float, float, float, float]:
        fs = max(fs, self.cutoff_hz * 3.0, 1.0)
        omega = 2.0 * math.pi * min(self.cutoff_hz / fs, 0.49)
        sin_omega = math.sin(omega)
        cos_omega = math.cos(omega)
        q = 1.0 / math.sqrt(2.0)
        alpha = sin_omega / (2.0 * q)
        b0 = (1.0 - cos_omega) / 2.0
        b1 = 1.0 - cos_omega
        b2 = (1.0 - cos_omega) / 2.0
        a0 = 1.0 + alpha
        a1 = -2.0 * cos_omega
        a2 = 1.0 - alpha
        return b0 / a0, b1 / a0, b2 / a0, a1 / a0, a2 / a0

    def filter(self, value: float, fs: float) -> float:
        if not self.initialized:
            self.z1 = value
            self.z2 = value
            self.initialized = True
            self.last_fs = fs
            self.coeffs = self._compute_coeffs(fs)
            return value
        if self.coeffs is None or self.last_fs is None or abs(fs - self.last_fs) > 0.5:
            self.coeffs = self._compute_coeffs(fs)
            self.last_fs = fs
        b0, b1, b2, a1, a2 = self.coeffs
        out = (b0 * value) + self.z1
        self.z1 = (b1 * value) - (a1 * out) + self.z2
        self.z2 = (b2 * value) - (a2 * out)
        return out


def _wrap_angle_delta(current: float, previous: float) -> float:
    delta = current - previous
    while delta > math.pi:
        delta -= 2.0 * math.pi
    while delta < -math.pi:
        delta += 2.0 * math.pi
    return delta


class _StateMachineCounter:
    def __init__(self, config: EngineConfig):
        self.config = config
        self.state = "READY"
        self.floor_y: float | None = None
        self.prev_hip: float | None = None
        self.prev_foot: float | None = None
        self.prev_hip_fast: float | None = None
        self.prev_foot_fast: float | None = None
        self.prev_hip_motion: float | None = None
        self.hip_baseline: float | None = None
        self.foot_baseline: float | None = None
        self.saw_descent = False
        self.rebound_recovered = False
        self.last_count_frame: int | None = None
        self.running_count = 0
        self.interval_history: deque[int] = deque(maxlen=5)

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
        hip_vel = 0.0 if self.prev_hip_motion is None else hip_motion - self.prev_hip_motion
        self.prev_hip_motion = hip_motion

        if self.floor_y is None:
            self.floor_y = foot_motion
        else:
            self.floor_y = max(foot_motion, self.floor_y - self.config.floor_decay_ratio * leg_length)
        return foot_motion, hip_vel, leg_length, fast_mode

    def _advance(self, signal: SignalFrame, allow_count: bool) -> CounterEvent | None:
        if not signal.detected:
            return None
        assert signal.left_foot_y is not None
        assert signal.right_foot_y is not None
        foot_motion, hip_vel, leg_length, fast_mode = self._update_filtered_signal(signal)

        contact_threshold = self.floor_y - (self.config.contact_margin_ratio * leg_length)
        symmetry_y = abs(signal.left_foot_y - signal.right_foot_y)
        contact_gate = foot_motion >= contact_threshold and symmetry_y <= (self.config.symmetry_y_ratio * leg_length)
        descend_ratio = self.config.fast_descend_velocity_ratio if fast_mode else self.config.descend_velocity_ratio
        ascend_ratio = self.config.fast_ascend_velocity_ratio if fast_mode else self.config.ascend_velocity_ratio
        min_refractory = self.config.fast_min_refractory_frames if fast_mode else self.config.min_refractory_frames
        descending = hip_vel >= (descend_ratio * leg_length)
        ascending = hip_vel <= -(ascend_ratio * leg_length)
        enough_refractory = self.last_count_frame is None or (signal.frame_idx - self.last_count_frame) >= min_refractory

        if self.state == "REBOUND_LOCK":
            if ascending:
                self.rebound_recovered = True
            if self.rebound_recovered and descending and enough_refractory:
                self.state = "CONTACT" if contact_gate else "READY"
                self.rebound_recovered = False
                self.saw_descent = descending and contact_gate
            return None

        if contact_gate:
            self.state = "CONTACT"
        elif self.state == "CONTACT":
            self.state = "READY"
            self.saw_descent = False

        if self.state == "CONTACT" and descending:
            self.saw_descent = True

        if self.state == "CONTACT" and self.saw_descent and ascending and enough_refractory:
            self.state = "REBOUND_LOCK"
            self.saw_descent = False
            self.rebound_recovered = ascending
            if not allow_count:
                return None
            if self.last_count_frame is not None:
                self.interval_history.append(signal.frame_idx - self.last_count_frame)
            self.running_count += 1
            self.last_count_frame = signal.frame_idx
            return CounterEvent(
                frame_idx=signal.frame_idx,
                time_sec=signal.time_sec,
                running_count=self.running_count,
            )
        return None

    def warmup(self, signal: SignalFrame) -> None:
        self._advance(signal, allow_count=False)

    def step(self, signal: SignalFrame) -> CounterEvent | None:
        return self._advance(signal, allow_count=True)


class RealtimeCounterEngine:
    def __init__(self, config: EngineConfig):
        self.config = config
        self.state_engine = _StateMachineCounter(config)
        self.wrist_speed_filter = _ButterworthLowPass(config.wrist_speed_cutoff_hz)
        self.hip_history: deque[float] = deque(maxlen=config.motion_window_frames)
        self.foot_history: deque[float] = deque(maxlen=config.motion_window_frames)
        self.leg_history: deque[float] = deque(maxlen=config.motion_window_frames)
        self.recent_hip_history: deque[float] = deque(maxlen=config.recent_window_frames)
        self.candidate_interval_history: deque[int] = deque(maxlen=config.adaptive_gap_history)
        self.candidate_hip_history: deque[float] = deque(maxlen=config.adaptive_gap_history)
        self.candidate_foot_history: deque[float] = deque(maxlen=config.adaptive_gap_history)
        self.last_candidate_frame: int | None = None
        self.prev_time_sec: float | None = None
        self.prev_left_forearm_angle: float | None = None
        self.prev_right_forearm_angle: float | None = None
        self.last_speed_1: float | None = None
        self.last_speed_2: float | None = None
        self.last_global_peak_frame: int | None = None
        self.last_wrist_peak_frame: int | None = None
        self.last_accepted_frame: int | None = None
        self.accepted_running_count = 0
        self.jump_interval_history: deque[int] = deque(maxlen=config.adaptive_gap_history)
        self.wrist_speed_history: deque[float] = deque(maxlen=config.fft_window_frames)
        self.time_delta_history: deque[float] = deque(maxlen=config.fft_window_frames)
        self.wrist_peak_interval_history: deque[int] = deque(maxlen=config.wrist_period_history_size)
        self.accepted_cadence_history: deque[float] = deque(maxlen=config.accepted_profile_history_size)
        self.accepted_peak_speed_history: deque[float] = deque(maxlen=config.accepted_profile_history_size)
        self.current_wrist_speed_ratio = 0.0
        self.current_wrist_rotation_count = 0.0
        self.current_wrist_rotation_balance = 0.0
        self.last_decision: CounterDecision | None = None
        self.monitor = MonitorState()

    def _sample_rate(self, timestamp_sec: float) -> tuple[float, float]:
        if self.prev_time_sec is None:
            return 30.0, 1.0 / 30.0
        dt = max(1e-3, timestamp_sec - self.prev_time_sec)
        self.time_delta_history.append(dt)
        return 1.0 / dt, dt

    def _arm_landmarks_available(self, signal: SignalFrame) -> bool:
        return all(
            value is not None
            for value in (
                signal.left_shoulder_x,
                signal.left_shoulder_y,
                signal.right_shoulder_x,
                signal.right_shoulder_y,
                signal.left_elbow_x,
                signal.left_elbow_y,
                signal.right_elbow_x,
                signal.right_elbow_y,
                signal.left_wrist_x,
                signal.left_wrist_y,
                signal.right_wrist_x,
                signal.right_wrist_y,
            )
        )

    def _transform_arm_points(
        self,
        signal: SignalFrame,
    ) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float], tuple[float, float]]:
        assert signal.left_shoulder_x is not None
        assert signal.left_shoulder_y is not None
        assert signal.right_shoulder_x is not None
        assert signal.right_shoulder_y is not None
        assert signal.left_elbow_x is not None
        assert signal.left_elbow_y is not None
        assert signal.right_elbow_x is not None
        assert signal.right_elbow_y is not None
        assert signal.left_wrist_x is not None
        assert signal.left_wrist_y is not None
        assert signal.right_wrist_x is not None
        assert signal.right_wrist_y is not None

        shoulder_center_x = (signal.left_shoulder_x + signal.right_shoulder_x) / 2.0
        shoulder_center_y = (signal.left_shoulder_y + signal.right_shoulder_y) / 2.0
        shoulder_dx = signal.right_shoulder_x - signal.left_shoulder_x
        shoulder_dy = signal.right_shoulder_y - signal.left_shoulder_y
        shoulder_width = max(1e-6, math.hypot(shoulder_dx, shoulder_dy))
        cos_a = shoulder_dx / shoulder_width
        sin_a = shoulder_dy / shoulder_width

        def transform(px: float, py: float) -> tuple[float, float]:
            dx = (px - shoulder_center_x) / shoulder_width
            dy = (py - shoulder_center_y) / shoulder_width
            return (
                (cos_a * dx) + (sin_a * dy),
                (-sin_a * dx) + (cos_a * dy),
            )

        return (
            transform(signal.left_elbow_x, signal.left_elbow_y),
            transform(signal.right_elbow_x, signal.right_elbow_y),
            transform(signal.left_wrist_x, signal.left_wrist_y),
            transform(signal.right_wrist_x, signal.right_wrist_y),
        )

    def _wrist_observation(self, signal: SignalFrame, dt: float) -> tuple[float, float, float]:
        if not self._arm_landmarks_available(signal):
            self.prev_left_forearm_angle = None
            self.prev_right_forearm_angle = None
            return 0.0, 0.0, 0.0
        left_elbow, right_elbow, left_wrist, right_wrist = self._transform_arm_points(signal)

        def single_side(
            elbow: tuple[float, float],
            wrist: tuple[float, float],
            previous_angle: float | None,
        ) -> tuple[float, float, float | None]:
            vec_x = wrist[0] - elbow[0]
            vec_y = wrist[1] - elbow[1]
            forearm_len = math.hypot(vec_x, vec_y)
            if forearm_len < 1e-3:
                return 0.0, 0.0, previous_angle
            angle = math.atan2(vec_y, vec_x)
            if previous_angle is None:
                return 0.0, 0.0, angle
            delta_angle = _wrap_angle_delta(angle, previous_angle)
            angular_speed = abs(delta_angle) / dt
            tangential_speed = angular_speed * forearm_len
            return tangential_speed, abs(delta_angle) / (2.0 * math.pi), angle

        left_speed, left_turns, left_angle = single_side(left_elbow, left_wrist, self.prev_left_forearm_angle)
        right_speed, right_turns, right_angle = single_side(right_elbow, right_wrist, self.prev_right_forearm_angle)
        self.prev_left_forearm_angle = left_angle
        self.prev_right_forearm_angle = right_angle
        return (left_speed + right_speed) / 2.0, left_turns, right_turns

    def _dominant_frequency(self, values: deque[float], min_hz: float, max_hz: float) -> tuple[float, float]:
        if len(values) < max(8, self.config.fft_window_frames // 2):
            return 0.0, 0.0
        if not self.time_delta_history:
            return 0.0, 0.0
        sample_dt = median(self.time_delta_history)
        series = np.asarray(values, dtype=np.float64)
        series = series - float(np.mean(series))
        if np.allclose(series, 0.0):
            return 0.0, 0.0
        window = np.hanning(len(series))
        fft_values = np.fft.rfft(series * window)
        freqs = np.fft.rfftfreq(len(series), d=max(sample_dt, 1e-3))
        power = np.abs(fft_values) ** 2
        band_mask = (freqs >= min_hz) & (freqs <= max_hz)
        if not np.any(band_mask):
            return 0.0, 0.0
        band_power = power[band_mask]
        band_sum = float(np.sum(band_power))
        if band_sum <= 0.0:
            return 0.0, 0.0
        peak_index = int(np.argmax(band_power))
        peak_power = float(band_power[peak_index])
        peak_freq = float(freqs[band_mask][peak_index])
        return peak_freq, peak_power / band_sum

    def _wrist_period_stats(self, fs: float) -> tuple[float, float, float]:
        if not self.wrist_peak_interval_history:
            return 0.0, 0.0, 0.0
        intervals = list(self.wrist_peak_interval_history)
        period_frames = float(median(intervals))
        if period_frames <= 0.0:
            return 0.0, 0.0, 0.0
        spread = median(abs(value - period_frames) for value in intervals) if len(intervals) > 1 else 0.0
        confidence = max(0.0, min(1.0, 1.0 - (spread / max(period_frames, 1.0))))
        cadence_hz = fs / period_frames if fs > 0 else 0.0
        return period_frames, cadence_hz, confidence

    def _cadence_profile_ratio(self, wrist_cadence_hz: float) -> float:
        if wrist_cadence_hz <= 0.0 or not self.accepted_cadence_history:
            return 1.0
        reference = float(median(self.accepted_cadence_history))
        if reference <= 1e-6:
            return 1.0
        ratio = min(wrist_cadence_hz, reference) / max(wrist_cadence_hz, reference)
        return max(0.0, min(1.0, ratio))

    def _adaptive_peak_floor(self) -> float:
        if not self.accepted_peak_speed_history:
            return self.config.min_wrist_peak_speed_ratio
        reference = float(median(self.accepted_peak_speed_history))
        return min(self.config.min_wrist_peak_speed_ratio, reference * 0.75)

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

    def _observe_candidate(self, frame_idx: int, metrics: dict[str, float]) -> None:
        motion_valid = (
            metrics["hip_range_ratio"] >= self.config.min_hip_range_ratio
            and metrics["foot_range_ratio"] >= self.config.min_foot_range_ratio
        )
        if not motion_valid:
            return
        if self.last_candidate_frame is not None and frame_idx > self.last_candidate_frame:
            self.candidate_interval_history.append(frame_idx - self.last_candidate_frame)
        self.last_candidate_frame = frame_idx
        self.candidate_hip_history.append(metrics["hip_range_ratio"])
        self.candidate_foot_history.append(metrics["foot_range_ratio"])

    def _cadence_locked(self) -> bool:
        if not self.config.adaptive_gap_enabled:
            return False
        if len(self.candidate_interval_history) < self.config.adaptive_gap_min_intervals:
            return False
        if not self.candidate_hip_history or not self.candidate_foot_history:
            return False
        return (
            median(self.candidate_hip_history) >= self.config.adaptive_motion_hip_ratio
            and median(self.candidate_foot_history) >= self.config.adaptive_motion_foot_ratio
        )

    def _effective_limits(self) -> tuple[int, float, bool]:
        min_gap_frames = self.config.min_count_gap_frames
        min_recent_hip_ratio = self.config.min_recent_hip_range_ratio
        cadence_locked = self._cadence_locked()

        if cadence_locked:
            raw_interval_median = median(self.candidate_interval_history)
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
        wrist_peak_count: int,
        wrist_peak_speed_ratio: float,
        wrist_energy_ratio: float,
        wrist_fft_peak_hz: float,
        wrist_fft_power_ratio: float,
        wrist_to_jump_ratio: float,
        wrist_rotation_count: float,
        wrist_rotation_balance: float,
        wrist_rotation_cadence_hz: float,
        wrist_period_frames: float,
        wrist_period_confidence: float,
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
            wrist_peak_count=wrist_peak_count,
            wrist_peak_speed_ratio=wrist_peak_speed_ratio,
            wrist_energy_ratio=wrist_energy_ratio,
            wrist_fft_peak_hz=wrist_fft_peak_hz,
            wrist_fft_power_ratio=wrist_fft_power_ratio,
            wrist_to_jump_ratio=wrist_to_jump_ratio,
            wrist_rotation_count=wrist_rotation_count,
            wrist_rotation_balance=wrist_rotation_balance,
            wrist_rotation_cadence_hz=wrist_rotation_cadence_hz,
            wrist_period_frames=wrist_period_frames,
            wrist_period_confidence=wrist_period_confidence,
            min_gap_frames=min_gap_frames,
            cadence_locked=cadence_locked,
        )

    def _observe_peak(self, frame_idx: int) -> int | None:
        if self.last_speed_2 is None or self.last_speed_1 is None or len(self.wrist_speed_history) < 1:
            return None
        current = self.wrist_speed_history[-1]
        if self.last_speed_1 < self._adaptive_peak_floor():
            return None
        if self.last_speed_2 > self.last_speed_1 or current > self.last_speed_1:
            return None
        peak_frame = frame_idx - 1
        if self.last_global_peak_frame is not None and (peak_frame - self.last_global_peak_frame) > 0:
            self.wrist_peak_interval_history.append(peak_frame - self.last_global_peak_frame)
        self.last_global_peak_frame = peak_frame
        if (
            self.last_wrist_peak_frame is not None
            and (peak_frame - self.last_wrist_peak_frame) < self.config.wrist_peak_refractory_frames
        ):
            return None
        self.last_wrist_peak_frame = peak_frame
        return peak_frame

    def _update_monitor(
        self,
        signal: SignalFrame,
        metrics: dict[str, float],
        fs: float,
    ) -> tuple[float, float, float]:
        period_frames, cadence_hz, confidence = self._wrist_period_stats(fs)
        cadence_profile_ratio = self._cadence_profile_ratio(cadence_hz)
        self.monitor = MonitorState(
            frame_idx=signal.frame_idx,
            time_sec=signal.time_sec,
            detected=signal.detected,
            contact_gate=(self.state_engine.state == "CONTACT"),
            in_air=False,
            jump_height_ratio=metrics["foot_range_ratio"],
            hip_lift_ratio=metrics["hip_range_ratio"],
            wrist_speed_ratio=self.current_wrist_speed_ratio,
            wrist_peak_count=0,
            wrist_rotation_count=self.current_wrist_rotation_count,
            wrist_rotation_balance=self.current_wrist_rotation_balance,
            wrist_period_frames=period_frames,
            wrist_period_sec=(period_frames / fs) if fs > 0 and period_frames > 0 else 0.0,
            wrist_period_confidence=confidence,
            wrist_cadence_hz=cadence_hz,
            cadence_profile_ratio=cadence_profile_ratio,
        )
        return period_frames, cadence_hz, confidence

    def warmup(self, signal: SignalFrame) -> None:
        self.last_decision = None
        self._step_internal(signal, allow_count=False)

    def step(self, signal: SignalFrame) -> CounterEvent | None:
        self.last_decision = None
        return self._step_internal(signal, allow_count=True)

    def _step_internal(self, signal: SignalFrame, allow_count: bool) -> CounterEvent | None:
        if not signal.detected:
            self.monitor = MonitorState(frame_idx=signal.frame_idx, time_sec=signal.time_sec, detected=False)
            self.prev_time_sec = signal.time_sec
            return None

        fs, dt = self._sample_rate(signal.time_sec)
        wrist_speed_raw, left_turns, right_turns = self._wrist_observation(signal, dt)
        wrist_speed_ratio = self.wrist_speed_filter.filter(wrist_speed_raw, fs)
        self.wrist_speed_history.append(wrist_speed_ratio)
        peak_frame = self._observe_peak(signal.frame_idx)
        self.last_speed_2 = self.last_speed_1
        self.last_speed_1 = wrist_speed_ratio
        self.current_wrist_speed_ratio = wrist_speed_ratio
        self.current_wrist_rotation_count = (left_turns + right_turns) / 2.0
        self.current_wrist_rotation_balance = (
            min(left_turns, right_turns) / max(left_turns, right_turns)
            if max(left_turns, right_turns) > 1e-6
            else 0.0
        )

        if peak_frame is not None and self.last_speed_1 is not None:
            self.accepted_peak_speed_history.append(self.last_speed_1)

        self._update_motion_history(signal)
        metrics = self.motion_metrics()
        wrist_period_frames, wrist_cadence_hz, wrist_period_confidence = self._update_monitor(signal, metrics, fs)
        candidate = self.state_engine._advance(signal, allow_count)
        self.prev_time_sec = signal.time_sec

        if not allow_count or candidate is None:
            return None

        self._observe_candidate(candidate.frame_idx, metrics)
        min_gap_frames, min_recent_hip_ratio, cadence_locked = self._effective_limits()
        required_history = max(3, self.config.motion_window_frames // 2)
        wrist_peak_count = 1 if peak_frame is not None else 0
        wrist_peak_speed_ratio = self.current_wrist_speed_ratio
        wrist_energy_ratio = float(np.mean(self.wrist_speed_history)) if self.wrist_speed_history else 0.0
        wrist_fft_peak_hz, wrist_fft_power_ratio = self._dominant_frequency(self.wrist_speed_history, 1.8, 12.0)
        wrist_to_jump_ratio = wrist_fft_peak_hz / max(1.0, 2.0) if wrist_fft_peak_hz > 0 else 0.0
        wrist_rotation_count = self.current_wrist_rotation_count
        wrist_rotation_balance = self.current_wrist_rotation_balance

        reject_reason: str | None = None
        if len(self.hip_history) < required_history:
            reject_reason = "insufficient_window"
        elif self.last_accepted_frame is not None and (candidate.frame_idx - self.last_accepted_frame) < min_gap_frames:
            reject_reason = "min_gap"
        elif metrics["hip_range_ratio"] < self.config.min_hip_range_ratio:
            reject_reason = "hip_range"
        elif metrics["foot_range_ratio"] < self.config.min_foot_range_ratio and not self._foot_floor_override(metrics):
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

        if reject_reason is not None:
            self._set_decision(
                signal,
                False,
                reject_reason,
                0,
                metrics["foot_range_ratio"],
                metrics["hip_range_ratio"],
                wrist_peak_count,
                wrist_peak_speed_ratio,
                wrist_energy_ratio,
                wrist_fft_peak_hz,
                wrist_fft_power_ratio,
                wrist_to_jump_ratio,
                wrist_rotation_count,
                wrist_rotation_balance,
                wrist_cadence_hz,
                wrist_period_frames,
                wrist_period_confidence,
                min_gap_frames,
                cadence_locked,
            )
            return None

        self.last_accepted_frame = candidate.frame_idx
        self.accepted_running_count += 1
        if wrist_cadence_hz > 0.0:
            self.accepted_cadence_history.append(wrist_cadence_hz)
        self._set_decision(
            signal,
            True,
            "accepted",
            0,
            metrics["foot_range_ratio"],
            metrics["hip_range_ratio"],
            wrist_peak_count,
            wrist_peak_speed_ratio,
            wrist_energy_ratio,
            wrist_fft_peak_hz,
            wrist_fft_power_ratio,
            wrist_to_jump_ratio,
            wrist_rotation_count,
            wrist_rotation_balance,
            wrist_cadence_hz,
            wrist_period_frames,
            wrist_period_confidence,
            min_gap_frames,
            cadence_locked,
        )
        return CounterEvent(
            frame_idx=candidate.frame_idx,
            time_sec=candidate.time_sec,
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

    effective_warmup = max(warmup_frames, config.fft_window_frames)
    start_index = max(0, start_frame - effective_warmup)
    for signal in signals[start_index:]:
        if signal.frame_idx > effective_end_frame:
            break
        if signal.frame_idx < start_frame:
            engine.warmup(signal)
            continue
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
            takeoff_hip_ratio=values[1],
            min_jump_height_ratio=values[2],
            min_hip_lift_ratio=values[3],
            min_count_gap_frames=values[4],
            min_accept_score=values[5],
            contact_margin_ratio=values[6],
        )
        for values in product(
            [0.014, 0.016, 0.020],
            [0.008, 0.010, 0.012],
            [0.016, 0.018, 0.022],
            [0.008, 0.010, 0.012],
            [5, 6, 7],
            [2, 3],
            [0.08, 0.10, 0.12],
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
