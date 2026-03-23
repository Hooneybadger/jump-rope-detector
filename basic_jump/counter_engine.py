from __future__ import annotations

import json
import time
import xml.etree.ElementTree as ET
from collections import deque
from dataclasses import asdict, dataclass
from itertools import product
from pathlib import Path
from statistics import median

import cv2
import mediapipe as mp


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
ALL_POSE_LANDMARKS = tuple(mp_pose.PoseLandmark)


@dataclass(frozen=True)
class EngineConfig:
    ema_alpha_hip: float = 0.45
    ema_alpha_foot: float = 0.55
    fast_ema_alpha_hip: float = 0.25
    fast_ema_alpha_foot: float = 0.35
    foot_visibility_threshold: float = 0.35
    hip_visibility_threshold: float = 0.50
    contact_margin_ratio: float = 0.08
    symmetry_y_ratio: float = 0.12
    descend_velocity_ratio: float = 0.006
    ascend_velocity_ratio: float = 0.004
    fast_descend_velocity_ratio: float = 0.002
    fast_ascend_velocity_ratio: float = 0.001
    floor_decay_ratio: float = 0.004
    min_refractory_frames: int = 4
    fast_min_refractory_frames: int = 2
    fast_mode_cadence_threshold: int = 7
    motion_window_frames: int = 18
    recent_window_frames: int = 5
    min_hip_range_ratio: float = 0.045
    min_foot_range_ratio: float = 0.035
    min_recent_hip_range_ratio: float = 0.032
    max_foot_to_hip_ratio: float = 2.5
    guard_low_hip_range_ratio: float = 0.09
    guard_high_foot_range_ratio: float = 0.12
    guard_recent_hip_range_ratio: float = 0.04
    balanced_override_hip_range_ratio: float = 0.075
    balanced_override_foot_range_ratio: float = 0.065
    balanced_override_recent_hip_ratio: float = 0.024
    balanced_override_min_ratio: float = 0.60
    balanced_override_max_ratio: float = 1.50
    extended_override_hip_range_ratio: float = 0.055
    extended_override_foot_range_ratio: float = 0.100
    extended_override_recent_hip_ratio: float = 0.028
    extended_override_min_ratio: float = 1.60
    extended_override_max_ratio: float = 2.00
    extended_override_recent_to_hip_ratio: float = 0.45
    foot_floor_override_hip_range_ratio: float = 0.10
    foot_floor_override_foot_range_ratio: float = 0.028
    foot_floor_override_recent_hip_ratio: float = 0.060
    foot_floor_override_max_ratio: float = 0.31
    stale_tail_guard_hip_range_ratio: float = 0.20
    stale_tail_guard_recent_to_hip_ratio: float = 0.177
    min_count_gap_frames: int = 9
    adaptive_gap_enabled: bool = True
    adaptive_gap_factor: float = 0.70
    adaptive_gap_history: int = 5
    adaptive_gap_min_intervals: int = 1
    adaptive_gap_floor_frames: int = 4
    adaptive_motion_hip_ratio: float = 0.06
    adaptive_motion_foot_ratio: float = 0.05
    adaptive_recent_hip_enabled: bool = True
    adaptive_recent_hip_floor: float = 0.032

    def to_dict(self) -> dict[str, float | int]:
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


@dataclass
class CounterDecision:
    frame_idx: int
    time_sec: float
    accepted: bool
    reason: str
    hip_range_ratio: float
    foot_range_ratio: float
    recent_hip_range_ratio: float
    foot_to_hip_ratio: float
    min_gap_frames: int
    min_recent_hip_ratio: float
    cadence_locked: bool


@dataclass(frozen=True)
class LabelWindowConfig:
    start_offset_frames: int = -15
    end_offset_frames: int = 0
    warmup_frames: int = 4

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


def _timestamp_mode(root: ET.Element) -> str:
    avg_ts_per_frame = float(root.findtext("AverageTimeStampsPerFrame", "0"))
    return "ms" if avg_ts_per_frame < 100 else "ticks"


def _timestamp_to_frame(timestamp_raw: int, mode: str, fps: float) -> tuple[int, float]:
    if mode == "ms":
        time_sec = timestamp_raw / 1000.0
        return int(round(time_sec * fps)), time_sec
    frame_idx = int(round(timestamp_raw / 512.0))
    return frame_idx, frame_idx / fps


def parse_label_file(path: str | Path, fps: float = 30.0) -> list[LabelEvent]:
    path = Path(path)
    root = ET.parse(path).getroot()
    mode = _timestamp_mode(root)
    keyframes = root.find("Keyframes")
    if keyframes is None:
        return []

    raw_events: list[LabelEvent] = []
    for index, keyframe in enumerate(keyframes.findall("Keyframe")):
        point_count, anomaly_tags = _parse_label_points(keyframe)
        frame_idx, time_sec = _timestamp_to_frame(int(keyframe.findtext("Timestamp", "0")), mode, fps)
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
        left_hip.visibility < config.hip_visibility_threshold
        or right_hip.visibility < config.hip_visibility_threshold
    ):
        return SignalFrame(frame_idx=frame_idx, time_sec=timestamp_sec, detected=False)

    left_foot_y = _pick_foot_y(lms, "left", config.foot_visibility_threshold)
    right_foot_y = _pick_foot_y(lms, "right", config.foot_visibility_threshold)
    if left_foot_y is None or right_foot_y is None:
        return SignalFrame(frame_idx=frame_idx, time_sec=timestamp_sec, detected=False)

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
        left_hip_y=float(left_hip.y),
        right_hip_y=float(right_hip.y),
        left_foot_y=left_foot_y,
        right_foot_y=right_foot_y,
        leg_length=leg_length,
    )


def landmark_visibility_ratio(result, visibility_threshold: float = 0.30) -> float:
    if not result.pose_landmarks:
        return 0.0
    lms = result.pose_landmarks.landmark
    visible_count = sum(
        1 for landmark in ALL_POSE_LANDMARKS if float(lms[landmark.value].visibility) >= visibility_threshold
    )
    return visible_count / max(1, len(ALL_POSE_LANDMARKS))


def full_landmarks_visible(
    result,
    visibility_threshold: float = 0.30,
    required_ratio: float = 1.0,
) -> bool:
    return landmark_visibility_ratio(result, visibility_threshold) >= required_ratio


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


class _StateMachineCounter:
    def __init__(self, config: EngineConfig):
        self.config = config
        self.state = "READY"
        self.floor_y: float | None = None
        self.prev_hip: float | None = None
        self.prev_foot: float | None = None
        self.prev_hip_fast: float | None = None
        self.prev_foot_fast: float | None = None
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

    def _update_filtered_signal(
        self,
        signal: SignalFrame,
    ) -> tuple[float, float, float, bool]:
        assert signal.left_hip_y is not None
        assert signal.right_hip_y is not None
        assert signal.left_foot_y is not None
        assert signal.right_foot_y is not None
        assert signal.leg_length is not None

        mean_hip_y_raw = (signal.left_hip_y + signal.right_hip_y) / 2.0
        mean_foot_y_raw = (signal.left_foot_y + signal.right_foot_y) / 2.0
        mean_hip_y_std = self._ema(self.config.ema_alpha_hip, mean_hip_y_raw, self.prev_hip)
        mean_foot_y_std = self._ema(self.config.ema_alpha_foot, mean_foot_y_raw, self.prev_foot)
        mean_hip_y_fast = self._ema(self.config.fast_ema_alpha_hip, mean_hip_y_raw, self.prev_hip_fast)
        mean_foot_y_fast = self._ema(self.config.fast_ema_alpha_foot, mean_foot_y_raw, self.prev_foot_fast)
        fast_mode = len(self.interval_history) >= 3 and median(self.interval_history) <= self.config.fast_mode_cadence_threshold
        mean_hip_y = mean_hip_y_fast if fast_mode else mean_hip_y_std
        mean_foot_y = mean_foot_y_fast if fast_mode else mean_foot_y_std
        prev_hip_y = self.prev_hip_fast if fast_mode else self.prev_hip
        hip_vel = 0.0 if prev_hip_y is None else mean_hip_y - prev_hip_y

        self.prev_hip = mean_hip_y_std
        self.prev_foot = mean_foot_y_std
        self.prev_hip_fast = mean_hip_y_fast
        self.prev_foot_fast = mean_foot_y_fast

        if self.floor_y is None:
            self.floor_y = mean_foot_y
        else:
            self.floor_y = max(mean_foot_y, self.floor_y - self.config.floor_decay_ratio * signal.leg_length)
        return mean_foot_y, hip_vel, signal.leg_length, fast_mode

    def warmup(self, signal: SignalFrame) -> None:
        if not signal.detected:
            return
        self._update_filtered_signal(signal)
        self.state = "READY"
        self.saw_descent = False
        self.rebound_recovered = False

    def step(self, signal: SignalFrame) -> CounterEvent | None:
        if not signal.detected:
            return None
        assert signal.left_foot_y is not None
        assert signal.right_foot_y is not None
        mean_foot_y, hip_vel, leg_length, fast_mode = self._update_filtered_signal(signal)

        contact_threshold = self.floor_y - (self.config.contact_margin_ratio * leg_length)
        symmetry_y = abs(signal.left_foot_y - signal.right_foot_y)
        contact_gate = mean_foot_y >= contact_threshold and symmetry_y <= (
            self.config.symmetry_y_ratio * leg_length
        )
        descend_ratio = self.config.fast_descend_velocity_ratio if fast_mode else self.config.descend_velocity_ratio
        ascend_ratio = self.config.fast_ascend_velocity_ratio if fast_mode else self.config.ascend_velocity_ratio
        min_refractory = self.config.fast_min_refractory_frames if fast_mode else self.config.min_refractory_frames
        descending = hip_vel >= (descend_ratio * leg_length)
        ascending = hip_vel <= -(ascend_ratio * leg_length)
        enough_refractory = (
            self.last_count_frame is None
            or (signal.frame_idx - self.last_count_frame) >= min_refractory
        )

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

        if (
            self.state == "CONTACT"
            and self.saw_descent
            and ascending
            and enough_refractory
        ):
            if self.last_count_frame is not None:
                self.interval_history.append(signal.frame_idx - self.last_count_frame)
            self.running_count += 1
            self.last_count_frame = signal.frame_idx
            self.state = "REBOUND_LOCK"
            self.saw_descent = False
            self.rebound_recovered = ascending
            return CounterEvent(
                frame_idx=signal.frame_idx,
                time_sec=signal.time_sec,
                running_count=self.running_count,
        )
        return None


class RealtimeCounterEngine:
    def __init__(self, config: EngineConfig):
        self.config = config
        self.state_engine = _StateMachineCounter(config)
        self.hip_history: deque[float] = deque(maxlen=config.motion_window_frames)
        self.foot_history: deque[float] = deque(maxlen=config.motion_window_frames)
        self.leg_history: deque[float] = deque(maxlen=config.motion_window_frames)
        self.recent_hip_history: deque[float] = deque(maxlen=config.recent_window_frames)
        self.candidate_interval_history: deque[int] = deque(maxlen=config.adaptive_gap_history)
        self.candidate_hip_history: deque[float] = deque(maxlen=config.adaptive_gap_history)
        self.candidate_foot_history: deque[float] = deque(maxlen=config.adaptive_gap_history)
        self.last_candidate_frame: int | None = None
        self.last_accepted_frame: int | None = None
        self.accepted_running_count = 0
        self.last_decision: CounterDecision | None = None

    def _update_motion_history(self, signal: SignalFrame) -> None:
        if not signal.detected:
            return
        assert signal.left_hip_y is not None
        assert signal.right_hip_y is not None
        assert signal.left_foot_y is not None
        assert signal.right_foot_y is not None
        assert signal.leg_length is not None

        mean_hip = (signal.left_hip_y + signal.right_hip_y) / 2.0
        mean_foot = (signal.left_foot_y + signal.right_foot_y) / 2.0
        self.hip_history.append(mean_hip)
        self.foot_history.append(mean_foot)
        self.leg_history.append(signal.leg_length)
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
            recent_hip_range_ratio = (
                max(self.recent_hip_history) - min(self.recent_hip_history)
            ) / leg_median
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

    def _reject(
        self,
        candidate: CounterEvent,
        reason: str,
        metrics: dict[str, float],
        min_gap_frames: int,
        min_recent_hip_ratio: float,
        cadence_locked: bool,
    ) -> None:
        self.last_decision = CounterDecision(
            frame_idx=candidate.frame_idx,
            time_sec=candidate.time_sec,
            accepted=False,
            reason=reason,
            hip_range_ratio=metrics["hip_range_ratio"],
            foot_range_ratio=metrics["foot_range_ratio"],
            recent_hip_range_ratio=metrics["recent_hip_range_ratio"],
            foot_to_hip_ratio=metrics["foot_to_hip_ratio"],
            min_gap_frames=min_gap_frames,
            min_recent_hip_ratio=min_recent_hip_ratio,
            cadence_locked=cadence_locked,
        )

    def _balanced_motion_override(self, metrics: dict[str, float], cadence_locked: bool) -> bool:
        if not cadence_locked:
            return False
        return (
            metrics["hip_range_ratio"] >= self.config.balanced_override_hip_range_ratio
            and metrics["foot_range_ratio"] >= self.config.balanced_override_foot_range_ratio
            and metrics["recent_hip_range_ratio"] >= self.config.balanced_override_recent_hip_ratio
            and self.config.balanced_override_min_ratio
            <= metrics["foot_to_hip_ratio"]
            <= self.config.balanced_override_max_ratio
        )

    def _extended_motion_override(self, metrics: dict[str, float], cadence_locked: bool) -> bool:
        if not cadence_locked:
            return False
        recent_to_hip_ratio = metrics["recent_hip_range_ratio"] / max(metrics["hip_range_ratio"], 1e-6)
        return (
            metrics["hip_range_ratio"] >= self.config.extended_override_hip_range_ratio
            and metrics["foot_range_ratio"] >= self.config.extended_override_foot_range_ratio
            and metrics["recent_hip_range_ratio"] >= self.config.extended_override_recent_hip_ratio
            and self.config.extended_override_min_ratio
            <= metrics["foot_to_hip_ratio"]
            <= self.config.extended_override_max_ratio
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

    def warmup(self, signal: SignalFrame) -> None:
        self.last_decision = None
        if not signal.detected:
            return
        self._update_motion_history(signal)
        self.state_engine.warmup(signal)

    def step(self, signal: SignalFrame) -> CounterEvent | None:
        self.last_decision = None
        if not signal.detected:
            return None

        self._update_motion_history(signal)
        candidate = self.state_engine.step(signal)
        if candidate is None:
            return None

        metrics = self.motion_metrics()
        self._observe_candidate(candidate.frame_idx, metrics)
        min_gap_frames, min_recent_hip_ratio, cadence_locked = self._effective_limits()

        required_history = max(3, self.config.motion_window_frames // 2)
        if len(self.hip_history) < required_history:
            self._reject(candidate, "insufficient_window", metrics, min_gap_frames, min_recent_hip_ratio, cadence_locked)
            return None
        if (
            self.last_accepted_frame is not None
            and (candidate.frame_idx - self.last_accepted_frame) < min_gap_frames
        ):
            self._reject(candidate, "min_gap", metrics, min_gap_frames, min_recent_hip_ratio, cadence_locked)
            return None
        if metrics["hip_range_ratio"] < self.config.min_hip_range_ratio:
            self._reject(candidate, "hip_range", metrics, min_gap_frames, min_recent_hip_ratio, cadence_locked)
            return None
        if (
            metrics["foot_range_ratio"] < self.config.min_foot_range_ratio
            and not self._foot_floor_override(metrics)
        ):
            self._reject(candidate, "foot_range", metrics, min_gap_frames, min_recent_hip_ratio, cadence_locked)
            return None
        if (
            metrics["recent_hip_range_ratio"] < min_recent_hip_ratio
            and not self._balanced_motion_override(metrics, cadence_locked)
            and not self._extended_motion_override(metrics, cadence_locked)
        ):
            self._reject(candidate, "recent_hip_range", metrics, min_gap_frames, min_recent_hip_ratio, cadence_locked)
            return None
        if metrics["foot_to_hip_ratio"] > self.config.max_foot_to_hip_ratio:
            self._reject(candidate, "foot_to_hip_ratio", metrics, min_gap_frames, min_recent_hip_ratio, cadence_locked)
            return None
        if (
            metrics["hip_range_ratio"] < self.config.guard_low_hip_range_ratio
            and metrics["foot_range_ratio"] >= self.config.guard_high_foot_range_ratio
            and metrics["recent_hip_range_ratio"] < self.config.guard_recent_hip_range_ratio
        ):
            self._reject(candidate, "foot_dominant_low_hip", metrics, min_gap_frames, min_recent_hip_ratio, cadence_locked)
            return None
        if self._stale_tail_reject(metrics, cadence_locked):
            self._reject(candidate, "stale_tail", metrics, min_gap_frames, min_recent_hip_ratio, cadence_locked)
            return None

        self.last_accepted_frame = candidate.frame_idx
        self.accepted_running_count += 1
        self.last_decision = CounterDecision(
            frame_idx=candidate.frame_idx,
            time_sec=candidate.time_sec,
            accepted=True,
            reason="accepted",
            hip_range_ratio=metrics["hip_range_ratio"],
            foot_range_ratio=metrics["foot_range_ratio"],
            recent_hip_range_ratio=metrics["recent_hip_range_ratio"],
            foot_to_hip_ratio=metrics["foot_to_hip_ratio"],
            min_gap_frames=min_gap_frames,
            min_recent_hip_ratio=min_recent_hip_ratio,
            cadence_locked=cadence_locked,
        )
        return CounterEvent(
            frame_idx=candidate.frame_idx,
            time_sec=candidate.time_sec,
            running_count=self.accepted_running_count,
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

    effective_warmup = max(warmup_frames, config.motion_window_frames)
    start_index = max(0, start_frame - effective_warmup)
    for signal in signals[start_index:]:
        if signal.frame_idx > effective_end_frame:
            break
        event = engine.step(signal)
        if event is not None and start_frame <= event.frame_idx <= effective_end_frame:
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
        predicted_count = len(events)
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
    summary = {
        "overall_count_accuracy": (1.0 - (total_abs_error / total_gt_count)) if total_gt_count else 0.0,
        "exact_video_count_accuracy": exact_matches / len(results) if results else 0.0,
        "videos": [asdict(result) for result in results],
        "total_gt_count": total_gt_count,
        "total_predicted_count": total_predicted_count,
        "signed_total_error": total_predicted_count - total_gt_count,
        "total_abs_error": total_abs_error,
    }
    return summary


def default_search_configs(limit: int | None = None) -> list[EngineConfig]:
    configs = [
        EngineConfig(
            min_hip_range_ratio=values[0],
            min_foot_range_ratio=values[1],
            min_recent_hip_range_ratio=values[2],
            adaptive_recent_hip_floor=values[3],
            min_count_gap_frames=values[4],
            max_foot_to_hip_ratio=values[5],
            guard_low_hip_range_ratio=values[6],
            guard_high_foot_range_ratio=values[7],
            guard_recent_hip_range_ratio=values[8],
        )
        for values in product(
            [0.045, 0.05, 0.055],
            [0.035, 0.04, 0.045],
            [0.032, 0.035, 0.04],
            [0.028, 0.03, 0.032],
            [9, 10],
            [2.5, 3.0],
            [0.09, 0.10],
            [0.12],
            [0.04, 0.05],
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
