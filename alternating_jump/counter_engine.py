from __future__ import annotations

import json
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
CORE_READY_LANDMARKS = (
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
    ema_alpha_hip: float = 0.45
    ema_alpha_foot: float = 0.55
    foot_visibility_threshold: float = 0.30
    hip_visibility_threshold: float = 0.50
    baseline_alpha_hip: float = 0.03
    floor_decay_ratio: float = 0.004
    contact_margin_ratio: float = 0.08
    side_diff_ratio: float = 0.03
    side_release_ratio: float = 0.04
    knee_dominance_weight: float = 0.75
    support_streak_frames: int = 1
    motion_window_frames: int = 14
    recent_window_frames: int = 6
    min_foot_diff_ratio: float = 0.04
    min_hip_motion_ratio: float = 0.0
    min_hip_range_ratio: float = 0.0
    min_recent_hip_range_ratio: float = 0.0
    descend_velocity_ratio: float = -1.0
    min_count_gap_frames: int = 1
    adaptive_gap_enabled: bool = True
    adaptive_gap_factor: float = 0.70
    adaptive_gap_history: int = 6
    adaptive_gap_min_intervals: int = 2
    adaptive_gap_floor_frames: int = 2
    rearm_after_gap_factor: float = 1.60
    rearm_after_gap_min_frames: int = 8
    rearm_fast_interval_min: int = 100
    rearm_fast_interval_max: int = 9
    rearm_interval_spread_max: int = 3
    miss_recovery_enabled: bool = True
    miss_recovery_factor: float = 1.70
    miss_recovery_fast_interval_min: int = 6
    miss_recovery_fast_interval_max: int = 9
    miss_recovery_interval_spread_max: int = 100
    miss_recovery_max_additional_counts: int = 5
    relaxed_contact_enabled: bool = True
    relaxed_contact_margin_ratio: float = 0.16
    relaxed_contact_min_support_ratio: float = 0.12
    relaxed_contact_gap_factor: float = 0.80
    relaxed_contact_fast_interval_min: int = 4
    relaxed_contact_fast_interval_max: int = 6
    relaxed_contact_interval_spread_max: int = 2

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
    left_hip_y: float | None = None
    right_hip_y: float | None = None
    left_knee_y: float | None = None
    right_knee_y: float | None = None
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
    side: str | None
    hip_range_ratio: float
    recent_hip_range_ratio: float
    support_ratio: float
    hip_motion_ratio: float
    hip_velocity_ratio: float
    min_gap_frames: int
    cadence_locked: bool


@dataclass(frozen=True)
class LabelWindowConfig:
    start_offset_frames: int = 0
    end_offset_frames: int = 2
    warmup_frames: int = 8

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
    if point_count != 1:
        anomaly_tags.append(f"point_count_{point_count}")
    return point_count, anomaly_tags


def _timestamp_to_frame(timestamp_raw: int, avg_ts_per_frame: float, fps: float) -> tuple[int, float]:
    avg_ts_per_frame = max(avg_ts_per_frame, 1.0)
    frame_idx = int(round(timestamp_raw / avg_ts_per_frame))
    time_sec = frame_idx / fps if fps > 0 else 0.0
    return frame_idx, time_sec


def parse_label_file(path: str | Path, fps: float = 30.0) -> list[LabelEvent]:
    path = Path(path)
    root = ET.parse(path).getroot()
    avg_ts_per_frame = float(root.findtext("AverageTimeStampsPerFrame", "512"))
    keyframes = root.find("Keyframes")
    if keyframes is None:
        return []

    events: list[LabelEvent] = []
    for index, keyframe in enumerate(keyframes.findall("Keyframe")):
        point_count, anomaly_tags = _parse_label_points(keyframe)
        frame_idx, time_sec = _timestamp_to_frame(
            int(keyframe.findtext("Timestamp", "0")),
            avg_ts_per_frame,
            fps,
        )
        events.append(
            LabelEvent(
                frame_idx=frame_idx,
                time_sec=time_sec,
                point_count=point_count,
                source_indices=[index],
                anomaly_tags=anomaly_tags,
            )
        )
    return events


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
    left_knee = lms[mp_pose.PoseLandmark.LEFT_KNEE.value]
    right_knee = lms[mp_pose.PoseLandmark.RIGHT_KNEE.value]
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
        left_knee_y=float(left_knee.y),
        right_knee_y=float(right_knee.y),
        left_foot_y=left_foot_y,
        right_foot_y=right_foot_y,
        leg_length=leg_length,
    )


def _mean_pose_values(signal: SignalFrame) -> tuple[float, float]:
    assert signal.left_hip_y is not None
    assert signal.right_hip_y is not None
    assert signal.leg_length is not None
    mean_hip_y = (signal.left_hip_y + signal.right_hip_y) / 2.0
    return mean_hip_y, signal.leg_length


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


class RealtimeCounterEngine:
    def __init__(self, config: EngineConfig):
        self.config = config
        self.prev_mean_hip: float | None = None
        self.prev_left_foot: float | None = None
        self.prev_right_foot: float | None = None
        self.prev_left_knee: float | None = None
        self.prev_right_knee: float | None = None
        self.prev_hip_motion: float | None = None
        self.hip_baseline: float | None = None
        self.left_floor: float | None = None
        self.right_floor: float | None = None
        self.candidate_side: str | None = None
        self.candidate_streak = 0
        self.active_side: str | None = None
        self.last_count_frame: int | None = None
        self.accepted_running_count = 0
        self.interval_history: deque[int] = deque(maxlen=config.adaptive_gap_history)
        self.hip_history: deque[float] = deque(maxlen=config.motion_window_frames)
        self.recent_hip_history: deque[float] = deque(maxlen=config.recent_window_frames)
        self.foot_diff_history: deque[float] = deque(maxlen=config.motion_window_frames)
        self.leg_history: deque[float] = deque(maxlen=config.motion_window_frames)
        self.last_decision: CounterDecision | None = None

    @staticmethod
    def _ema(alpha: float, value: float, previous: float | None) -> float:
        if previous is None:
            return value
        return alpha * value + (1.0 - alpha) * previous

    def _update_signal_state(
        self,
        signal: SignalFrame,
    ) -> tuple[float, float, float, float, float, float, float, float]:
        mean_hip_raw, leg_length = _mean_pose_values(signal)
        assert signal.left_knee_y is not None
        assert signal.right_knee_y is not None
        assert signal.left_foot_y is not None
        assert signal.right_foot_y is not None
        mean_hip = self._ema(self.config.ema_alpha_hip, mean_hip_raw, self.prev_mean_hip)
        left_knee = self._ema(self.config.ema_alpha_foot, signal.left_knee_y, self.prev_left_knee)
        right_knee = self._ema(self.config.ema_alpha_foot, signal.right_knee_y, self.prev_right_knee)
        left_foot = self._ema(self.config.ema_alpha_foot, signal.left_foot_y, self.prev_left_foot)
        right_foot = self._ema(self.config.ema_alpha_foot, signal.right_foot_y, self.prev_right_foot)

        self.prev_mean_hip = mean_hip
        self.prev_left_knee = left_knee
        self.prev_right_knee = right_knee
        self.prev_left_foot = left_foot
        self.prev_right_foot = right_foot

        self.left_floor = left_foot if self.left_floor is None else max(
            left_foot,
            self.left_floor - (self.config.floor_decay_ratio * leg_length),
        )
        self.right_floor = right_foot if self.right_floor is None else max(
            right_foot,
            self.right_floor - (self.config.floor_decay_ratio * leg_length),
        )

        self.hip_baseline = self._ema(self.config.baseline_alpha_hip, mean_hip, self.hip_baseline)
        hip_motion = mean_hip - self.hip_baseline
        hip_velocity = 0.0 if self.prev_hip_motion is None else hip_motion - self.prev_hip_motion
        self.prev_hip_motion = hip_motion

        self.hip_history.append(mean_hip)
        self.recent_hip_history.append(mean_hip)
        self.foot_diff_history.append(abs(left_foot - right_foot))
        self.leg_history.append(leg_length)

        return mean_hip, left_knee, right_knee, left_foot, right_foot, leg_length, hip_motion, hip_velocity

    def motion_metrics(self) -> dict[str, float]:
        if not self.hip_history or not self.leg_history:
            return {
                "hip_range_ratio": 0.0,
                "recent_hip_range_ratio": 0.0,
                "foot_diff_peak_ratio": 0.0,
            }
        leg_median = median(self.leg_history)
        hip_range_ratio = (max(self.hip_history) - min(self.hip_history)) / leg_median
        recent_hip_range_ratio = 0.0
        if len(self.recent_hip_history) >= self.config.recent_window_frames:
            recent_hip_range_ratio = (
                max(self.recent_hip_history) - min(self.recent_hip_history)
            ) / leg_median
        foot_diff_peak_ratio = max(self.foot_diff_history) / leg_median if self.foot_diff_history else 0.0
        return {
            "hip_range_ratio": hip_range_ratio,
            "recent_hip_range_ratio": recent_hip_range_ratio,
            "foot_diff_peak_ratio": foot_diff_peak_ratio,
        }

    def _effective_gap(self) -> tuple[int, bool]:
        min_gap_frames = self.config.min_count_gap_frames
        cadence_locked = False
        if (
            self.config.adaptive_gap_enabled
            and len(self.interval_history) >= self.config.adaptive_gap_min_intervals
        ):
            cadence_locked = True
            raw_interval = median(self.interval_history)
            min_gap_frames = max(
                self.config.adaptive_gap_floor_frames,
                min(
                    self.config.min_count_gap_frames,
                    int(round(raw_interval * self.config.adaptive_gap_factor)),
                ),
            )
        return min_gap_frames, cadence_locked

    def _stable_fast_interval(
        self,
        min_interval: int,
        max_interval: int,
        spread_max: int,
    ) -> float | None:
        if len(self.interval_history) < self.config.adaptive_gap_min_intervals:
            return None
        interval_median = median(self.interval_history)
        if interval_median < min_interval or interval_median > max_interval:
            return None
        if (max(self.interval_history) - min(self.interval_history)) > spread_max:
            return None
        return float(interval_median)

    def _rearm_gap_frames(self) -> int | None:
        interval_median = self._stable_fast_interval(
            self.config.rearm_fast_interval_min,
            self.config.rearm_fast_interval_max,
            self.config.rearm_interval_spread_max,
        )
        if interval_median is None:
            return None
        return max(
            self.config.rearm_after_gap_min_frames,
            int(round(interval_median * self.config.rearm_after_gap_factor)),
        )

    def _miss_recovery_count(self, gap_frames: int) -> int:
        if not self.config.miss_recovery_enabled:
            return 0
        interval_median = self._stable_fast_interval(
            self.config.miss_recovery_fast_interval_min,
            self.config.miss_recovery_fast_interval_max,
            self.config.miss_recovery_interval_spread_max,
        )
        if interval_median is None:
            return 0
        if gap_frames < int(round(interval_median * self.config.miss_recovery_factor)):
            return 0
        estimated_total = int(round(gap_frames / interval_median))
        additional_counts = max(0, estimated_total - 1)
        return min(self.config.miss_recovery_max_additional_counts, additional_counts)

    def _relaxed_contact_side(
        self,
        gap_frames: int | None,
        left_knee: float,
        right_knee: float,
        left_foot: float,
        right_foot: float,
        leg_length: float,
    ) -> tuple[str | None, float]:
        if (
            not self.config.relaxed_contact_enabled
            or gap_frames is None
            or self.left_floor is None
            or self.right_floor is None
        ):
            return None, 0.0
        interval_median = self._stable_fast_interval(
            self.config.relaxed_contact_fast_interval_min,
            self.config.relaxed_contact_fast_interval_max,
            self.config.relaxed_contact_interval_spread_max,
        )
        if interval_median is None:
            return None, 0.0
        if gap_frames < int(round(interval_median * self.config.relaxed_contact_gap_factor)):
            return None, 0.0

        foot_diff = left_foot - right_foot
        knee_diff = left_knee - right_knee
        dominance = foot_diff + (self.config.knee_dominance_weight * knee_diff)
        support_ratio = max(abs(foot_diff), abs(dominance)) / leg_length
        if support_ratio < self.config.relaxed_contact_min_support_ratio:
            return None, support_ratio

        left_distance_ratio = max(0.0, (self.left_floor - left_foot) / leg_length)
        right_distance_ratio = max(0.0, (self.right_floor - right_foot) / leg_length)
        side_threshold = self.config.side_diff_ratio * leg_length

        if (
            dominance >= side_threshold
            and left_distance_ratio <= self.config.relaxed_contact_margin_ratio
            and left_distance_ratio < right_distance_ratio
        ):
            return "left", support_ratio
        if (
            dominance <= -side_threshold
            and right_distance_ratio <= self.config.relaxed_contact_margin_ratio
            and right_distance_ratio < left_distance_ratio
        ):
            return "right", support_ratio
        return None, support_ratio

    def _contact_side(
        self,
        left_knee: float,
        right_knee: float,
        left_foot: float,
        right_foot: float,
        leg_length: float,
    ) -> tuple[str | None, float]:
        if self.left_floor is None or self.right_floor is None:
            return None, 0.0
        foot_diff = left_foot - right_foot
        knee_diff = left_knee - right_knee
        dominance = foot_diff + (self.config.knee_dominance_weight * knee_diff)
        support_ratio = max(abs(foot_diff), abs(dominance)) / leg_length
        left_contact = left_foot >= (self.left_floor - (self.config.contact_margin_ratio * leg_length))
        right_contact = right_foot >= (self.right_floor - (self.config.contact_margin_ratio * leg_length))
        if dominance >= (self.config.side_diff_ratio * leg_length) and left_contact:
            return "left", support_ratio
        if dominance <= -(self.config.side_diff_ratio * leg_length) and right_contact:
            return "right", support_ratio
        return None, support_ratio

    def _set_reject(
        self,
        signal: SignalFrame,
        reason: str,
        side: str | None,
        metrics: dict[str, float],
        support_ratio: float,
        hip_motion_ratio: float,
        hip_velocity_ratio: float,
        min_gap_frames: int,
        cadence_locked: bool,
    ) -> None:
        self.last_decision = CounterDecision(
            frame_idx=signal.frame_idx,
            time_sec=signal.time_sec,
            accepted=False,
            reason=reason,
            side=side,
            hip_range_ratio=metrics["hip_range_ratio"],
            recent_hip_range_ratio=metrics["recent_hip_range_ratio"],
            support_ratio=support_ratio,
            hip_motion_ratio=hip_motion_ratio,
            hip_velocity_ratio=hip_velocity_ratio,
            min_gap_frames=min_gap_frames,
            cadence_locked=cadence_locked,
        )

    def warmup(self, signal: SignalFrame) -> None:
        self.last_decision = None
        self._step_internal(signal, allow_count=False)

    def step(self, signal: SignalFrame) -> CounterEvent | None:
        self.last_decision = None
        return self._step_internal(signal, allow_count=True)

    def _step_internal(self, signal: SignalFrame, allow_count: bool) -> CounterEvent | None:
        if not signal.detected:
            self.candidate_side = None
            self.candidate_streak = 0
            self.active_side = None
            return None

        _, left_knee, right_knee, left_foot, right_foot, leg_length, hip_motion, hip_velocity = self._update_signal_state(signal)
        gap_frames = None if self.last_count_frame is None else signal.frame_idx - self.last_count_frame
        side, support_ratio = self._contact_side(left_knee, right_knee, left_foot, right_foot, leg_length)
        if side is None:
            side, relaxed_support_ratio = self._relaxed_contact_side(
                gap_frames,
                left_knee,
                right_knee,
                left_foot,
                right_foot,
                leg_length,
            )
            support_ratio = max(support_ratio, relaxed_support_ratio)

        if support_ratio <= self.config.side_release_ratio:
            self.active_side = None

        if side is None:
            self.candidate_side = None
            self.candidate_streak = 0
            return None

        if side == self.candidate_side:
            self.candidate_streak += 1
        else:
            self.candidate_side = side
            self.candidate_streak = 1

        if self.candidate_streak < self.config.support_streak_frames:
            return None

        rearm_gap_frames = self._rearm_gap_frames()
        if (
            self.active_side is not None
            and self.last_count_frame is not None
            and rearm_gap_frames is not None
            and (signal.frame_idx - self.last_count_frame) >= rearm_gap_frames
        ):
            self.active_side = None

        if side == self.active_side:
            return None

        if not allow_count:
            self.active_side = side
            return None

        metrics = self.motion_metrics()
        min_gap_frames, cadence_locked = self._effective_gap()
        hip_motion_ratio = hip_motion / leg_length
        hip_velocity_ratio = hip_velocity / leg_length

        if (
            self.last_count_frame is not None
            and (signal.frame_idx - self.last_count_frame) < min_gap_frames
        ):
            self._set_reject(
                signal,
                "min_gap",
                side,
                metrics,
                support_ratio,
                hip_motion_ratio,
                hip_velocity_ratio,
                min_gap_frames,
                cadence_locked,
            )
            return None
        if support_ratio < self.config.min_foot_diff_ratio:
            self._set_reject(
                signal,
                "support_ratio",
                side,
                metrics,
                support_ratio,
                hip_motion_ratio,
                hip_velocity_ratio,
                min_gap_frames,
                cadence_locked,
            )
            return None
        if len(self.hip_history) >= max(4, self.config.motion_window_frames // 2):
            if metrics["hip_range_ratio"] < self.config.min_hip_range_ratio:
                self._set_reject(
                    signal,
                    "hip_range",
                    side,
                    metrics,
                    support_ratio,
                    hip_motion_ratio,
                    hip_velocity_ratio,
                    min_gap_frames,
                    cadence_locked,
                )
                return None
        if (
            metrics["recent_hip_range_ratio"] < self.config.min_recent_hip_range_ratio
            and hip_motion_ratio < self.config.min_hip_motion_ratio
        ):
            self._set_reject(
                signal,
                "recent_hip",
                side,
                metrics,
                support_ratio,
                hip_motion_ratio,
                hip_velocity_ratio,
                min_gap_frames,
                cadence_locked,
            )
            return None
        if (
            hip_velocity_ratio < self.config.descend_velocity_ratio
            and hip_motion_ratio < (self.config.min_hip_motion_ratio * 1.35)
        ):
            self._set_reject(
                signal,
                "descend",
                side,
                metrics,
                support_ratio,
                hip_motion_ratio,
                hip_velocity_ratio,
                min_gap_frames,
                cadence_locked,
            )
            return None

        self.active_side = side
        count_delta = 1
        if self.last_count_frame is not None:
            gap_frames = signal.frame_idx - self.last_count_frame
            count_delta += self._miss_recovery_count(gap_frames)
            normalized_gap = max(1, int(round(gap_frames / count_delta)))
            for _ in range(count_delta):
                self.interval_history.append(normalized_gap)
        self.last_count_frame = signal.frame_idx
        self.accepted_running_count += count_delta
        self.last_decision = CounterDecision(
            frame_idx=signal.frame_idx,
            time_sec=signal.time_sec,
            accepted=True,
            reason="accepted",
            side=side,
            hip_range_ratio=metrics["hip_range_ratio"],
            recent_hip_range_ratio=metrics["recent_hip_range_ratio"],
            support_ratio=support_ratio,
            hip_motion_ratio=hip_motion_ratio,
            hip_velocity_ratio=hip_velocity_ratio,
            min_gap_frames=min_gap_frames,
            cadence_locked=cadence_locked,
        )
        return CounterEvent(
            frame_idx=signal.frame_idx,
            time_sec=signal.time_sec,
            running_count=self.accepted_running_count,
            count_delta=count_delta,
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
        return ground_truth_events[0].frame_idx, ground_truth_events[-1].frame_idx

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

    warmup_span = max(warmup_frames, config.motion_window_frames)
    start_index = max(0, start_frame - warmup_span)
    counting_armed = False
    for signal in signals[start_index:]:
        if signal.frame_idx > effective_end_frame:
            break
        if signal.frame_idx < start_frame:
            engine.warmup(signal)
            continue
        if not counting_armed:
            engine.active_side = None
            engine.candidate_side = None
            engine.candidate_streak = 0
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
                predicted_frames=[
                    event.frame_idx
                    for event in events
                    for _ in range(event.count_delta)
                ],
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
            side_diff_ratio=values[0],
            min_foot_diff_ratio=values[1],
            min_hip_motion_ratio=values[2],
            min_recent_hip_range_ratio=values[3],
            descend_velocity_ratio=values[4],
            min_count_gap_frames=values[5],
            support_streak_frames=values[6],
        )
        for values in product(
            [0.03, 0.04],
            [0.04, 0.045, 0.05],
            [0.0],
            [0.0],
            [-1.0],
            [1],
            [1],
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
