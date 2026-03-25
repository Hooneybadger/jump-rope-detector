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
    arm_visibility_threshold: float = 0.30
    foot_cutoff_hz: float = 4.0
    hip_cutoff_hz: float = 3.0
    wrist_speed_cutoff_hz: float = 5.5
    shoulder_blend_alpha: float = 0.30
    floor_decay_ratio: float = 0.003
    contact_margin_ratio: float = 0.08
    symmetry_y_ratio: float = 0.14
    min_count_gap_frames: int = 8
    min_airborne_frames: int = 6
    max_airborne_frames: int = 32
    min_jump_height_ratio: float = 0.11
    min_hip_lift_ratio: float = 0.06
    min_long_airborne_height_ratio: float = 0.08
    long_airborne_frames: int = 10
    min_wrist_peak_count: int = 2
    min_wrist_peak_speed_ratio: float = 0.85
    min_wrist_energy_ratio: float = 0.42
    wrist_peak_refractory_frames: int = 2
    fft_window_frames: int = 48
    min_fft_peak_hz: float = 3.2
    min_fft_power_ratio: float = 0.34
    min_fft_wrist_to_jump_ratio: float = 1.55
    adaptive_gap_enabled: bool = True
    adaptive_gap_factor: float = 0.72
    adaptive_gap_history: int = 5
    adaptive_gap_min_intervals: int = 2
    adaptive_gap_floor_frames: int = 6
    search_wrist_peak_count: int = 2

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
    min_gap_frames: int
    cadence_locked: bool


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
    left_wrist = _visible_xy(lms, mp_pose.PoseLandmark.LEFT_WRIST, config.arm_visibility_threshold)
    right_wrist = _visible_xy(lms, mp_pose.PoseLandmark.RIGHT_WRIST, config.arm_visibility_threshold)
    if None in {left_shoulder, right_shoulder, left_elbow, right_elbow, left_wrist, right_wrist}:
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
        left_shoulder_x=left_shoulder[0],
        left_shoulder_y=left_shoulder[1],
        right_shoulder_x=right_shoulder[0],
        right_shoulder_y=right_shoulder[1],
        left_elbow_x=left_elbow[0],
        left_elbow_y=left_elbow[1],
        right_elbow_x=right_elbow[0],
        right_elbow_y=right_elbow[1],
        left_wrist_x=left_wrist[0],
        left_wrist_y=left_wrist[1],
        right_wrist_x=right_wrist[0],
        right_wrist_y=right_wrist[1],
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


class RealtimeCounterEngine:
    def __init__(self, config: EngineConfig):
        self.config = config
        self.hip_filter = _ButterworthLowPass(config.hip_cutoff_hz)
        self.foot_filter = _ButterworthLowPass(config.foot_cutoff_hz)
        self.wrist_speed_filter = _ButterworthLowPass(config.wrist_speed_cutoff_hz)
        self.prev_time_sec: float | None = None
        self.prev_left_wrist: tuple[float, float] | None = None
        self.prev_right_wrist: tuple[float, float] | None = None
        self.prev_shoulder_center: tuple[float, float] | None = None
        self.floor_y: float | None = None
        self.hip_ground_baseline: float | None = None
        self.in_air = False
        self.air_start_frame: int | None = None
        self.air_start_time: float | None = None
        self.air_peak_height_ratio = 0.0
        self.air_peak_hip_ratio = 0.0
        self.air_peak_frames: list[int] = []
        self.air_peak_speeds: list[float] = []
        self.last_wrist_peak_frame: int | None = None
        self.last_speed_1: float | None = None
        self.last_speed_2: float | None = None
        self.last_accepted_frame: int | None = None
        self.accepted_running_count = 0
        self.jump_interval_history: deque[int] = deque(maxlen=config.adaptive_gap_history)
        self.foot_height_history: deque[float] = deque(maxlen=config.fft_window_frames)
        self.hip_lift_history: deque[float] = deque(maxlen=config.fft_window_frames)
        self.wrist_speed_history: deque[float] = deque(maxlen=config.fft_window_frames)
        self.time_delta_history: deque[float] = deque(maxlen=config.fft_window_frames)
        self.last_decision: CounterDecision | None = None

    @staticmethod
    def _ema(alpha: float, value: float, previous: float | None) -> float:
        if previous is None:
            return value
        return alpha * value + (1.0 - alpha) * previous

    def _sample_rate(self, timestamp_sec: float) -> tuple[float, float]:
        if self.prev_time_sec is None:
            return 30.0, 1.0 / 30.0
        dt = max(1e-3, timestamp_sec - self.prev_time_sec)
        self.time_delta_history.append(dt)
        fs = 1.0 / dt
        return fs, dt

    def _wrist_speed_ratio(self, signal: SignalFrame, dt: float, leg_length: float) -> float:
        assert signal.left_wrist_x is not None
        assert signal.left_wrist_y is not None
        assert signal.right_wrist_x is not None
        assert signal.right_wrist_y is not None
        assert signal.left_shoulder_x is not None
        assert signal.left_shoulder_y is not None
        assert signal.right_shoulder_x is not None
        assert signal.right_shoulder_y is not None

        shoulder_center = (
            (signal.left_shoulder_x + signal.right_shoulder_x) / 2.0,
            (signal.left_shoulder_y + signal.right_shoulder_y) / 2.0,
        )
        shoulder_motion = (0.0, 0.0)
        if self.prev_shoulder_center is not None:
            shoulder_motion = (
                shoulder_center[0] - self.prev_shoulder_center[0],
                shoulder_center[1] - self.prev_shoulder_center[1],
            )

        left_wrist = (signal.left_wrist_x, signal.left_wrist_y)
        right_wrist = (signal.right_wrist_x, signal.right_wrist_y)
        left_speed = 0.0
        right_speed = 0.0
        if self.prev_left_wrist is not None:
            left_dx = (left_wrist[0] - self.prev_left_wrist[0]) - shoulder_motion[0]
            left_dy = (left_wrist[1] - self.prev_left_wrist[1]) - shoulder_motion[1]
            left_speed = math.hypot(left_dx, left_dy) / dt
        if self.prev_right_wrist is not None:
            right_dx = (right_wrist[0] - self.prev_right_wrist[0]) - shoulder_motion[0]
            right_dy = (right_wrist[1] - self.prev_right_wrist[1]) - shoulder_motion[1]
            right_speed = math.hypot(right_dx, right_dy) / dt

        self.prev_left_wrist = left_wrist
        self.prev_right_wrist = right_wrist
        self.prev_shoulder_center = shoulder_center
        return ((left_speed + right_speed) / 2.0) / max(leg_length, 1e-6)

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
        if float(np.sum(band_power)) <= 0.0:
            return 0.0, 0.0
        peak_index = int(np.argmax(band_power))
        peak_power = float(band_power[peak_index])
        peak_freq = float(freqs[band_mask][peak_index])
        power_ratio = peak_power / float(np.sum(band_power))
        return peak_freq, power_ratio

    def _effective_gap(self) -> tuple[int, bool]:
        min_gap_frames = self.config.min_count_gap_frames
        cadence_locked = False
        if (
            self.config.adaptive_gap_enabled
            and len(self.jump_interval_history) >= self.config.adaptive_gap_min_intervals
        ):
            cadence_locked = True
            interval_median = median(self.jump_interval_history)
            min_gap_frames = max(
                self.config.adaptive_gap_floor_frames,
                min(
                    self.config.min_count_gap_frames,
                    int(round(interval_median * self.config.adaptive_gap_factor)),
                ),
            )
        return min_gap_frames, cadence_locked

    def _set_reject(
        self,
        signal: SignalFrame,
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
        min_gap_frames: int,
        cadence_locked: bool,
    ) -> None:
        self.last_decision = CounterDecision(
            frame_idx=signal.frame_idx,
            time_sec=signal.time_sec,
            accepted=False,
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
            min_gap_frames=min_gap_frames,
            cadence_locked=cadence_locked,
        )

    def _reset_air_phase(self) -> None:
        self.in_air = False
        self.air_start_frame = None
        self.air_start_time = None
        self.air_peak_height_ratio = 0.0
        self.air_peak_hip_ratio = 0.0
        self.air_peak_frames.clear()
        self.air_peak_speeds.clear()

    def _observe_peak(self, frame_idx: int) -> None:
        if self.last_speed_2 is None or self.last_speed_1 is None:
            return
        current = self.wrist_speed_history[-1]
        if self.last_speed_1 < self.config.min_wrist_peak_speed_ratio:
            return
        if self.last_speed_2 > self.last_speed_1 or current > self.last_speed_1:
            return
        if (
            self.last_wrist_peak_frame is not None
            and (frame_idx - self.last_wrist_peak_frame) < self.config.wrist_peak_refractory_frames
        ):
            return
        peak_frame = frame_idx - 1
        self.air_peak_frames.append(peak_frame)
        self.air_peak_speeds.append(self.last_speed_1)
        self.last_wrist_peak_frame = peak_frame

    def warmup(self, signal: SignalFrame) -> None:
        self.last_decision = None
        self._step_internal(signal, allow_count=False)

    def step(self, signal: SignalFrame) -> CounterEvent | None:
        self.last_decision = None
        return self._step_internal(signal, allow_count=True)

    def _step_internal(self, signal: SignalFrame, allow_count: bool) -> CounterEvent | None:
        if not signal.detected:
            self._reset_air_phase()
            self.prev_time_sec = signal.time_sec
            return None

        mean_hip_y_raw, mean_foot_y_raw, leg_length = _mean_pose_values(signal)
        fs, dt = self._sample_rate(signal.time_sec)
        filtered_hip_y = self.hip_filter.filter(mean_hip_y_raw, fs)
        filtered_foot_y = self.foot_filter.filter(mean_foot_y_raw, fs)
        wrist_speed_ratio_raw = self._wrist_speed_ratio(signal, dt, leg_length)
        wrist_speed_ratio = self.wrist_speed_filter.filter(wrist_speed_ratio_raw, fs)

        symmetry_y_ratio = abs(signal.left_foot_y - signal.right_foot_y) / max(leg_length, 1e-6)
        if self.floor_y is None:
            self.floor_y = filtered_foot_y
        else:
            self.floor_y = max(filtered_foot_y, self.floor_y - (self.config.floor_decay_ratio * leg_length))

        contact_gate = (
            filtered_foot_y >= (self.floor_y - (self.config.contact_margin_ratio * leg_length))
            and symmetry_y_ratio <= self.config.symmetry_y_ratio
        )
        if contact_gate:
            self.hip_ground_baseline = self._ema(
                self.config.shoulder_blend_alpha,
                filtered_hip_y,
                self.hip_ground_baseline,
            )
        if self.hip_ground_baseline is None:
            self.hip_ground_baseline = filtered_hip_y

        jump_height_ratio = max(0.0, (self.floor_y - filtered_foot_y) / max(leg_length, 1e-6))
        hip_lift_ratio = max(0.0, (self.hip_ground_baseline - filtered_hip_y) / max(leg_length, 1e-6))

        self.foot_height_history.append(jump_height_ratio)
        self.hip_lift_history.append(hip_lift_ratio)
        self.wrist_speed_history.append(wrist_speed_ratio)
        self._observe_peak(signal.frame_idx)
        self.last_speed_2 = self.last_speed_1
        self.last_speed_1 = wrist_speed_ratio

        if not self.in_air:
            if not contact_gate and jump_height_ratio >= (self.config.min_long_airborne_height_ratio * 0.7):
                self.in_air = True
                self.air_start_frame = signal.frame_idx
                self.air_start_time = signal.time_sec
                self.air_peak_height_ratio = jump_height_ratio
                self.air_peak_hip_ratio = hip_lift_ratio
                self.air_peak_frames.clear()
                self.air_peak_speeds.clear()
                self.last_wrist_peak_frame = None
            self.prev_time_sec = signal.time_sec
            return None

        self.air_peak_height_ratio = max(self.air_peak_height_ratio, jump_height_ratio)
        self.air_peak_hip_ratio = max(self.air_peak_hip_ratio, hip_lift_ratio)

        if not contact_gate:
            if self.air_start_frame is not None:
                airtime_frames = signal.frame_idx - self.air_start_frame + 1
                if airtime_frames > self.config.max_airborne_frames:
                    self._reset_air_phase()
            self.prev_time_sec = signal.time_sec
            return None

        if self.air_start_frame is None:
            self._reset_air_phase()
            self.prev_time_sec = signal.time_sec
            return None

        airtime_frames = signal.frame_idx - self.air_start_frame
        wrist_peak_count = len(self.air_peak_frames)
        wrist_peak_speed_ratio = max(self.air_peak_speeds, default=0.0)
        wrist_fft_peak_hz, wrist_fft_power_ratio = self._dominant_frequency(
            self.wrist_speed_history,
            min_hz=2.0,
            max_hz=12.0,
        )
        jump_fft_hz, _ = self._dominant_frequency(
            self.foot_height_history,
            min_hz=0.8,
            max_hz=4.0,
        )
        wrist_to_jump_ratio = wrist_fft_peak_hz / max(jump_fft_hz, 1e-6) if jump_fft_hz > 0 else 0.0
        wrist_energy_ratio = float(np.mean(self.air_peak_speeds)) if self.air_peak_speeds else 0.0
        min_gap_frames, cadence_locked = self._effective_gap()

        if not allow_count:
            self._reset_air_phase()
            self.prev_time_sec = signal.time_sec
            return None
        if (
            self.last_accepted_frame is not None
            and (signal.frame_idx - self.last_accepted_frame) < min_gap_frames
        ):
            self._set_reject(
                signal,
                "min_gap",
                airtime_frames,
                self.air_peak_height_ratio,
                self.air_peak_hip_ratio,
                wrist_peak_count,
                wrist_peak_speed_ratio,
                wrist_energy_ratio,
                wrist_fft_peak_hz,
                wrist_fft_power_ratio,
                wrist_to_jump_ratio,
                min_gap_frames,
                cadence_locked,
            )
            self._reset_air_phase()
            self.prev_time_sec = signal.time_sec
            return None
        if airtime_frames < self.config.min_airborne_frames:
            self._set_reject(
                signal,
                "airtime_short",
                airtime_frames,
                self.air_peak_height_ratio,
                self.air_peak_hip_ratio,
                wrist_peak_count,
                wrist_peak_speed_ratio,
                wrist_energy_ratio,
                wrist_fft_peak_hz,
                wrist_fft_power_ratio,
                wrist_to_jump_ratio,
                min_gap_frames,
                cadence_locked,
            )
            self._reset_air_phase()
            self.prev_time_sec = signal.time_sec
            return None
        if (
            self.air_peak_height_ratio < self.config.min_jump_height_ratio
            and not (
                airtime_frames >= self.config.long_airborne_frames
                and self.air_peak_height_ratio >= self.config.min_long_airborne_height_ratio
            )
        ):
            self._set_reject(
                signal,
                "jump_height",
                airtime_frames,
                self.air_peak_height_ratio,
                self.air_peak_hip_ratio,
                wrist_peak_count,
                wrist_peak_speed_ratio,
                wrist_energy_ratio,
                wrist_fft_peak_hz,
                wrist_fft_power_ratio,
                wrist_to_jump_ratio,
                min_gap_frames,
                cadence_locked,
            )
            self._reset_air_phase()
            self.prev_time_sec = signal.time_sec
            return None
        if self.air_peak_hip_ratio < self.config.min_hip_lift_ratio:
            self._set_reject(
                signal,
                "hip_lift",
                airtime_frames,
                self.air_peak_height_ratio,
                self.air_peak_hip_ratio,
                wrist_peak_count,
                wrist_peak_speed_ratio,
                wrist_energy_ratio,
                wrist_fft_peak_hz,
                wrist_fft_power_ratio,
                wrist_to_jump_ratio,
                min_gap_frames,
                cadence_locked,
            )
            self._reset_air_phase()
            self.prev_time_sec = signal.time_sec
            return None
        wrist_peak_ok = (
            wrist_peak_count >= self.config.min_wrist_peak_count
            and wrist_peak_speed_ratio >= self.config.min_wrist_peak_speed_ratio
        )
        wrist_fft_ok = (
            wrist_fft_peak_hz >= self.config.min_fft_peak_hz
            and wrist_fft_power_ratio >= self.config.min_fft_power_ratio
            and wrist_to_jump_ratio >= self.config.min_fft_wrist_to_jump_ratio
        )
        if not wrist_peak_ok and not wrist_fft_ok:
            self._set_reject(
                signal,
                "wrist_rotation",
                airtime_frames,
                self.air_peak_height_ratio,
                self.air_peak_hip_ratio,
                wrist_peak_count,
                wrist_peak_speed_ratio,
                wrist_energy_ratio,
                wrist_fft_peak_hz,
                wrist_fft_power_ratio,
                wrist_to_jump_ratio,
                min_gap_frames,
                cadence_locked,
            )
            self._reset_air_phase()
            self.prev_time_sec = signal.time_sec
            return None
        if wrist_energy_ratio < self.config.min_wrist_energy_ratio:
            self._set_reject(
                signal,
                "wrist_energy",
                airtime_frames,
                self.air_peak_height_ratio,
                self.air_peak_hip_ratio,
                wrist_peak_count,
                wrist_peak_speed_ratio,
                wrist_energy_ratio,
                wrist_fft_peak_hz,
                wrist_fft_power_ratio,
                wrist_to_jump_ratio,
                min_gap_frames,
                cadence_locked,
            )
            self._reset_air_phase()
            self.prev_time_sec = signal.time_sec
            return None

        self.accepted_running_count += 1
        if self.last_accepted_frame is not None:
            self.jump_interval_history.append(signal.frame_idx - self.last_accepted_frame)
        self.last_accepted_frame = signal.frame_idx
        self.last_decision = CounterDecision(
            frame_idx=signal.frame_idx,
            time_sec=signal.time_sec,
            accepted=True,
            reason="accepted",
            airtime_frames=airtime_frames,
            jump_height_ratio=self.air_peak_height_ratio,
            hip_lift_ratio=self.air_peak_hip_ratio,
            wrist_peak_count=wrist_peak_count,
            wrist_peak_speed_ratio=wrist_peak_speed_ratio,
            wrist_energy_ratio=wrist_energy_ratio,
            wrist_fft_peak_hz=wrist_fft_peak_hz,
            wrist_fft_power_ratio=wrist_fft_power_ratio,
            wrist_to_jump_ratio=wrist_to_jump_ratio,
            min_gap_frames=min_gap_frames,
            cadence_locked=cadence_locked,
        )
        event = CounterEvent(
            frame_idx=signal.frame_idx,
            time_sec=signal.time_sec,
            running_count=self.accepted_running_count,
            count_delta=1,
        )
        self._reset_air_phase()
        self.prev_time_sec = signal.time_sec
        return event


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
            min_airborne_frames=values[0],
            min_jump_height_ratio=values[1],
            min_hip_lift_ratio=values[2],
            min_wrist_peak_speed_ratio=values[3],
            min_fft_power_ratio=values[4],
            min_count_gap_frames=values[5],
        )
        for values in product(
            [6, 7, 8],
            [0.10, 0.11, 0.12],
            [0.05, 0.06, 0.07],
            [0.75, 0.85, 0.95],
            [0.28, 0.34, 0.40],
            [8, 9, 10],
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
