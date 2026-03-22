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


def full_landmarks_visible(result, visibility_threshold: float = 0.30) -> bool:
    if not result.pose_landmarks:
        return False
    lms = result.pose_landmarks.landmark
    return all(float(lms[landmark.value].visibility) >= visibility_threshold for landmark in ALL_POSE_LANDMARKS)


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

    def step(self, signal: SignalFrame) -> CounterEvent | None:
        if not signal.detected:
            return None

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

        contact_threshold = self.floor_y - (self.config.contact_margin_ratio * signal.leg_length)
        symmetry_y = abs(signal.left_foot_y - signal.right_foot_y)
        contact_gate = mean_foot_y >= contact_threshold and symmetry_y <= (
            self.config.symmetry_y_ratio * signal.leg_length
        )
        descend_ratio = self.config.fast_descend_velocity_ratio if fast_mode else self.config.descend_velocity_ratio
        ascend_ratio = self.config.fast_ascend_velocity_ratio if fast_mode else self.config.ascend_velocity_ratio
        min_refractory = self.config.fast_min_refractory_frames if fast_mode else self.config.min_refractory_frames
        descending = hip_vel >= (descend_ratio * signal.leg_length)
        ascending = hip_vel <= -(ascend_ratio * signal.leg_length)
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


class RealtimeStartGate:
    def __init__(self, ready_hold_seconds: float = 1.0, countdown_seconds: float = 3.0):
        self.ready_hold_seconds = ready_hold_seconds
        self.countdown_seconds = countdown_seconds
        self.phase = "SEARCHING"
        self.ready_since_sec: float | None = None
        self.countdown_end_sec: float | None = None
        self.count_started_at_sec: float | None = None

    def update(self, full_body_ready: bool, timestamp_sec: float) -> StreamState:
        if self.phase == "SEARCHING":
            if full_body_ready:
                if self.ready_since_sec is None:
                    self.ready_since_sec = timestamp_sec
                if (timestamp_sec - self.ready_since_sec) >= self.ready_hold_seconds:
                    self.phase = "COUNTDOWN"
                    self.countdown_end_sec = timestamp_sec + self.countdown_seconds
            else:
                self.ready_since_sec = None
        elif self.phase == "COUNTDOWN":
            if not full_body_ready:
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

    start_index = max(0, start_frame - warmup_frames)
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
    summary = {
        "exact_video_count_accuracy": exact_matches / len(results) if results else 0.0,
        "videos": [asdict(result) for result in results],
        "total_abs_error": sum(abs(result.count_error) for result in results),
    }
    return summary


def default_search_configs(limit: int | None = None) -> list[EngineConfig]:
    configs = [
        EngineConfig(
            ema_alpha_hip=values[0],
            ema_alpha_foot=values[1],
            fast_ema_alpha_hip=values[2],
            fast_ema_alpha_foot=values[3],
            contact_margin_ratio=values[4],
            symmetry_y_ratio=values[5],
            descend_velocity_ratio=values[6],
            ascend_velocity_ratio=values[7],
            fast_descend_velocity_ratio=values[8],
            fast_ascend_velocity_ratio=values[9],
            floor_decay_ratio=values[10],
            min_refractory_frames=values[11],
            fast_min_refractory_frames=values[12],
            fast_mode_cadence_threshold=values[13],
        )
        for values in product(
            [0.45, 0.55],
            [0.55],
            [0.20, 0.25],
            [0.35],
            [0.08],
            [0.10, 0.12],
            [0.006, 0.008],
            [0.004, 0.006],
            [0.002, 0.003],
            [0.001],
            [0.004],
            [4],
            [1, 2],
            [7],
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
            summary["exact_video_count_accuracy"] > best_summary["exact_video_count_accuracy"]
            or (
                summary["exact_video_count_accuracy"] == best_summary["exact_video_count_accuracy"]
                and summary["total_abs_error"] < best_summary["total_abs_error"]
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
