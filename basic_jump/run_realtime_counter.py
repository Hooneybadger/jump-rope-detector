from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass
import sys
import time
from pathlib import Path
from statistics import median

import cv2
import mediapipe as mp

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from basic_jump.counter_engine import (
    EngineConfig,
    PoseSignalExtractor,
    RealtimeCounterEngine,
    RealtimeStartGate,
    full_landmarks_visible,
)


mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose


@dataclass(frozen=True)
class RealtimeFilterConfig:
    motion_window_frames: int = 18
    recent_window_frames: int = 5
    min_hip_range_ratio: float = 0.10
    min_foot_range_ratio: float = 0.065
    min_recent_hip_range_ratio: float = 0.05
    min_count_gap_frames: int = 20


class RealtimeCountFilter:
    def __init__(self, config: RealtimeFilterConfig):
        self.config = config
        self.hip_history: deque[float] = deque(maxlen=config.motion_window_frames)
        self.foot_history: deque[float] = deque(maxlen=config.motion_window_frames)
        self.leg_history: deque[float] = deque(maxlen=config.motion_window_frames)
        self.recent_hip_history: deque[float] = deque(maxlen=config.recent_window_frames)
        self.last_accepted_frame: int | None = None

    def update(self, signal) -> None:
        if not signal.detected:
            return
        mean_hip = (signal.left_hip_y + signal.right_hip_y) / 2.0
        mean_foot = (signal.left_foot_y + signal.right_foot_y) / 2.0
        self.hip_history.append(mean_hip)
        self.foot_history.append(mean_foot)
        self.leg_history.append(signal.leg_length)
        self.recent_hip_history.append(mean_hip)

    def metrics(self) -> dict[str, float]:
        if not self.hip_history or not self.foot_history or not self.leg_history:
            return {"hip_range_ratio": 0.0, "foot_range_ratio": 0.0, "recent_hip_range_ratio": 0.0}
        leg_median = median(self.leg_history)
        hip_range_ratio = (max(self.hip_history) - min(self.hip_history)) / leg_median
        foot_range_ratio = (max(self.foot_history) - min(self.foot_history)) / leg_median
        recent_hip_range_ratio = 0.0
        if len(self.recent_hip_history) >= self.config.recent_window_frames:
            recent_hip_range_ratio = (
                max(self.recent_hip_history) - min(self.recent_hip_history)
            ) / leg_median
        return {
            "hip_range_ratio": hip_range_ratio,
            "foot_range_ratio": foot_range_ratio,
            "recent_hip_range_ratio": recent_hip_range_ratio,
        }

    def accept(self, frame_idx: int) -> tuple[bool, str]:
        if len(self.hip_history) < max(3, self.config.motion_window_frames // 2):
            return False, "insufficient_window"
        if (
            self.last_accepted_frame is not None
            and (frame_idx - self.last_accepted_frame) < self.config.min_count_gap_frames
        ):
            return False, "min_gap"

        metrics = self.metrics()
        if metrics["hip_range_ratio"] < self.config.min_hip_range_ratio:
            return False, "hip_range"
        if metrics["foot_range_ratio"] < self.config.min_foot_range_ratio:
            return False, "foot_range"
        if metrics["recent_hip_range_ratio"] < self.config.min_recent_hip_range_ratio:
            return False, "recent_hip_range"

        self.last_accepted_frame = frame_idx
        return True, "accepted"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the grouped-jump counter on a realtime stream.")
    parser.add_argument("--source", default="0", help="Camera index like `0` or a video file path.")
    parser.add_argument("--save-output", default=None, help="Optional output video path for saving the realtime demo.")
    parser.add_argument("--no-display", action="store_true", help="Disable OpenCV window rendering.")
    parser.add_argument("--max-frames", type=int, default=None, help="Optional frame limit for smoke tests.")
    parser.add_argument("--ready-hold-seconds", type=float, default=1.0)
    parser.add_argument("--countdown-seconds", type=float, default=3.0)
    parser.add_argument("--ready-visibility-threshold", type=float, default=0.30)
    parser.add_argument("--motion-window-frames", type=int, default=18)
    parser.add_argument("--min-hip-range-ratio", type=float, default=0.10)
    parser.add_argument("--min-foot-range-ratio", type=float, default=0.065)
    parser.add_argument("--min-recent-hip-range-ratio", type=float, default=0.05)
    parser.add_argument("--min-count-gap-frames", type=int, default=20)
    parser.add_argument("--debug-filter", action="store_true", help="Print rejected count candidates and reasons.")
    return parser.parse_args()


def _open_capture(source: str) -> tuple[cv2.VideoCapture, bool, str]:
    if source.isdigit():
        return cv2.VideoCapture(int(source)), True, f"camera:{source}"
    path = Path(source)
    return cv2.VideoCapture(str(path)), False, str(path)


def _phase_color(phase: str) -> tuple[int, int, int]:
    if phase == "COUNTING":
        return (90, 190, 120)
    if phase == "COUNTDOWN":
        return (90, 170, 230)
    return (150, 150, 170)


def _draw_panel(frame, top_left: tuple[int, int], bottom_right: tuple[int, int], color: tuple[int, int, int], alpha: float) -> None:
    overlay = frame.copy()
    cv2.rectangle(overlay, top_left, bottom_right, color, thickness=-1)
    cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0.0, dst=frame)


def _draw_text(
    frame,
    text: str,
    origin: tuple[int, int],
    scale: float,
    color: tuple[int, int, int],
    thickness: int,
) -> None:
    cv2.putText(frame, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, (12, 12, 12), thickness + 2, cv2.LINE_AA)
    cv2.putText(frame, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def _draw_progress_bar(
    frame,
    top_left: tuple[int, int],
    size: tuple[int, int],
    progress: float,
    fill_color: tuple[int, int, int],
) -> None:
    x, y = top_left
    w, h = size
    progress = max(0.0, min(1.0, progress))
    _draw_panel(frame, (x, y), (x + w, y + h), (30, 30, 30), 0.55)
    fill_w = max(0, int(w * progress))
    if fill_w > 0:
        cv2.rectangle(frame, (x, y), (x + fill_w, y + h), fill_color, thickness=-1)
    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), thickness=1)


def _ease_out(progress: float) -> float:
    progress = max(0.0, min(1.0, progress))
    return 1.0 - ((1.0 - progress) * (1.0 - progress))


def _draw_overlay(
    frame,
    stream_state,
    running_count: int,
    full_body_ready: bool,
    countdown_total_sec: float,
    count_pulse_progress: float,
) -> None:
    height, width = frame.shape[:2]
    accent = _phase_color(stream_state.phase)
    pulse = _ease_out(count_pulse_progress)

    panel_x, panel_y = 18, 18
    panel_w, panel_h = min(260, width - 36), 108
    _draw_panel(frame, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), (18, 22, 28), 0.58)
    cv2.line(frame, (panel_x + 14, panel_y + 16), (panel_x + 14, panel_y + panel_h - 16), accent, 4)

    if pulse > 0.0:
        flash_color = tuple(int((1.0 - pulse) * 255 + pulse * channel) for channel in accent)
        cv2.rectangle(
            frame,
            (panel_x - 1, panel_y - 1),
            (panel_x + panel_w + 1, panel_y + panel_h + 1),
            flash_color,
            thickness=2,
        )

    _draw_text(frame, "Count", (panel_x + 32, panel_y + 30), 0.58, (210, 220, 230), 1)
    count_scale = 1.35 + (0.20 * pulse)
    count_thickness = 2 + int(round(pulse))
    count_color = (245, 245, 245) if pulse <= 0.0 else (255, 255, 255)
    _draw_text(frame, str(running_count), (panel_x + 32, panel_y + 72), count_scale, count_color, count_thickness)
    _draw_text(frame, stream_state.phase.lower(), (panel_x + 120, panel_y + 34), 0.58, accent, 1)
    status_line = "Full body ready" if full_body_ready else "Align full body"
    _draw_text(frame, status_line, (panel_x + 120, panel_y + 60), 0.54, (220, 220, 220), 1)

    if stream_state.phase == "SEARCHING":
        progress_value = stream_state.ready_progress
    elif stream_state.phase == "COUNTDOWN":
        progress_value = 1.0 - min(1.0, stream_state.countdown_remaining_sec / countdown_total_sec)
    else:
        progress_value = 1.0
    _draw_progress_bar(frame, (panel_x + 32, panel_y + 84), (panel_w - 52, 8), progress_value, accent)

    if stream_state.phase == "COUNTDOWN":
        countdown_number = max(1, int(stream_state.countdown_remaining_sec) + 1)
        center_box_w, center_box_h = 144, 88
        center_x = (width - center_box_w) // 2
        center_y = (height - center_box_h) // 2
        _draw_panel(frame, (center_x, center_y), (center_x + center_box_w, center_y + center_box_h), (18, 22, 28), 0.55)
        _draw_text(frame, f"Starting in {countdown_number}", (center_x + 16, center_y + 52), 0.9, (245, 245, 245), 2)


def _create_video_writer(path: str | Path, fps: float, frame_shape) -> cv2.VideoWriter:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    height, width = frame_shape[:2]
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps if fps > 0 else 30.0,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Unable to open output video for writing: {path}")
    return writer


def main() -> None:
    args = parse_args()
    capture, is_camera, source_label = _open_capture(args.source)
    if not capture.isOpened():
        raise RuntimeError(f"Unable to open stream source: {args.source}")

    fps = float(capture.get(cv2.CAP_PROP_FPS) or 30.0)
    config = EngineConfig()
    extractor = PoseSignalExtractor(config)
    gate = RealtimeStartGate(
        ready_hold_seconds=args.ready_hold_seconds,
        countdown_seconds=args.countdown_seconds,
    )
    engine: RealtimeCounterEngine | None = None
    count_filter = RealtimeCountFilter(
        RealtimeFilterConfig(
            motion_window_frames=args.motion_window_frames,
            min_hip_range_ratio=args.min_hip_range_ratio,
            min_foot_range_ratio=args.min_foot_range_ratio,
            min_recent_hip_range_ratio=args.min_recent_hip_range_ratio,
            min_count_gap_frames=args.min_count_gap_frames,
        )
    )
    writer: cv2.VideoWriter | None = None
    accepted_count = 0
    count_pulse_total_frames = 10
    count_pulse_remaining_frames = 0

    frame_idx = 0
    stream_start_sec = time.monotonic()
    last_phase = gate.phase
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if args.max_frames is not None and frame_idx >= args.max_frames:
                break

            timestamp_sec = (time.monotonic() - stream_start_sec) if is_camera else (frame_idx / fps if fps > 0 else 0.0)
            signal, result = extractor.process_bgr_frame(frame, frame_idx, timestamp_sec)
            full_body_ready = full_landmarks_visible(result, args.ready_visibility_threshold)
            stream_state = gate.update(full_body_ready, timestamp_sec)

            if stream_state.phase != last_phase:
                print(f"[phase] {last_phase} -> {stream_state.phase} @ {timestamp_sec:.2f}s")
                last_phase = stream_state.phase
                if stream_state.phase == "COUNTING":
                    engine = RealtimeCounterEngine(config)
                    count_filter = RealtimeCountFilter(count_filter.config)
                    accepted_count = 0
                    print("[count] counter armed")

            if stream_state.phase == "COUNTING" and engine is not None:
                count_filter.update(signal)
                event = engine.step(signal)
                if event is not None:
                    accepted, reason = count_filter.accept(event.frame_idx)
                    if accepted:
                        accepted_count += 1
                        count_pulse_remaining_frames = count_pulse_total_frames
                        print(f"[count] {accepted_count} @ frame={event.frame_idx} time={event.time_sec:.2f}s")
                    elif args.debug_filter:
                        metrics = count_filter.metrics()
                        print(
                            f"[reject] frame={event.frame_idx} reason={reason} "
                            f"hip_range={metrics['hip_range_ratio']:.3f} "
                            f"foot_range={metrics['foot_range_ratio']:.3f} "
                            f"recent_hip={metrics['recent_hip_range_ratio']:.3f}"
                        )

            if not args.no_display or args.save_output:
                display_frame = frame.copy()
                if result.pose_landmarks:
                    mp_drawing.draw_landmarks(
                        display_frame,
                        result.pose_landmarks,
                        mp_pose.POSE_CONNECTIONS,
                        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style(),
                    )
                count_pulse_progress = (
                    count_pulse_remaining_frames / count_pulse_total_frames
                    if count_pulse_total_frames > 0
                    else 0.0
                )
                _draw_overlay(
                    display_frame,
                    stream_state,
                    accepted_count,
                    full_body_ready,
                    args.countdown_seconds,
                    count_pulse_progress,
                )

                if args.save_output:
                    if writer is None:
                        writer = _create_video_writer(args.save_output, fps, display_frame.shape)
                    writer.write(display_frame)

            if not args.no_display:
                cv2.imshow("basic_jump realtime counter", display_frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break

            if count_pulse_remaining_frames > 0:
                count_pulse_remaining_frames -= 1
            frame_idx += 1
    finally:
        extractor.close()
        capture.release()
        if writer is not None:
            writer.release()
        if not args.no_display:
            cv2.destroyAllWindows()

    final_count = accepted_count
    print(f"[done] source={source_label} frames={frame_idx} final_count={final_count}")
    if args.save_output:
        print(f"[saved] {args.save_output}")


if __name__ == "__main__":
    main()
