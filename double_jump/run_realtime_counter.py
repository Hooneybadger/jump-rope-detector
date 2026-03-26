from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import mediapipe as mp

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from double_jump.counter_engine import (
    EngineConfig,
    PoseSignalExtractor,
    RealtimeCounterEngine,
    RealtimeStartGate,
    core_landmarks_visible,
)


mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the double-jump counter on a realtime stream.")
    parser.add_argument("--source", default="0", help="Camera index like `0` or a video file path.")
    parser.add_argument("--save-output", default=None, help="Optional output video path for saving the realtime demo.")
    parser.add_argument("--no-display", action="store_true", help="Disable OpenCV window rendering.")
    parser.add_argument("--max-frames", type=int, default=None, help="Optional frame limit for smoke tests.")
    parser.add_argument("--ready-hold-seconds", type=float, default=1.0)
    parser.add_argument("--countdown-seconds", type=float, default=3.0)
    parser.add_argument("--ready-visibility-threshold", type=float, default=0.30)
    parser.add_argument("--ready-visible-ratio", type=float, default=0.80)
    parser.add_argument("--ready-dropout-seconds", type=float, default=0.35)
    parser.add_argument("--min-airborne-frames", type=int, default=EngineConfig.min_airborne_frames)
    parser.add_argument("--min-jump-height-ratio", type=float, default=EngineConfig.min_jump_height_ratio)
    parser.add_argument("--min-hip-lift-ratio", type=float, default=EngineConfig.min_hip_lift_ratio)
    parser.add_argument("--min-snap-peak-ratio", type=float, default=EngineConfig.min_snap_peak_ratio)
    parser.add_argument("--min-snap-energy-ratio", type=float, default=EngineConfig.min_snap_energy_ratio)
    parser.add_argument("--min-rotation-energy-ratio", type=float, default=EngineConfig.min_rotation_energy_ratio)
    parser.add_argument("--min-count-gap-frames", type=int, default=EngineConfig.min_count_gap_frames)
    parser.add_argument("--disable-adaptive-filter", action="store_true", help="Disable cadence-adaptive realtime thresholds.")
    parser.add_argument("--debug-filter", action="store_true", help="Print rejected count candidates and reasons.")
    return parser.parse_args()


def _open_capture(source: str) -> tuple[cv2.VideoCapture, bool, str]:
    if source.isdigit():
        return cv2.VideoCapture(int(source)), True, f"camera:{source}"
    path = Path(source)
    return cv2.VideoCapture(str(path)), False, str(path)


def _frame_timestamp(frame_idx: int, fps: float, is_camera: bool, stream_start_sec: float) -> float:
    if is_camera:
        return time.monotonic() - stream_start_sec
    return frame_idx / fps if fps > 0 else 0.0


def _ensure_engine(
    engine: RealtimeCounterEngine | None,
    stream_phase: str,
    count_ready: bool,
    config: EngineConfig,
) -> RealtimeCounterEngine | None:
    if stream_phase == "SEARCHING" and not count_ready:
        return None
    if engine is None and count_ready:
        engine = RealtimeCounterEngine(config)
    return engine


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


def _draw_text(frame, text: str, origin: tuple[int, int], scale: float, color: tuple[int, int, int], thickness: int) -> None:
    cv2.putText(frame, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, (12, 12, 12), thickness + 2, cv2.LINE_AA)
    cv2.putText(frame, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def _draw_progress_bar(frame, top_left: tuple[int, int], size: tuple[int, int], progress: float, fill_color: tuple[int, int, int]) -> None:
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


def _draw_overlay(frame, stream_state, running_count: int, count_ready: bool, countdown_total_sec: float, count_pulse_progress: float) -> None:
    height, width = frame.shape[:2]
    accent = _phase_color(stream_state.phase)
    pulse = _ease_out(count_pulse_progress)

    panel_x, panel_y = 18, 18
    panel_w, panel_h = min(280, width - 36), 108
    _draw_panel(frame, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), (18, 22, 28), 0.58)
    cv2.line(frame, (panel_x + 14, panel_y + 16), (panel_x + 14, panel_y + panel_h - 16), accent, 4)

    if pulse > 0.0:
        flash_color = tuple(int((1.0 - pulse) * 255 + pulse * channel) for channel in accent)
        cv2.rectangle(frame, (panel_x - 1, panel_y - 1), (panel_x + panel_w + 1, panel_y + panel_h + 1), flash_color, thickness=2)

    _draw_text(frame, "Count", (panel_x + 32, panel_y + 30), 0.58, (210, 220, 230), 1)
    _draw_text(frame, str(running_count), (panel_x + 32, panel_y + 72), 1.35 + (0.20 * pulse), (255, 255, 255), 2 + int(round(pulse)))
    _draw_text(frame, stream_state.phase.lower(), (panel_x + 128, panel_y + 34), 0.58, accent, 1)
    status_line = "Ready to count" if count_ready else "Show wrists and feet"
    _draw_text(frame, status_line, (panel_x + 128, panel_y + 60), 0.54, (220, 220, 220), 1)

    if stream_state.phase == "SEARCHING":
        progress_value = stream_state.ready_progress
    elif stream_state.phase == "COUNTDOWN":
        progress_value = 1.0 - min(1.0, stream_state.countdown_remaining_sec / countdown_total_sec)
    else:
        progress_value = 1.0
    _draw_progress_bar(frame, (panel_x + 32, panel_y + 84), (panel_w - 52, 8), progress_value, accent)


def _create_video_writer(path: str | Path, fps: float, frame_shape) -> cv2.VideoWriter:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    height, width = frame_shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps if fps > 0 else 30.0, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Unable to open output video for writing: {path}")
    return writer


def main() -> None:
    args = parse_args()
    capture, is_camera, source_label = _open_capture(args.source)
    if not capture.isOpened():
        raise RuntimeError(f"Unable to open stream source: {args.source}")

    fps = float(capture.get(cv2.CAP_PROP_FPS) or 30.0)
    config = EngineConfig(
        min_airborne_frames=args.min_airborne_frames,
        min_jump_height_ratio=args.min_jump_height_ratio,
        min_hip_lift_ratio=args.min_hip_lift_ratio,
        min_snap_peak_ratio=args.min_snap_peak_ratio,
        min_snap_energy_ratio=args.min_snap_energy_ratio,
        min_rotation_energy_ratio=args.min_rotation_energy_ratio,
        min_count_gap_frames=args.min_count_gap_frames,
        adaptive_gap_enabled=not args.disable_adaptive_filter,
    )
    extractor = PoseSignalExtractor(config)
    gate = RealtimeStartGate(
        ready_hold_seconds=args.ready_hold_seconds,
        countdown_seconds=args.countdown_seconds,
        ready_dropout_seconds=args.ready_dropout_seconds,
    )
    engine: RealtimeCounterEngine | None = None
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

            timestamp_sec = _frame_timestamp(frame_idx, fps, is_camera, stream_start_sec)
            signal, result = extractor.process_bgr_frame(frame, frame_idx, timestamp_sec)
            count_ready = core_landmarks_visible(result, args.ready_visibility_threshold, args.ready_visible_ratio)
            stream_state = gate.update(count_ready, timestamp_sec)
            phase_changed = stream_state.phase != last_phase
            engine = _ensure_engine(engine, stream_state.phase, count_ready, config)

            if phase_changed:
                print(f"[phase] {last_phase} -> {stream_state.phase} @ {timestamp_sec:.2f}s")
                if stream_state.phase == "COUNTING":
                    accepted_count = 0
                    print("[count] counter armed")
                last_phase = stream_state.phase
            if stream_state.phase != "COUNTING" and engine is not None:
                engine.warmup(signal)
            elif engine is not None:
                event = engine.step(signal)
                if event is not None:
                    accepted_count = event.running_count
                    count_pulse_remaining_frames = count_pulse_total_frames
                    print(f"[count] {accepted_count} @ frame={event.frame_idx} time={event.time_sec:.2f}s")
                elif args.debug_filter and engine.last_decision is not None and not engine.last_decision.accepted:
                    decision = engine.last_decision
                    print(
                        f"[reject] frame={decision.frame_idx} reason={decision.reason} "
                        f"air={decision.airtime_frames} jump={decision.jump_height_ratio:.3f} hip={decision.hip_lift_ratio:.3f} "
                        f"peaks={decision.wrist_peak_count} peak={decision.wrist_peak_speed_ratio:.3f} "
                        f"energy={decision.wrist_energy_ratio:.3f} rot={decision.wrist_rotation_ratio:.3f} "
                        f"rot_share={decision.wrist_rotation_share:.2f} fft={decision.wrist_fft_peak_hz:.2f}Hz "
                        f"fft_pow={decision.wrist_fft_power_ratio:.2f} ratio={decision.wrist_to_jump_ratio:.2f}"
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
                count_pulse_progress = count_pulse_remaining_frames / count_pulse_total_frames if count_pulse_total_frames > 0 else 0.0
                _draw_overlay(display_frame, stream_state, accepted_count, count_ready, args.countdown_seconds, count_pulse_progress)

                if args.save_output:
                    if writer is None:
                        writer = _create_video_writer(args.save_output, fps, display_frame.shape)
                    writer.write(display_frame)

            if not args.no_display:
                cv2.imshow("double_jump realtime counter", display_frame)
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

    print(f"[done] source={source_label} frames={frame_idx} final_count={accepted_count}")
    if args.save_output:
        print(f"[saved] {args.save_output}")


if __name__ == "__main__":
    main()
