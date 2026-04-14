from __future__ import annotations

import argparse
import sys
import time
from dataclasses import replace
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
    base_config = EngineConfig()
    parser = argparse.ArgumentParser(description="Run the redesigned double-jump counter on a realtime stream.")
    parser.add_argument("--source", default="0", help="Camera index like `0` or a video file path.")
    parser.add_argument("--save-output", default=None, help="Optional output video path for saving the realtime demo.")
    parser.add_argument("--no-display", action="store_true", help="Disable OpenCV window rendering.")
    parser.add_argument("--max-frames", type=int, default=None, help="Optional frame limit for smoke tests.")
    parser.add_argument("--ready-hold-seconds", type=float, default=1.0)
    parser.add_argument("--countdown-seconds", type=float, default=3.0)
    parser.add_argument("--ready-visibility-threshold", type=float, default=0.30)
    parser.add_argument("--ready-visible-ratio", type=float, default=0.80)
    parser.add_argument("--ready-dropout-seconds", type=float, default=0.35)
    parser.add_argument("--contact-margin-ratio", type=float, default=base_config.contact_margin_ratio)
    parser.add_argument("--min-hip-range-ratio", type=float, default=base_config.min_hip_range_ratio)
    parser.add_argument("--min-foot-range-ratio", type=float, default=base_config.min_foot_range_ratio)
    parser.add_argument("--min-recent-hip-range-ratio", type=float, default=base_config.min_recent_hip_range_ratio)
    parser.add_argument("--max-foot-to-hip-ratio", type=float, default=base_config.max_foot_to_hip_ratio)
    parser.add_argument("--wrist-flow-peak-ratio", type=float, default=base_config.wrist_flow_peak_ratio)
    parser.add_argument("--wrist-flow-mean-ratio", type=float, default=base_config.wrist_flow_mean_ratio)
    parser.add_argument("--wrist-flow-active-ratio", type=float, default=base_config.wrist_flow_active_ratio)
    parser.add_argument("--min-wrist-flow-active-frames", type=int, default=base_config.min_wrist_flow_active_frames)
    parser.add_argument("--wrist-rotation-peak-ratio", type=float, default=base_config.wrist_rotation_peak_ratio)
    parser.add_argument("--wrist-rotation-mean-ratio", type=float, default=base_config.wrist_rotation_mean_ratio)
    parser.add_argument("--wrist-sync-min-ratio", type=float, default=base_config.wrist_sync_min_ratio)
    parser.add_argument("--ankle-flow-active-ratio", type=float, default=base_config.ankle_flow_active_ratio)
    parser.add_argument("--min-rope-pass-hints", type=int, default=base_config.min_rope_pass_hints)
    parser.add_argument("--classifier-confidence-threshold", type=float, default=base_config.classifier_confidence_threshold)
    parser.add_argument("--classifier-model-path", default=base_config.classifier_model_path)
    parser.add_argument("--min-count-gap-frames", type=int, default=base_config.min_count_gap_frames)
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
        engine = RealtimeCounterEngine(config, enable_realtime_compensation=True)
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


def _draw_overlay(
    frame,
    stream_state,
    running_count: int,
    count_ready: bool,
    countdown_total_sec: float,
    count_pulse_progress: float,
    monitor,
) -> None:
    height, width = frame.shape[:2]
    accent = _phase_color(stream_state.phase)
    pulse = _ease_out(count_pulse_progress)

    panel_x, panel_y = 18, 18
    panel_w, panel_h = min(340, width - 36), 210
    _draw_panel(frame, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), (18, 22, 28), 0.58)
    cv2.line(frame, (panel_x + 14, panel_y + 16), (panel_x + 14, panel_y + panel_h - 16), accent, 4)

    if pulse > 0.0:
        flash_color = tuple(int((1.0 - pulse) * 255 + pulse * channel) for channel in accent)
        cv2.rectangle(frame, (panel_x - 1, panel_y - 1), (panel_x + panel_w + 1, panel_y + panel_h + 1), flash_color, thickness=2)

    _draw_text(frame, "Count", (panel_x + 32, panel_y + 30), 0.58, (210, 220, 230), 1)
    _draw_text(frame, str(running_count), (panel_x + 32, panel_y + 72), 1.35 + (0.20 * pulse), (255, 255, 255), 2 + int(round(pulse)))
    _draw_text(frame, stream_state.phase.lower(), (panel_x + 128, panel_y + 34), 0.58, accent, 1)
    status_line = "Ready to count" if count_ready else "Show hips and feet"
    _draw_text(frame, status_line, (panel_x + 128, panel_y + 60), 0.54, (220, 220, 220), 1)

    if stream_state.phase == "SEARCHING":
        progress_value = stream_state.ready_progress
    elif stream_state.phase == "COUNTDOWN":
        progress_value = 1.0 - min(1.0, stream_state.countdown_remaining_sec / countdown_total_sec)
    else:
        progress_value = 1.0
    _draw_progress_bar(frame, (panel_x + 32, panel_y + 84), (panel_w - 52, 8), progress_value, accent)

    if monitor is not None and monitor.detected:
        wrist_line = (
            f"Flow {monitor.wrist_flow_ratio:3.3f}"
            f"  base {monitor.wrist_flow_baseline_ratio:3.3f}"
            f"  peak {monitor.wrist_flow_peak_ratio:3.3f}"
        )
        phase_line = (
            f"Foot {monitor.jump_height_ratio:3.2f}"
            f"  Hip {monitor.hip_lift_ratio:3.2f}"
            f"  contact {int(monitor.contact_gate)}"
        )
        jump_line = (
            f"Rot {monitor.wrist_rotation_ratio:3.3f}"
            f"  sync {monitor.wrist_sync_ratio:3.2f}"
            f"  rope {monitor.rope_pass_hints}"
        )
        detail_line = (
            f"Air active {monitor.wrist_flow_active_frames:02d}"
            f"  airborne {int(monitor.in_air)}"
            f"  ankle {monitor.ankle_flow_ratio:3.3f}"
            f"  adaptive {int(monitor.cadence_locked)}"
        )
        classifier_line = (
            f"Cycle {monitor.cycle_label}"
            f"  conf {monitor.cycle_confidence:3.2f}"
            f"  src {monitor.cycle_source}"
        )
        _draw_text(frame, wrist_line, (panel_x + 32, panel_y + 116), 0.47, (230, 230, 230), 1)
        _draw_text(frame, phase_line, (panel_x + 32, panel_y + 136), 0.47, (230, 230, 230), 1)
        _draw_text(frame, jump_line, (panel_x + 32, panel_y + 156), 0.47, (230, 230, 230), 1)
        _draw_text(frame, detail_line, (panel_x + 32, panel_y + 176), 0.47, (230, 230, 230), 1)
        _draw_text(frame, classifier_line, (panel_x + 32, panel_y + 196), 0.47, (230, 230, 230), 1)


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
    config = replace(
        EngineConfig(),
        contact_margin_ratio=args.contact_margin_ratio,
        min_hip_range_ratio=args.min_hip_range_ratio,
        min_foot_range_ratio=args.min_foot_range_ratio,
        min_recent_hip_range_ratio=args.min_recent_hip_range_ratio,
        max_foot_to_hip_ratio=args.max_foot_to_hip_ratio,
        wrist_flow_peak_ratio=args.wrist_flow_peak_ratio,
        wrist_flow_mean_ratio=args.wrist_flow_mean_ratio,
        wrist_flow_active_ratio=args.wrist_flow_active_ratio,
        min_wrist_flow_active_frames=args.min_wrist_flow_active_frames,
        wrist_rotation_peak_ratio=args.wrist_rotation_peak_ratio,
        wrist_rotation_mean_ratio=args.wrist_rotation_mean_ratio,
        wrist_sync_min_ratio=args.wrist_sync_min_ratio,
        ankle_flow_active_ratio=args.ankle_flow_active_ratio,
        min_rope_pass_hints=args.min_rope_pass_hints,
        classifier_confidence_threshold=args.classifier_confidence_threshold,
        classifier_model_path=args.classifier_model_path,
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
            count_ready = signal.detected or core_landmarks_visible(
                result,
                args.ready_visibility_threshold,
                args.ready_visible_ratio,
            )
            stream_state = gate.update(count_ready, timestamp_sec)
            phase_changed = stream_state.phase != last_phase
            engine = _ensure_engine(engine, stream_state.phase, count_ready, config)

            if phase_changed:
                print(f"[phase] {last_phase} -> {stream_state.phase} @ {timestamp_sec:.2f}s")
                if stream_state.phase == "COUNTING":
                    if engine is not None:
                        engine.arm_for_counting()
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
                    print(
                        f"[count] {accepted_count} ({event.count_delta:+d}) "
                        f"@ frame={event.frame_idx} time={event.time_sec:.2f}s"
                    )
                elif args.debug_filter and engine.last_decision is not None and not engine.last_decision.accepted:
                    decision = engine.last_decision
                    print(
                        f"[reject] frame={decision.frame_idx} reason={decision.reason} "
                        f"foot={decision.jump_height_ratio:.3f} "
                        f"hip={decision.hip_lift_ratio:.3f} "
                        f"flow={decision.current_wrist_flow_ratio:.3f} "
                        f"flow_peak={decision.wrist_flow_peak_ratio:.3f} "
                        f"flow_mean={decision.wrist_flow_mean_ratio:.3f} "
                        f"flow_active={decision.wrist_flow_active_frames} "
                        f"flow_base={decision.wrist_flow_baseline_ratio:.3f} "
                        f"rot_peak={decision.wrist_rotation_peak_ratio:.3f} "
                        f"rot_mean={decision.wrist_rotation_mean_ratio:.3f} "
                        f"sync={decision.wrist_sync_peak_ratio:.2f} "
                        f"rope={decision.rope_pass_hints} "
                        f"class={decision.classifier_label} "
                        f"class_conf={decision.classifier_confidence:.2f} "
                        f"class_src={decision.classifier_source} "
                        f"min_gap={decision.min_gap_frames} "
                        f"adaptive={int(decision.cadence_locked)}"
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
                _draw_overlay(
                    display_frame,
                    stream_state,
                    accepted_count,
                    count_ready,
                    args.countdown_seconds,
                    count_pulse_progress,
                    None if engine is None else engine.monitor,
                )

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
