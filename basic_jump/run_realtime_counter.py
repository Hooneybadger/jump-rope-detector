from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the grouped-jump counter on a realtime stream.")
    parser.add_argument("--source", default="0", help="Camera index like `0` or a video file path.")
    parser.add_argument("--no-display", action="store_true", help="Disable OpenCV window rendering.")
    parser.add_argument("--max-frames", type=int, default=None, help="Optional frame limit for smoke tests.")
    parser.add_argument("--ready-hold-seconds", type=float, default=1.0)
    parser.add_argument("--countdown-seconds", type=float, default=3.0)
    parser.add_argument("--ready-visibility-threshold", type=float, default=0.30)
    return parser.parse_args()


def _open_capture(source: str) -> tuple[cv2.VideoCapture, bool, str]:
    if source.isdigit():
        return cv2.VideoCapture(int(source)), True, f"camera:{source}"
    path = Path(source)
    return cv2.VideoCapture(str(path)), False, str(path)


def _draw_overlay(frame, stream_state, running_count: int, full_body_ready: bool) -> None:
    lines = [
        f"phase: {stream_state.phase}",
        f"full_body_ready: {full_body_ready}",
        f"ready_progress: {stream_state.ready_progress:.2f}",
        f"countdown: {stream_state.countdown_remaining_sec:.1f}s",
        f"count: {running_count}",
        "quit: q",
    ]
    y = 32
    for line in lines:
        cv2.putText(frame, line, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (40, 230, 40), 2, cv2.LINE_AA)
        y += 34


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
                    print("[count] counter armed")

            if stream_state.phase == "COUNTING" and engine is not None:
                event = engine.step(signal)
                if event is not None:
                    print(f"[count] {event.running_count} @ frame={event.frame_idx} time={event.time_sec:.2f}s")

            if not args.no_display:
                display_frame = frame.copy()
                if result.pose_landmarks:
                    mp_drawing.draw_landmarks(
                        display_frame,
                        result.pose_landmarks,
                        mp_pose.POSE_CONNECTIONS,
                        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style(),
                    )
                running_count = 0 if engine is None else engine.running_count
                _draw_overlay(display_frame, stream_state, running_count, full_body_ready)
                cv2.imshow("basic_jump realtime counter", display_frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break

            frame_idx += 1
    finally:
        extractor.close()
        capture.release()
        if not args.no_display:
            cv2.destroyAllWindows()

    final_count = 0 if engine is None else engine.running_count
    print(f"[done] source={source_label} frames={frame_idx} final_count={final_count}")


if __name__ == "__main__":
    main()
