from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from alternating_jump.counter_engine import (
    EngineConfig,
    LabelEvent,
    LabelWindowConfig,
    SignalFrame,
    VideoResult,
    VideoMeta,
    extract_signal_stream,
    load_ground_truth,
    run_dataset,
    save_summary,
    search_best_config,
    summarize_results,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the alternating-jump dataset evaluator.")
    parser.add_argument("--video-dir", default="alternating_jump/video")
    parser.add_argument("--label-dir", default="alternating_jump/label")
    parser.add_argument("--grid-search", action="store_true", help="Run a small config sweep before final evaluation.")
    parser.add_argument("--search-limit", type=int, default=None, help="Optional limit for the config sweep.")
    parser.add_argument("--label-start-offset", type=int, default=0, help="Frame offset applied before the first GT label.")
    parser.add_argument("--label-end-offset", type=int, default=2, help="Frame offset applied after the last GT label.")
    parser.add_argument("--warmup-frames", type=int, default=8, help="Warmup frames processed before counting starts.")
    parser.add_argument("--output", default="alternating_jump/artifacts/dataset_eval_results.json")
    parser.add_argument(
        "--output-dir",
        default="alternating_jump/output",
        help="Directory that receives the latest JSON summary and a readable text report.",
    )
    parser.add_argument(
        "--render-videos",
        action="store_true",
        help="Render validation UI videos into the output directory.",
    )
    parser.add_argument(
        "--render-padding-frames",
        type=int,
        default=20,
        help="Extra frames rendered before and after the evaluation window.",
    )
    return parser.parse_args()


def build_signal_cache(video_dir: Path, stems: list[str]) -> dict[str, tuple[VideoMeta, list[SignalFrame]]]:
    base_config = EngineConfig()
    return {
        stem: extract_signal_stream(video_dir / f"{stem}.mp4", base_config)
        for stem in stems
    }


def render_text_report(
    config: EngineConfig,
    window_config: LabelWindowConfig,
    results,
    summary: dict[str, object],
) -> str:
    lines = [
        "Alternating Jump Dataset Evaluation",
        "",
        f"Config: {config.to_dict()}",
        f"Label window: {window_config.to_dict()}",
        "",
    ]
    for result in results:
        lines.append(
            f"{result.stem}: predicted={result.predicted_count} gt={result.gt_count} "
            f"error={result.count_error:+d} exact={result.exact_match} "
            f"window=[{result.eval_start_frame},{result.eval_end_frame}]"
        )
    lines.extend(
        [
            "",
            f"Overall Count Accuracy: {summary['overall_count_accuracy']:.4f}",
            (
                f"Total Count: predicted={summary['total_predicted_count']} "
                f"gt={summary['total_gt_count']} signed_error={summary['signed_total_error']:+d}"
            ),
            f"Exact Video Count Accuracy: {summary['exact_video_count_accuracy']:.4f}",
            f"Total Abs Error: {summary['total_abs_error']}",
            "",
        ]
    )
    return "\n".join(lines)


def save_output_bundle(
    output_dir: Path,
    config: EngineConfig,
    window_config: LabelWindowConfig,
    results,
    summary: dict[str, object],
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "dataset_eval_results.json"
    report_path = output_dir / "dataset_eval_report.txt"
    save_summary(summary_path, config, summary, window_config)
    report_path.write_text(
        render_text_report(config, window_config, results, summary),
        encoding="utf-8",
    )
    return summary_path, report_path


def _draw_panel(
    frame,
    top_left: tuple[int, int],
    bottom_right: tuple[int, int],
    color: tuple[int, int, int],
    alpha: float = 0.60,
) -> None:
    overlay = frame.copy()
    cv2.rectangle(overlay, top_left, bottom_right, color, thickness=-1)
    cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0.0, dst=frame)


def _draw_text(
    frame,
    text: str,
    origin: tuple[int, int],
    scale: float,
    color: tuple[int, int, int],
    thickness: int = 1,
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
    width, height = size
    progress = max(0.0, min(1.0, progress))
    _draw_panel(frame, (x, y), (x + width, y + height), (28, 30, 34), 0.75)
    fill_width = int(round(width * progress))
    if fill_width > 0:
        cv2.rectangle(frame, (x, y), (x + fill_width, y + height), fill_color, thickness=-1)
    cv2.rectangle(frame, (x, y), (x + width, y + height), (255, 255, 255), thickness=1)


def render_validation_video(
    video_path: Path,
    output_path: Path,
    result: VideoResult,
    ground_truth_events: list[LabelEvent],
    padding_frames: int,
) -> None:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Unable to open video for validation render: {video_path}")

    fps = float(capture.get(cv2.CAP_PROP_FPS) or 30.0)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    start_frame = max(0, result.eval_start_frame - padding_frames)
    end_frame = min(max(0, total_frames - 1), result.eval_end_frame + padding_frames)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps if fps > 0 else 30.0,
        (width, height),
    )
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"Unable to open validation render output: {output_path}")

    pred_frames = result.predicted_frames
    gt_frames = [event.frame_idx for event in ground_truth_events]
    pred_idx = 0
    gt_idx = 0
    pred_count = 0
    gt_count = 0
    while pred_idx < len(pred_frames) and pred_frames[pred_idx] < start_frame:
        pred_count += 1
        pred_idx += 1
    while gt_idx < len(gt_frames) and gt_frames[gt_idx] < start_frame:
        gt_count += 1
        gt_idx += 1

    capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    frame_idx = start_frame
    pred_pulse = 0
    gt_pulse = 0
    window_span = max(1, result.eval_end_frame - result.eval_start_frame)

    try:
        while frame_idx <= end_frame:
            ok, frame = capture.read()
            if not ok:
                break

            pred_hits = 0
            while pred_idx < len(pred_frames) and pred_frames[pred_idx] == frame_idx:
                pred_count += 1
                pred_idx += 1
                pred_hits += 1

            gt_hits = 0
            while gt_idx < len(gt_frames) and gt_frames[gt_idx] == frame_idx:
                gt_count += 1
                gt_idx += 1
                gt_hits += 1

            pred_pulse = 8 if pred_hits else max(0, pred_pulse - 1)
            gt_pulse = 8 if gt_hits else max(0, gt_pulse - 1)

            if frame_idx < result.eval_start_frame:
                phase = "PRE-ROLL"
                progress = 0.0
            elif frame_idx > result.eval_end_frame:
                phase = "POST"
                progress = 1.0
            else:
                phase = "COUNTING"
                progress = (frame_idx - result.eval_start_frame) / window_span

            counts_match = pred_count == gt_count
            accent = (92, 200, 126) if counts_match else (66, 176, 245)
            if pred_hits and gt_hits:
                accent = (118, 216, 118)
            elif pred_hits:
                accent = (84, 214, 255)
            elif gt_hits:
                accent = (255, 196, 72)

            panel_w = min(430, width - 24)
            panel_h = 138
            _draw_panel(frame, (16, 16), (16 + panel_w, 16 + panel_h), (18, 22, 28), 0.62)
            cv2.line(frame, (30, 28), (30, 16 + panel_h - 14), accent, 4)

            _draw_text(frame, f"Validation {result.stem}", (48, 42), 0.72, (244, 244, 244), 2)
            _draw_text(frame, f"Phase {phase}", (48, 68), 0.56, accent, 1)
            _draw_text(
                frame,
                f"Frame {frame_idx}  Window [{result.eval_start_frame}, {result.eval_end_frame}]",
                (48, 92),
                0.52,
                (220, 220, 220),
                1,
            )
            _draw_text(frame, f"Pred {pred_count}/{result.predicted_count}", (48, 118), 0.64, (255, 255, 255), 2)
            _draw_text(frame, f"GT {gt_count}/{result.gt_count}", (220, 118), 0.64, (232, 232, 232), 2)
            _draw_text(
                frame,
                "MATCH" if counts_match else "CHECK",
                (350, 118),
                0.64,
                accent,
                2,
            )

            event_text = "Pred event"
            if pred_hits > 1:
                event_text = f"Pred event x{pred_hits}"
            if pred_hits and gt_hits:
                event_text = f"{event_text} + GT label"
            elif gt_hits:
                event_text = "GT label"
            _draw_text(frame, event_text, (48, 142), 0.50, accent if (pred_hits or gt_hits) else (180, 180, 180), 1)
            _draw_progress_bar(frame, (48, 154), (panel_w - 68, 10), progress, accent)

            if pred_pulse > 0:
                border_color = (84, 214, 255)
                cv2.rectangle(frame, (8, 8), (width - 8, height - 8), border_color, thickness=2)
            if gt_pulse > 0:
                marker_color = (255, 196, 72)
                cv2.rectangle(frame, (12, 12), (width - 12, height - 12), marker_color, thickness=1)

            writer.write(frame)
            frame_idx += 1
    finally:
        writer.release()
        capture.release()


def render_validation_videos(
    video_dir: Path,
    output_dir: Path,
    results: list[VideoResult],
    ground_truth: dict[str, list[LabelEvent]],
    padding_frames: int,
) -> list[Path]:
    rendered_paths: list[Path] = []
    for result in results:
        output_path = output_dir / f"{result.stem}_validation.mp4"
        render_validation_video(
            video_dir / f"{result.stem}.mp4",
            output_path,
            result,
            ground_truth[result.stem],
            padding_frames,
        )
        rendered_paths.append(output_path)
    return rendered_paths


def main() -> None:
    args = parse_args()
    video_dir = Path(args.video_dir)
    label_dir = Path(args.label_dir)
    output_path = Path(args.output)
    output_dir = Path(args.output_dir)

    ground_truth = load_ground_truth(label_dir, video_dir)
    signal_cache = build_signal_cache(video_dir, sorted(ground_truth))

    window_config = LabelWindowConfig(
        start_offset_frames=args.label_start_offset,
        end_offset_frames=args.label_end_offset,
        warmup_frames=args.warmup_frames,
    )
    config = EngineConfig()
    if args.grid_search:
        config, _ = search_best_config(signal_cache, ground_truth, window_config, args.search_limit)

    results = run_dataset(signal_cache, ground_truth, config, window_config)
    summary = summarize_results(results)
    save_summary(output_path, config, summary, window_config)
    output_summary_path, output_report_path = save_output_bundle(
        output_dir,
        config,
        window_config,
        results,
        summary,
    )

    print("Config:", config.to_dict())
    print("Label window:", window_config.to_dict())
    for result in results:
        print(
            f"{result.stem}: predicted={result.predicted_count} gt={result.gt_count} "
            f"error={result.count_error:+d} exact={result.exact_match} "
            f"window=[{result.eval_start_frame},{result.eval_end_frame}]"
        )
    print(f"Overall Count Accuracy: {summary['overall_count_accuracy']:.4f}")
    print(
        f"Total Count: predicted={summary['total_predicted_count']} "
        f"gt={summary['total_gt_count']} signed_error={summary['signed_total_error']:+d}"
    )
    print(f"Exact Video Count Accuracy: {summary['exact_video_count_accuracy']:.4f}")
    print(f"Total Abs Error: {summary['total_abs_error']}")
    print(f"Saved summary to: {output_path}")
    print(f"Saved output json to: {output_summary_path}")
    print(f"Saved output report to: {output_report_path}")
    if args.render_videos:
        render_dir = output_dir / "validation_videos"
        rendered_paths = render_validation_videos(
            video_dir,
            render_dir,
            results,
            ground_truth,
            args.render_padding_frames,
        )
        print(f"Saved validation videos to: {render_dir}")
        for path in rendered_paths:
            print(f"  - {path}")


if __name__ == "__main__":
    main()
