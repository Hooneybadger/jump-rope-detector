from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from basic_jump.counter_engine import (
    EngineConfig,
    LabelWindowConfig,
    SignalFrame,
    VideoMeta,
    extract_signal_stream,
    load_ground_truth,
    run_dataset,
    save_summary,
    search_best_config,
    summarize_results,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the grouped-jump dataset evaluator.")
    parser.add_argument("--video-dir", default="basic_jump/video")
    parser.add_argument("--label-dir", default="basic_jump/label")
    parser.add_argument("--grid-search", action="store_true", help="Run a small config sweep before final evaluation.")
    parser.add_argument("--search-limit", type=int, default=None, help="Optional limit for the small config sweep.")
    parser.add_argument("--label-start-offset", type=int, default=-15, help="Frame offset applied before the first GT label.")
    parser.add_argument("--label-end-offset", type=int, default=3, help="Frame offset applied after the last GT label.")
    parser.add_argument("--warmup-frames", type=int, default=4, help="Warmup frames processed before counting starts.")
    parser.add_argument("--output", default="basic_jump/artifacts/dataset_eval_results.json")
    return parser.parse_args()


def build_signal_cache(video_dir: Path, stems: list[str]) -> dict[str, tuple[VideoMeta, list[SignalFrame]]]:
    return {
        stem: extract_signal_stream(video_dir / f"{stem}.mp4", EngineConfig())
        for stem in stems
    }


def main() -> None:
    args = parse_args()
    video_dir = Path(args.video_dir)
    label_dir = Path(args.label_dir)

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
    save_summary(args.output, config, summary, window_config)

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
    print(f"Saved summary to: {args.output}")


if __name__ == "__main__":
    main()
