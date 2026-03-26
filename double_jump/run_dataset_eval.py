from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from double_jump.counter_engine import (
    EngineConfig,
    LabelWindowConfig,
    extract_signal_stream,
    load_ground_truth,
    run_dataset,
    save_summary,
    search_best_config,
    summarize_results,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the double-jump dataset evaluator.")
    parser.add_argument("--video-dir", default="double_jump/video")
    parser.add_argument("--label-dir", default="double_jump/label")
    parser.add_argument("--grid-search", action="store_true", help="Run a small config sweep before final evaluation.")
    parser.add_argument("--search-limit", type=int, default=None, help="Optional limit for the small config sweep.")
    parser.add_argument("--label-start-offset", type=int, default=-15)
    parser.add_argument("--label-end-offset", type=int, default=3)
    parser.add_argument("--warmup-frames", type=int, default=10)
    parser.add_argument("--output", default="double_jump/artifacts/dataset_eval_results.json")
    return parser.parse_args()


def build_signal_cache(video_dir: Path, stems: list[str]):
    return {stem: extract_signal_stream(video_dir / f"{stem}.mp4", EngineConfig()) for stem in stems}


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

    if args.grid_search:
        config, _ = search_best_config(signal_cache, ground_truth, window_config, limit=args.search_limit)
    else:
        config = EngineConfig()

    results = run_dataset(signal_cache, ground_truth, config, window_config)
    summary = summarize_results(results)
    save_summary(args.output, config, summary, window_config)
    print(f"[saved] {args.output}")
    print(f"[summary] overall={summary['overall_count_accuracy']:.4f} exact={summary['exact_video_count_accuracy']:.4f}")


if __name__ == "__main__":
    main()
