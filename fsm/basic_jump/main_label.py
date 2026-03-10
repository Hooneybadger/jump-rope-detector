import argparse
import os

import pandas as pd

from jump_rope_pipeline import run_pipeline
from jump_rope_settings import OUTPUT_DIR, TARGET_VIDEO_STEM, build_video_jobs, get_summary_csv_name


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Jump rope tracker (labeled mode)")
    parser.add_argument(
        "--target-stem",
        default=TARGET_VIDEO_STEM,
        help="video stem in input/video (ex: 03)",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    target_stem = (args.target_stem or "").strip()
    if target_stem:
        return run_pipeline(
            target_stem=target_stem,
            require_label=True,
        )

    jobs = build_video_jobs(require_label=True)
    if not jobs:
        print("[WARN] No labeled video jobs found.")
        return []

    all_rows = []
    for job in jobs:
        stem = str(job["stem"])
        print(f"\n[BATCH] isolated_run stem={stem}")
        rows = run_pipeline(
            target_stem=stem,
            require_label=True,
            write_summary=False,
        )
        all_rows.extend(rows)

    if all_rows:
        summary_df = pd.DataFrame(all_rows)
        summary_csv = os.path.join(OUTPUT_DIR, get_summary_csv_name(use_realtime=False))
        summary_df.to_csv(summary_csv, index=False)
        print(f"\n[SUMMARY] Saved: {summary_csv}")
        print(
            "[SUMMARY] strict_f1_mean="
            f"{summary_df['strict_f1'].mean():.3f} "
            f"strict_f1_min={summary_df['strict_f1'].min():.3f} "
            f"adjusted_f1_mean={summary_df['adjusted_f1'].mean():.3f} "
            f"full_strict_f1_mean={summary_df['full_strict_f1'].mean():.3f} "
            f"full_adjusted_f1_mean={summary_df['full_adjusted_f1'].mean():.3f}"
        )
        print("[SUMMARY] Per video:")
        for _, row in summary_df.iterrows():
            print(
                f"[SUMMARY] {row['video_stem']}: "
                f"strict_f1={row['strict_f1']:.3f} adjusted_f1={row['adjusted_f1']:.3f} "
                f"full_strict_f1={row['full_strict_f1']:.3f} "
                f"full_adjusted_f1={row['full_adjusted_f1']:.3f} "
                f"matched={int(row['matched'])}/{int(row['label_count'])} "
                f"strict_extra={int(row['strict_extra_detected'])} "
                f"adjusted_extra={int(row['adjusted_extra_detected'])} "
                f"full_strict_extra={int(row['full_strict_extra_detected'])} "
                f"full_adjusted_extra={int(row['full_adjusted_extra_detected'])} "
                f"missed={int(row['missed_labels'])}"
            )
    return all_rows


if __name__ == "__main__":
    main()
