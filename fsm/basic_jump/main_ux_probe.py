import argparse
import glob
import os

import numpy as np
import pandas as pd

from jump_rope_settings import OUTPUT_DIR


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Summarize UX probe sessions")
    parser.add_argument(
        "--session-dir",
        action="append",
        default=[],
        help="explicit UX probe session directory (repeatable)",
    )
    parser.add_argument(
        "--session-glob",
        action="append",
        default=[],
        help="glob pattern for UX probe session directories (repeatable)",
    )
    parser.add_argument(
        "--all-matches",
        action="store_true",
        help="do not collapse to the latest session per stem",
    )
    parser.add_argument(
        "--lag-threshold-ms",
        type=int,
        default=1000,
        help="threshold used for pass/fail summary",
    )
    parser.add_argument(
        "--output-csv",
        default=os.path.join(OUTPUT_DIR, "ux_probe_summary.csv"),
        help="output CSV path",
    )
    return parser.parse_args(argv)


def _default_session_globs():
    return [os.path.join(OUTPUT_DIR, "ux_*", "*")]


def _read_detected_events(session_dir):
    candidates = []
    for file_name in sorted(os.listdir(session_dir)):
        if file_name.endswith("_detected_events.csv"):
            file_path = os.path.join(session_dir, file_name)
            candidates.append(file_path)

    if not candidates:
        return "", None

    for file_path in candidates:
        if os.path.getsize(file_path) > 1:
            return file_path, pd.read_csv(file_path)
    return candidates[0], None


def _infer_stem(session_dir, detected_events_path):
    if detected_events_path:
        return os.path.basename(detected_events_path).replace("_detected_events.csv", "")
    return os.path.basename(session_dir)


def _collect_session_row(session_dir):
    frame_log_path = os.path.join(session_dir, "frame_log.csv")
    if not os.path.exists(frame_log_path) or os.path.getsize(frame_log_path) <= 1:
        return None

    frame_log_df = pd.read_csv(frame_log_path)
    if "overlay_counter" not in frame_log_df.columns:
        return None

    overlay_counter = frame_log_df["overlay_counter"].ffill().fillna(0).astype(int)
    monotonic_overlay = bool((overlay_counter.diff().fillna(0) >= 0).all())
    overlay_last_count = int(overlay_counter.iloc[-1]) if len(overlay_counter) else 0
    overlay_max_count = int(overlay_counter.max()) if len(overlay_counter) else 0
    overlay_mode = ""
    if "overlay_counter_mode" in frame_log_df.columns and not frame_log_df.empty:
        overlay_mode = str(frame_log_df["overlay_counter_mode"].mode().iloc[0])

    detected_events_path, detected_events_df = _read_detected_events(session_dir)
    stem = _infer_stem(session_dir, detected_events_path)

    raw_commit_lags = []
    positive_commit_lags = []
    final_count = 0
    if detected_events_df is not None and len(detected_events_df):
        final_count = int(len(detected_events_df))
        for idx, event_ts_ms in enumerate(detected_events_df["timestamp_ms"].astype(int).tolist(), start=1):
            shown_ts = frame_log_df.loc[overlay_counter >= idx, "timestamp_ms"]
            if len(shown_ts) == 0:
                continue
            raw_lag_ms = int(shown_ts.iloc[0]) - int(event_ts_ms)
            raw_commit_lags.append(raw_lag_ms)
            positive_commit_lags.append(max(0, raw_lag_ms))

    row = {
        "session": os.path.basename(session_dir),
        "session_dir": session_dir,
        "stem": stem,
        "frame_log_path": frame_log_path,
        "detected_events_path": detected_events_path,
        "overlay_mode": overlay_mode,
        "final_count": final_count,
        "overlay_last_count": overlay_last_count,
        "overlay_max_count": overlay_max_count,
        "final_live_delta": final_count - overlay_last_count,
        "monotonic_overlay": monotonic_overlay,
        "first_event_raw_gap_ms": np.nan,
        "first_event_positive_gap_ms": np.nan,
        "mean_positive_commit_lag_ms": np.nan,
        "p95_positive_commit_lag_ms": np.nan,
        "max_positive_commit_lag_ms": np.nan,
        "session_mtime_s": os.path.getmtime(frame_log_path),
    }
    if raw_commit_lags:
        row["first_event_raw_gap_ms"] = float(raw_commit_lags[0])
        row["first_event_positive_gap_ms"] = float(positive_commit_lags[0])
        row["mean_positive_commit_lag_ms"] = float(np.mean(positive_commit_lags))
        row["p95_positive_commit_lag_ms"] = float(np.percentile(positive_commit_lags, 95))
        row["max_positive_commit_lag_ms"] = float(np.max(positive_commit_lags))
    return row


def _discover_session_dirs(session_dirs, session_globs):
    discovered = list(session_dirs)
    for pattern in session_globs or _default_session_globs():
        for path in sorted(glob.glob(pattern)):
            if os.path.isdir(path):
                discovered.append(path)
    # preserve order while deduplicating
    seen = set()
    ordered = []
    for path in discovered:
        norm_path = os.path.normpath(path)
        if norm_path in seen:
            continue
        seen.add(norm_path)
        ordered.append(norm_path)
    return ordered


def _select_latest_per_stem(rows):
    latest_by_stem = {}
    for row in rows:
        stem = row["stem"]
        best = latest_by_stem.get(stem)
        if best is None or row["session_mtime_s"] > best["session_mtime_s"]:
            latest_by_stem[stem] = row
    return [latest_by_stem[key] for key in sorted(latest_by_stem)]


def main(argv=None):
    args = parse_args(argv)
    session_dirs = _discover_session_dirs(args.session_dir, args.session_glob)
    if not session_dirs:
        print("[WARN] No UX probe session directories found.")
        return []

    collected_rows = []
    for session_dir in session_dirs:
        row = _collect_session_row(session_dir)
        if row is None:
            print(f"[WARN] skipped_session={session_dir}")
            continue
        collected_rows.append(row)

    if not collected_rows:
        print("[WARN] No usable UX probe sessions collected.")
        return []

    if args.all_matches:
        selected_rows = sorted(collected_rows, key=lambda item: item["session"])
    else:
        selected_rows = _select_latest_per_stem(collected_rows)

    summary_df = pd.DataFrame(selected_rows)
    summary_df = summary_df.drop(columns=["session_mtime_s"])
    output_csv = args.output_csv
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    summary_df.to_csv(output_csv, index=False)

    lag_threshold_ms = max(0, int(args.lag_threshold_ms))
    lag_series = summary_df["max_positive_commit_lag_ms"].dropna()
    lag_ok = bool((lag_series <= lag_threshold_ms).all()) if len(lag_series) else True
    delta_ok = bool((summary_df["final_live_delta"] == 0).all())
    monotonic_ok = bool(summary_df["monotonic_overlay"].all())

    print(f"[SUMMARY] Saved: {output_csv}")
    print(
        "[SUMMARY] "
        f"all_final_live_delta_zero={delta_ok} "
        f"all_monotonic={monotonic_ok} "
        f"all_positive_commit_lag_le_{lag_threshold_ms}ms={lag_ok}"
    )
    print("[SUMMARY] Per session:")
    for _, row in summary_df.iterrows():
        max_lag = row["max_positive_commit_lag_ms"]
        if pd.isna(max_lag):
            max_lag_text = "nan"
        else:
            max_lag_text = str(int(round(float(max_lag))))
        print(
            f"[SUMMARY] {row['stem']}: "
            f"mode={row['overlay_mode']} "
            f"final={int(row['final_count'])} "
            f"overlay_last={int(row['overlay_last_count'])} "
            f"delta={int(row['final_live_delta'])} "
            f"monotonic={bool(row['monotonic_overlay'])} "
            f"max_positive_commit_lag_ms={max_lag_text}"
        )
    return selected_rows


if __name__ == "__main__":
    main()
