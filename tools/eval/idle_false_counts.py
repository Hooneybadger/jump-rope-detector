"""Label-free false-count check: run an engine where the person is visible but not rope jumping.

Uses the single-person external videos (not in manifest/excluded_ids.txt). Every 3 s window (0.5 s hop) with
the full body visible is classified as in screen_segments.py:
  * jumping   periodic hip motion at 1.3-4.6 Hz (r >= 0.5, amplitude >= 0.02 torso);
  * idle      not jumping, detrended hip std < 0.02 torso and ankle-height std < 0.02 torso
              (standing, talking, resting: the feet stay on the floor);
  * other     not jumping but moving (walking, demonstrating, stretching, tricks, irregular jumping).
Runs of idle (or other) windows of at least --min-seconds are counted with the app pipeline
(tools/eval/evaluate_engines.run_record): every count there is a false count for idle, and a count that
needs a look for other.

    python tools/eval/idle_false_counts.py --mode basic --override '{"basic": {...}}'
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools" / "external_data"))
from screen_segments import ACCEPT_MAX, ACCEPT_MIN, HOP_SEC, WINDOW_SEC, _interp_nan, _period  # noqa: E402
from tools.eval.evaluate_engines import run_record  # noqa: E402

NOSE, L_SH, R_SH, L_HIP, R_HIP, L_ANK, R_ANK = 0, 11, 12, 23, 24, 27, 28


def window_classes(npz: Path) -> tuple[np.ndarray, np.ndarray]:
    """Per-frame class: 0 = not usable, 1 = jumping, 2 = idle, 3 = other motion."""
    data = np.load(npz)
    t, lm, fps = data["t"], data["lm"], float(data["fps"])
    vis = np.nan_to_num(lm[:, :, 3])
    valid = np.all(vis[:, [NOSE, L_HIP, R_HIP, L_ANK, R_ANK]] >= 0.5, axis=1)
    valid &= np.nan_to_num(lm[:, [L_ANK, R_ANK], 1], nan=2.0).max(axis=1) <= 1.0
    torso = np.abs((lm[:, L_HIP, 1] + lm[:, R_HIP, 1]) / 2 - (lm[:, L_SH, 1] + lm[:, R_SH, 1]) / 2)
    # normalised per window: a whole-video torso median is skewed by close-ups and intro shots
    hip = _interp_nan((lm[:, L_HIP, 1] + lm[:, R_HIP, 1]) / 2)
    ankle = _interp_nan((lm[:, L_ANK, 1] + lm[:, R_ANK, 1]) / 2)
    win, hop = int(round(WINDOW_SEC * fps)), max(1, int(round(HOP_SEC * fps)))
    jumping, idle, other = (np.zeros(len(t), bool) for _ in range(3))
    for start in range(0, max(0, len(t) - win), hop):
        sl = slice(start, start + win)
        if valid[sl].mean() < 0.9:
            continue
        scale = float(np.nanmedian(torso[sl]))
        freq, r = _period(hip[sl], fps)
        x = np.arange(win)
        amplitude = float(np.std(hip[sl] - np.polyval(np.polyfit(x, hip[sl], 1), x))) / scale
        feet = float(np.std(ankle[sl] - np.polyval(np.polyfit(x, ankle[sl], 1), x))) / scale
        if r >= 0.5 and ACCEPT_MIN <= freq <= ACCEPT_MAX and amplitude >= 0.02:
            jumping[sl] = True
        elif amplitude < 0.02 and feet < 0.02:
            idle[sl] = True
        else:
            other[sl] = True
    classes = np.zeros(len(t), int)
    classes[other & ~jumping] = 3
    classes[idle & ~jumping & ~other] = 2
    classes[jumping] = 1
    return t, classes


def runs(t: np.ndarray, classes: np.ndarray, value: int, min_seconds: float) -> list[tuple[float, float]]:
    out, i = [], 0
    while i < len(classes):
        if classes[i] != value:
            i += 1
            continue
        j = i
        while j < len(classes) and classes[j] == value:
            j += 1
        if t[j - 1] - t[i] >= min_seconds:
            out.append((float(t[i]), float(t[j - 1])))
        i = j
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="basic", choices=["basic", "alternating", "double"])
    parser.add_argument("--rate", default="30")
    parser.add_argument("--override", default="{}")
    parser.add_argument("--min-seconds", type=float, default=4.0)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    overrides = json.loads(args.override)
    excluded = {line.split()[0] for line in (ROOT / "external_data" / "manifest" / "excluded_ids.txt").read_text(encoding="utf-8").splitlines() if line.strip()}
    totals = {"idle": [0, 0.0], "other": [0, 0.0]}
    for npz in sorted((ROOT / "external_data" / "landmarks" / "cc").glob("*.npz")):
        if npz.stem in excluded:
            continue
        t, classes = window_classes(npz)
        for name, value in (("idle", 2), ("other", 3)):
            for start, end in runs(t, classes, value, args.min_seconds):
                record = dict(id=npz.stem, mode=args.mode, npz=str(npz), labels_sec=[start + 0.5, end], segment=[start, end])
                result = run_record(record, args.rate, overrides, True, end_pad=0.0)
                count = sum(e["count_delta"] > 0 for e in result["events"] if start <= e["time_sec"] <= end)
                totals[name][0] += count
                totals[name][1] += end - start
                if args.verbose and count:
                    print(f"{npz.stem} {name} {start:7.1f}-{end:7.1f}s counts={count}", flush=True)
    for name, (count, seconds) in totals.items():
        print(f"{args.mode} {name:5s} {seconds / 60:6.1f} min  counts={count}  per_min={count / max(seconds / 60, 1e-9):.2f}")


if __name__ == "__main__":
    main()
