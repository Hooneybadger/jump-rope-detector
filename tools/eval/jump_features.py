"""Per-jump body features that separate double-unders from single (basic) jumps without seeing the rope.

For each labelled jump (benchmark KVA labels, cached native-fps landmarks):
  airtime_sec      both feet above the local floor (rolling 95th percentile of foot y) by >3% of leg length
  foot_rise        peak foot lift during that airtime / leg length
  hip_rise         peak hip lift during the airtime / leg length
  cycle_sec        time to the previous labelled jump
  air_fraction     airtime / cycle
  wrist_turn       summed forearm angle change (rad) during the airtime, larger of the two arms
                   (a double-under needs two hand circles per jump)

    python tools/eval/jump_features.py            # prints per-mode distributions, writes jump_features.csv
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.eval.evaluate_engines import bench_records  # noqa: E402

L_SH, R_SH, L_EL, R_EL, L_WR, R_WR, L_HIP, R_HIP = 11, 12, 13, 14, 15, 16, 23, 24
L_FOOT, R_FOOT = (27, 29, 31), (28, 30, 32)


def _interp(values: np.ndarray) -> np.ndarray:
    values = values.astype(float).copy()
    bad = np.isnan(values)
    if bad.all() or not bad.any():
        return values
    idx = np.arange(len(values))
    values[bad] = np.interp(idx[bad], idx[~bad], values[~bad])
    return values


def body_series(npz: Path) -> dict:
    data = np.load(npz)
    t, lm, fps = data["t"], data["lm"], float(data["fps"])
    left = _interp(np.nanmean(lm[:, L_FOOT, 1], axis=1))
    right = _interp(np.nanmean(lm[:, R_FOOT, 1], axis=1))
    hip = _interp((lm[:, L_HIP, 1] + lm[:, R_HIP, 1]) / 2)
    leg = np.nanmedian(np.maximum(left, right) - hip)
    window = max(3, int(round(2.0 * fps)))
    lower = np.maximum(left, right)  # higher = lower in image; floor = where feet rest
    floor = np.array([np.nanpercentile(lower[max(0, i - window // 2): i + window // 2 + 1], 95) for i in range(len(t))])
    hip_base = np.array([np.nanpercentile(hip[max(0, i - window // 2): i + window // 2 + 1], 90) for i in range(len(t))])
    airborne = (floor - lower) > 0.03 * leg  # the lower foot is off the floor => both feet are
    angles = []
    for shoulder, elbow, wrist in ((L_SH, L_EL, L_WR), (R_SH, R_EL, R_WR)):
        dx = _interp(lm[:, wrist, 0] - lm[:, elbow, 0])
        dy = _interp(lm[:, wrist, 1] - lm[:, elbow, 1])
        angles.append(np.unwrap(np.arctan2(dy, dx)))
    return dict(t=t, fps=fps, leg=leg, floor=floor, lower=lower, hip=hip, hip_base=hip_base,
                airborne=airborne, angles=angles)


def jump_features(series: dict, label_sec: float, previous_sec: float | None) -> dict | None:
    t, air = series["t"], series["airborne"]
    center = int(np.argmin(np.abs(t - label_sec)))
    span = int(round(0.35 * series["fps"]))
    candidates = [i for i in range(max(0, center - span), min(len(t), center + span + 1)) if air[i]]
    if not candidates:
        return None
    pivot = min(candidates, key=lambda i: abs(i - center))
    start = end = pivot
    while start > 0 and air[start - 1]:
        start -= 1
    while end + 1 < len(t) and air[end + 1]:
        end += 1
    sl = slice(start, end + 1)
    leg = series["leg"]
    airtime = (end - start + 1) / series["fps"]
    foot_rise = float(np.max(series["floor"][sl] - series["lower"][sl]) / leg)
    hip_rise = float(np.max(series["hip_base"][sl] - series["hip"][sl]) / leg)
    turn = max(float(np.sum(np.abs(np.diff(a[max(0, start - 1): end + 2])))) for a in series["angles"])
    cycle = None if previous_sec is None else label_sec - previous_sec
    return dict(airtime_sec=round(airtime, 3), foot_rise=round(foot_rise, 3), hip_rise=round(hip_rise, 3),
                cycle_sec=None if cycle is None else round(cycle, 3),
                air_fraction=None if not cycle else round(airtime / cycle, 3), wrist_turn=round(turn, 2))


def main() -> None:
    rows = []
    for record in bench_records(ROOT / "external_data" / "landmarks" / "bench"):
        series = body_series(Path(record["npz"]))
        previous = None
        for label in record["labels_sec"]:
            features = jump_features(series, label, previous)
            previous = label
            if features:
                rows.append(dict(id=record["id"], mode=record["mode"], time_sec=round(label, 3), **features))
    out = ROOT / "external_data" / "eval" / "jump_features.csv"
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for key in ("airtime_sec", "foot_rise", "hip_rise", "cycle_sec", "air_fraction", "wrist_turn"):
        print(f"\n{key}")
        for mode in ("basic", "alternating", "double"):
            values = np.array([r[key] for r in rows if r["mode"] == mode and r[key] is not None], dtype=float)
            values = values[np.isfinite(values) & (values < 10)]
            print(f"  {mode:11s} n={len(values):4d} p10={np.percentile(values, 10):.3f} p50={np.median(values):.3f} "
                  f"p90={np.percentile(values, 90):.3f}")
    print("\nper video median (airtime, foot_rise, air_fraction, wrist_turn):")
    for vid in dict.fromkeys(r["id"] for r in rows):
        sel = [r for r in rows if r["id"] == vid]
        med = lambda k: np.nanmedian([r[k] if r[k] is not None else np.nan for r in sel])  # noqa: E731
        print(f"  {vid:16s} n={len(sel):3d} air={med('airtime_sec'):.3f}s rise={med('foot_rise'):.3f} "
              f"frac={med('air_fraction'):.2f} cycle={med('cycle_sec'):.3f}s wrist={med('wrist_turn'):.2f}rad")


if __name__ == "__main__":
    np.seterr(all="ignore")
    main()
