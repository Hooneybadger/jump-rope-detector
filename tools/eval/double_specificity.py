"""Label-free check of the double-under decision on visually reviewed external segments.

Runs the double-under engine (as the app would, 30 Hz + SignalResampler) on every segment that the
visual review marked as basic or alternating: every double-under counted there is a false positive.
Segments marked double report the count next to the automatic foot-touchdown count (an unverified
proxy for the number of jumps; segments may include rests or misses).
Variants: classifier only, hand ratio decides when usable (default), hand ratio only adds doubles (assist).
"""
from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools" / "external_data"))
from app.pose_protocol import PoseSamplingClock, plausible_pose  # noqa: E402
from app.signal_resampler import SignalResampler  # noqa: E402
from tools.eval.evaluate_engines import pose_result  # noqa: E402

import double_jump.counter_engine as engine_module  # noqa: E402

VARIANTS = {
    "classifier_only": dict(hand_ratio_enabled=False),
    "hand_decide": {},
    "hand_assist": dict(hand_ratio_mode="assist"),
}


def run(npz: Path, start: float, end: float, overrides: dict) -> int:
    config = replace(engine_module.EngineConfig(), **overrides)
    extractor = engine_module.PoseSignalExtractor(config, initialize_pose=False)
    engine = engine_module.RealtimeCounterEngine(config, enable_realtime_compensation=False)
    resampler, clock = SignalResampler(), PoseSamplingClock(30.0)
    data = np.load(npz)
    engine.frame_height_px = float(data["height"]) * float(data["inference_scale"])
    count, armed = 0, False
    for index, (t, row) in enumerate(zip(data["t"], data["lm"])):
        if t < start - 3.0:
            continue
        if t > end:
            break
        if not clock.accept(float(t)):
            continue
        signal, _ = extractor.process_pose_result(plausible_pose(pose_result(row)), index, float(t))
        if not armed and t >= start:
            engine.arm_for_counting()
            armed = True
        for frame in resampler.push(signal):
            if not armed:
                engine.warmup(frame)
                continue
            event = engine.step(frame)
            if event is not None:
                count = int(event.running_count)
    return count


def main() -> None:
    np.seterr(all="ignore")
    review = json.loads((ROOT / "external_data" / "screening" / "review.json").read_text(encoding="utf-8"))
    segments = {i["id"]: i for i in json.loads((ROOT / "external_data" / "screening" / "segments.json").read_text(encoding="utf-8"))}
    from propose_labels import pose_touchdowns

    totals = {mode: {name: 0 for name in VARIANTS} for mode in ("basic", "double")}
    touchdown_total = 0
    seconds = {mode: 0.0 for mode in totals}
    for key, mode in review.items():
        if key.startswith("_") or mode not in totals:
            continue
        video_id, number = key.rsplit("_seg", 1)
        segment = segments[video_id]["segments"][int(number)]
        npz = ROOT / "external_data" / "landmarks" / "cc" / f"{video_id}.npz"
        counts = {name: run(npz, segment["start_sec"], segment["end_sec"], overrides) for name, overrides in VARIANTS.items()}
        touchdowns = len(pose_touchdowns(npz, segment["start_sec"], segment["end_sec"], alternate=False)[0])
        if mode == "double":
            touchdown_total += touchdowns
        for name, value in counts.items():
            totals[mode][name] += value
        seconds[mode] += segment["end_sec"] - segment["start_sec"]
        print(f"{key:22s} {mode:11s} {segment['end_sec'] - segment['start_sec']:5.1f}s jumps~{touchdowns:3d}  " +
              "  ".join(f"{name}={value}" for name, value in counts.items()), flush=True)
    print()
    for mode, values in totals.items():
        if seconds[mode]:
            print(f"{mode:11s} {seconds[mode] / 60:5.1f} min  " + "  ".join(f"{n}={v}" for n, v in values.items()))
    if touchdown_total:
        print(f"double segments: automatic foot-touchdown count {touchdown_total} (unverified proxy for the number of jumps)")


if __name__ == "__main__":
    main()
