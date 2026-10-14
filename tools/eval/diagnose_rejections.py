"""Tally why an engine rejected count candidates on cached landmarks (engine.last_decision.reason).

For every label, report whether a candidate was produced near it and, if so, the reject reason.
    python tools/eval/diagnose_rejections.py basic_09 basic_10 double_3 --rate native
"""
from __future__ import annotations

import argparse
import collections
import importlib
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from app.pose_protocol import PoseSamplingClock  # noqa: E402
from app.signal_resampler import SignalResampler  # noqa: E402
from tools.eval.evaluate_engines import HOOKS, bench_records, json_records, pose_result  # noqa: E402


def diagnose(record: dict, rate: str, tol: float) -> None:
    module = importlib.import_module(f"{record['mode']}_jump.counter_engine")
    config = module.EngineConfig()
    extractor = module.PoseSignalExtractor(config, initialize_pose=False)
    engine = module.RealtimeCounterEngine(config, enable_realtime_compensation=True)
    resampler = SignalResampler() if record["mode"] == "double" else None
    clock = None if rate == "native" else PoseSamplingClock(float(rate))
    data = np.load(record["npz"])
    labels = np.asarray(record["labels_sec"])
    start, end = labels[0] - 0.5, labels[-1] + 0.5
    decisions, armed, index, undetected, in_window = [], False, 0, 0, 0
    for t, row in zip(data["t"], data["lm"]):
        if t > end:
            break
        if clock is not None and not clock.accept(float(t)):
            continue
        signal, _ = extractor.process_pose_result(pose_result(row), index, float(t))
        index += 1
        if not armed and t >= start:
            if HOOKS[record["mode"]]:
                getattr(engine, HOOKS[record["mode"]])()
            armed = True
        for frame in ([signal] if resampler is None else resampler.push(signal)):
            if not armed:
                engine.warmup(frame)
                continue
            in_window += 1
            undetected += not frame.detected
            engine.step(frame)
            if engine.last_decision is not None:
                decisions.append((frame.time_sec, engine.last_decision.reason, engine.last_decision))
    reasons = collections.Counter(reason for _, reason, _ in decisions)
    per_label = collections.Counter()
    for label in labels:
        near = [d for d in decisions if abs(d[0] - label) <= tol]
        if not near:
            per_label["no_candidate"] += 1
        elif any(d[1] == "accepted" for d in near):
            per_label["accepted"] += 1
        else:
            per_label["rejected:" + near[0][1]] += 1
    print(f"\n{record['id']} ({record['mode']}, rate {rate}) labels={len(labels)} "
          f"median interval={np.median(np.diff(labels)):.3f}s undetected frames={undetected}/{in_window}")
    print("  candidate decisions:", dict(reasons))
    print("  per label:", dict(per_label))
    rejected = [d[2] for d in decisions if d[1] != "accepted"][:3]
    for decision in rejected:
        fields = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in vars(decision).items()
                  if k not in {"frame_idx", "accepted"}}
        print("  e.g.", fields)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("ids", nargs="+")
    parser.add_argument("--rate", default="native")
    parser.add_argument("--tol", type=float, default=0.25)
    parser.add_argument("--labels-json", help="external labels (landmarks from external_data/landmarks/cc); ids 'basic' etc. select a mode")
    args = parser.parse_args()
    if args.labels_json:
        records = {r["id"]: r for r in json_records(Path(args.labels_json), ROOT / "external_data" / "landmarks" / "cc")}
    else:
        records = {r["id"]: r for r in bench_records(ROOT / "external_data" / "landmarks" / "bench")}
    for record_id in args.ids:
        for record in ([records[record_id]] if record_id in records else [r for r in records.values() if r["mode"] == record_id]):
            diagnose(record, args.rate, args.tol)


if __name__ == "__main__":
    main()
