"""Offline evaluation of the counting engines on cached landmarks.

Replays cached PoseLandmarker output (tools/external_data/cache_landmarks.py) through the same
adapter path the app uses (pose result -> PoseSignalExtractor -> optional SignalResampler -> engine)
with the benchmark protocol: count from the first label - 0.5 s to the last label + 0.5 s, earlier
frames only warm the engine, readiness gate and countdown bypassed.

Reports, per video and per mode:
  count agreement (benchmark definition), MAE, off-by-one rate, signed error,
  event precision/recall/F1 (greedy one-to-one matching within --tol seconds),
  and every -1 compensation with its position relative to the labels.

Run with the Python 3.11 + mediapipe 0.10.21 environment:
    python tools/eval/evaluate_engines.py --rate 20 --rate 30 --rate native
"""
from __future__ import annotations

import argparse
import csv
import importlib
import json
import sys
import xml.etree.ElementTree as ET
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from app.pose_protocol import PoseSamplingClock, plausible_pose  # noqa: E402
from app.rope_trip import RopeTripPolicy  # noqa: E402
from tools.eval.engine_rate import scale_config  # noqa: E402
from app.signal_resampler import SignalResampler  # noqa: E402

WARMUP_SEC = 3.0
HOOKS = {"basic": None, "alternating": "begin_count_phase", "double": "arm_for_counting"}


def bench_records(landmark_dir: Path) -> list[dict]:
    """Benchmark videos with KVA labels, using the label rule of tools/compare_pose_backends.py."""
    records = []
    for mode in ("basic", "alternating", "double"):
        for label in sorted((ROOT / "benchmark_results" / "cache" / mode).glob("*.kva")):
            npz = landmark_dir / f"{mode}_{label.stem}.npz"
            if not npz.exists():
                continue
            xml = ET.parse(label).getroot()
            ticks = float(xml.findtext("AverageTimeStampsPerFrame"))
            fps = float(np.load(npz)["fps"])
            frames = sorted({
                int(round(int(k.findtext("Timestamp")) / ticks))
                for k in xml.findall("./Keyframes/Keyframe")
                if any(p.find("./PointList/Point") is not None for p in k.findall("./Drawings/Pencil"))
            })
            records.append(dict(id=f"{mode}_{label.stem}", mode=mode, npz=str(npz),
                                labels_sec=[f / fps for f in frames], group=f"bench_{mode}_{label.stem}"))
    return records


def json_records(path: Path, landmark_dir: Path) -> list[dict]:
    """External labels: [{"id", "mode", "labels_sec", "video_id"?, "group"?, "tier"?, "segment"?: [start, end]}]."""
    records = []
    for item in json.loads(path.read_text(encoding="utf-8")):
        npz = landmark_dir / f"{item.get('video_id', item['id'])}.npz"
        if npz.exists():
            records.append(dict(item, npz=str(npz), group=item.get("group", item["id"])))
    return records


def pose_result(row: np.ndarray):
    if np.isnan(row[0, 0]):
        return SimpleNamespace(pose_landmarks=None)
    points = [SimpleNamespace(x=float(x), y=float(y), z=float(z), visibility=float(v), presence=float(p))
              for x, y, z, v, p in row]
    return SimpleNamespace(pose_landmarks=SimpleNamespace(landmark=points))


def run_record(record: dict, rate: str, overrides: dict, resample_double: bool, scale_ref: dict | None = None,
               raw_pose: bool = False, trip_policy: str = "resume", end_pad: float = 0.5) -> dict:
    module = importlib.import_module(f"{record['mode']}_jump.counter_engine")
    config = replace(module.EngineConfig(), **overrides.get(record["mode"], {}))
    if scale_ref and record["mode"] in scale_ref:
        effective = float(np.load(record["npz"])["fps"]) if rate == "native" else min(float(rate), float(np.load(record["npz"])["fps"]))
        config = scale_config(config, record["mode"], effective, scale_ref[record["mode"]])
    extractor = module.PoseSignalExtractor(config, initialize_pose=False)
    engine = module.RealtimeCounterEngine(config, enable_realtime_compensation=True)
    resampler = SignalResampler() if (record["mode"] == "double" and resample_double) else None
    clock = None if rate == "native" else PoseSamplingClock(float(rate))

    data = np.load(record["npz"])
    times, landmarks = data["t"], data["lm"]
    if hasattr(engine, "frame_height_px"):  # height of the image the pose model saw when the cache was made
        engine.frame_height_px = float(data["height"]) * float(data["inference_scale"])
    labels = record["labels_sec"]
    seg_start, seg_end = record.get("segment") or (0.0, float(times[-1]) if len(times) else 0.0)
    start = max(seg_start, labels[0] - 0.5)
    end = min(seg_end, labels[-1] + end_pad)
    policy = RopeTripPolicy() if trip_policy == "resume" else None

    armed, count, signal_index, events = False, 0, 0, []
    for t, row in zip(times, landmarks):
        if t < seg_start - WARMUP_SEC:  # up to WARMUP_SEC before a segment only warms the engine
            continue
        if t > end:
            break
        if clock is not None and not clock.accept(float(t)):
            continue
        result = pose_result(row) if raw_pose else plausible_pose(pose_result(row))
        signal, _ = extractor.process_pose_result(result, signal_index, float(t))
        signal_index += 1
        if not armed and t >= start:
            if HOOKS[record["mode"]]:
                getattr(engine, HOOKS[record["mode"]])()
            armed = True
        for frame in ([signal] if resampler is None else resampler.push(signal)):
            if not armed:
                engine.warmup(frame)
                continue
            if policy is not None:
                policy.expire(frame.time_sec)
            event = engine.step(frame)
            if event is not None:
                count = int(event.running_count) if policy is None else policy.observe(event, frame.time_sec)
                events.append(asdict(event))
    return dict(count=count, events=events, counter_frames=signal_index, start=start, end=end,
                committed=None if policy is None else policy.committed, dropped=None if policy is None else policy.dropped)


def match_events(labels: list[float], predicted: list[float], tol: float) -> tuple[int, int, int]:
    pairs = sorted((abs(p - g), i, j) for i, p in enumerate(predicted) for j, g in enumerate(labels) if abs(p - g) <= tol)
    used_p, used_g = set(), set()
    for _, i, j in pairs:
        if i not in used_p and j not in used_g:
            used_p.add(i)
            used_g.add(j)
    tp = len(used_p)
    return tp, len(predicted) - tp, len(labels) - tp


def summarize(rows: list[dict]) -> dict:
    gt = sum(r["gt"] for r in rows)
    abs_err = sum(abs(r["error"]) for r in rows)
    tp, fp, fn = (sum(r[k] for r in rows) for k in ("tp", "fp", "fn"))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return dict(
        videos=len(rows), gt=gt, predicted=sum(r["count"] for r in rows),
        count_agreement=1 - abs_err / gt if gt else 0.0,
        mae=abs_err / len(rows) if rows else 0.0,
        off_by_one_rate=sum(abs(r["error"]) <= 1 for r in rows) / len(rows) if rows else 0.0,
        signed_error=sum(r["error"] for r in rows),
        precision=precision, recall=recall,
        f1=2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        decrements=sum(r["decrements"] for r in rows),
        decrements_after_last_label=sum(r["decrements_after_last_label"] for r in rows),
        committed=sum(r["committed"] or 0 for r in rows),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--landmarks", default=str(ROOT / "external_data" / "landmarks" / "bench"))
    parser.add_argument("--labels-json", help="external label file; default is the KVA benchmark set")
    parser.add_argument("--rate", action="append", help="counter input rate: 20, 30, 60 or native (repeatable)")
    parser.add_argument("--tol", type=float, default=0.25)
    parser.add_argument("--no-double-resample", action="store_true")
    parser.add_argument("--raw-pose", action="store_true", help="skip the app's plausible_pose filter")
    parser.add_argument("--trip-policy", choices=["resume", "legacy"], default="resume")
    parser.add_argument("--end-pad", type=float, default=0.5, help="seconds evaluated after the last label")
    parser.add_argument("--override", default="{}", help='JSON, e.g. {"basic": {"min_count_gap_frames": 6}}')
    parser.add_argument("--scale-ref", default="{}", help='reference Hz per mode for rate scaling, e.g. {"basic": 30, "alternating": 30}')
    parser.add_argument("--out", default=str(ROOT / "external_data" / "eval"))
    parser.add_argument("--tag", default="baseline")
    args = parser.parse_args()

    landmark_dir = Path(args.landmarks)
    records = json_records(Path(args.labels_json), landmark_dir) if args.labels_json else bench_records(landmark_dir)
    overrides = json.loads(args.override)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    report = dict(created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"), tolerance_sec=args.tol,
                  overrides=overrides, scale_ref=json.loads(args.scale_ref), double_resample=not args.no_double_resample, runs={})

    for rate in args.rate or ["20"]:
        rows = []
        for record in records:
            result = run_record(record, rate, overrides, not args.no_double_resample, json.loads(args.scale_ref), args.raw_pose,
                                args.trip_policy, record.get("end_pad", args.end_pad))
            positives = [e["time_sec"] for e in result["events"] if e["count_delta"] > 0]
            negatives = [e["time_sec"] for e in result["events"] if e["count_delta"] < 0]
            tp, fp, fn = match_events(record["labels_sec"], positives, args.tol)
            gt = len(record["labels_sec"])
            rows.append(dict(id=record["id"], mode=record["mode"], group=record["group"], tier=record.get("tier", "kva"),
                             gt=gt, count=result["count"], error=result["count"] - gt, tp=tp, fp=fp, fn=fn,
                             decrements=len(negatives),
                             decrements_after_last_label=sum(t > record["labels_sec"][-1] for t in negatives),
                             decrement_times=negatives, counter_frames=result["counter_frames"],
                             committed=result["committed"], dropped=result["dropped"]))
        by_mode = {mode: summarize([r for r in rows if r["mode"] == mode]) for mode in ("basic", "alternating", "double")}
        report["runs"][rate] = dict(by_mode={m: s for m, s in by_mode.items() if s["videos"]}, overall=summarize(rows), videos=rows)
        with (out_dir / f"{args.tag}_rate{rate}.csv").open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=[k for k in rows[0] if k != "decrement_times"], extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        print(f"\n== rate {rate} ==")
        for mode, s in report["runs"][rate]["by_mode"].items():
            print(f"{mode:12s} n={s['videos']:2d} gt={s['gt']:5d} pred={s['predicted']:5d} agree={s['count_agreement']:.1%} "
                  f"MAE={s['mae']:.1f} OBO={s['off_by_one_rate']:.0%} P={s['precision']:.1%} R={s['recall']:.1%} "
                  f"F1={s['f1']:.1%} engine-1s={s['decrements']} (after end {s['decrements_after_last_label']}) committed-1s={s['committed']}")
    (out_dir / f"{args.tag}.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
