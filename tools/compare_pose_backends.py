"""Loopback-only comparison harness for the actual browser Worker and legacy CPU.

Run with Python 3.11 and requirements-dev; see benchmark_results/README.md.
The same selected source frames/timestamps feed unchanged counting engines.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import platform
import sys
import threading
import time
import xml.etree.ElementTree as ET
from dataclasses import asdict
from pathlib import Path

import cv2
from fastapi import FastAPI, HTTPException, Response
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.pose_protocol import PoseSamplingClock, decode_pose  # noqa: E402
from app.signal_resampler import SignalResampler  # noqa: E402

BASE_OUT = ROOT / "benchmark_results"
FPS = int(os.environ.get("BENCHMARK_FPS", "60"))
OUT = BASE_OUT / "60fps" if FPS == 60 else BASE_OUT


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def manifest():
    records = []
    sources = [("basic", BASE_OUT / "cache" / "basic"),
               ("alternating", BASE_OUT / "cache" / "alternating"),
               ("double", BASE_OUT / "cache" / "double")]
    for mode, directory in sources:
        if not directory.is_dir():
            raise ValueError(f"missing validation data: {directory}")
        for video in sorted(directory.glob("*.mp4")):
            label = video.with_suffix(".kva")
            if not label.exists():
                continue
            capture = cv2.VideoCapture(str(video))
            fps = capture.get(cv2.CAP_PROP_FPS)
            total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            capture.release()
            xml = ET.parse(label).getroot()
            ticks = float(xml.findtext("AverageTimeStampsPerFrame"))
            labels = sorted({int(round(int(k.findtext("Timestamp")) / ticks))
                             for k in xml.findall("./Keyframes/Keyframe")
                             if any(p.find("./PointList/Point") is not None for p in k.findall("./Drawings/Pencil"))})
            if not labels or not fps or not total:
                raise ValueError(f"invalid clip/labels: {video}")
            start = max(0.0, labels[0] / fps - 0.5)
            end = min((total - 1) / fps, labels[-1] / fps + 0.5)
            indices = [min(total - 1, int(round(k * fps / FPS))) for k in range(int(end * FPS) + 1)]
            records.append(dict(id=f"{mode}_{video.stem}", mode=mode, video=str(video), label=str(label),
                                video_sha256=sha(video), label_sha256=sha(label), gt=len(labels),
                                labels_sec=[i / fps for i in labels], start=start, end=end,
                                source_fps=fps, analysis_fps=FPS, indices=indices))
    return records


class Evaluation:
    def __init__(self, record, legacy):
        self.record = record
        self.module = importlib.import_module(f"{record['mode']}_jump.counter_engine")
        self.config = self.module.EngineConfig()
        self.extractor = self.module.PoseSignalExtractor(self.config, initialize_pose=legacy)
        self.engine = self.module.RealtimeCounterEngine(self.config, enable_realtime_compensation=True)
        self.resampler = SignalResampler() if record["mode"] == "double" else None
        self.sampling_clock = PoseSamplingClock() if not legacy and FPS == 60 else None
        self.signal_index = 0
        self.armed = False
        self.count = 0
        self.timings = []
        self.events = []

    def step(self, signal, sample):
        frames = [signal] if self.resampler is None else self.resampler.push(signal)
        if not self.armed and signal.time_sec >= self.record["start"]:
            hook = {"basic": None, "alternating": "begin_count_phase", "double": "arm_for_counting"}[self.record["mode"]]
            if hook:
                getattr(self.engine, hook)()
            self.armed = True
        for frame in frames:
            if not self.armed:
                self.engine.warmup(frame)
            else:
                event = self.engine.step(frame)
                if event is not None:
                    self.count = int(event.running_count)
                    self.events.append(asdict(event))

    def output(self):
        return dict(counter_input_frames=self.signal_index, count=self.count, gt=self.record["gt"], error=self.count-self.record["gt"],
                    frames=len(self.timings), timings=self.timings, events=self.events)

    def close(self):
        self.extractor.close()


class FrameReader:
    def __init__(self):
        self.cap = None
        self.clip = None
        self.index = -1
        self.lock = threading.Lock()

    def frame(self, record, sample):
        with self.lock:
            index = record["indices"][sample]
            if self.clip != record["id"] or index < self.index:
                if self.cap:
                    self.cap.release()
                self.cap = cv2.VideoCapture(record["video"])
                self.clip, self.index = record["id"], -1
            while self.index < index:
                if not self.cap.grab():
                    raise ValueError("video decode failed")
                self.index += 1
            ok, frame = self.cap.retrieve()
            if not ok:
                raise ValueError("video decode failed")
            return cv2.resize(frame, (512, 288), interpolation=cv2.INTER_LINEAR)


def create_app():
    app = FastAPI()
    records = {r["id"]: r for r in manifest()}
    (OUT / "input_manifest.json").write_text(json.dumps(list(records.values()), indent=2), encoding="utf-8")
    reader = FrameReader()
    evaluations = {}

    @app.middleware("http")
    async def csp(request, call_next):
        response = await call_next(request)
        response.headers["Content-Security-Policy"] = "default-src 'self'; script-src 'self' 'wasm-unsafe-eval'; worker-src 'self'; connect-src 'self'"
        return response

    @app.get("/benchmark/manifest")
    def get_manifest():
        return list(records.values())

    @app.get("/benchmark/frame/{clip}/{sample}")
    def get_frame(clip: str, sample: int):
        record = records.get(clip)
        if not record or not 0 <= sample < len(record["indices"]):
            raise HTTPException(404)
        ok, image = cv2.imencode(".png", reader.frame(record, sample))
        if not ok:
            raise ValueError("PNG encode failed")
        return Response(image.tobytes(), media_type="image/png")

    @app.post("/benchmark/legacy/{clip}")
    def legacy(clip: str):
        record = records[clip]
        if os.environ.get("BENCHMARK_REUSE_LEGACY") == "1":
            previous = json.loads((BASE_OUT / "20fps" / "comparison.json").read_text(encoding="utf-8"))
            result = next(r["legacy"] for r in previous["results"] if r["id"] == clip)
            result["reused_20fps_baseline"] = True
            return result
        evaluator = Evaluation(record, True)
        legacy_record = {**record, "indices": sorted({int(round(k * record["source_fps"] / 20)) for k in range(int(record["end"]*20)+1)})}
        local_reader = FrameReader()
        try:
            for sample, index in enumerate(legacy_record["indices"]):
                frame = local_reader.frame(legacy_record, sample)
                ok, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 62])
                start = time.perf_counter()
                decoded = cv2.imdecode(jpeg, cv2.IMREAD_COLOR)
                signal, _ = evaluator.extractor.process_bgr_frame(decoded, sample, index / record["source_fps"])
                pose_end = time.perf_counter()
                evaluator.step(signal, sample)
                end = time.perf_counter()
                evaluator.timings.append(dict(total_ms=(end-start)*1000, extraction_ms=(pose_end-start)*1000,
                                              engine_ms=(end-pose_end)*1000))
            result = evaluator.output()
            (OUT / f"{clip}_legacy.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
            return result
        finally:
            evaluator.close()
            if local_reader.cap:
                local_reader.cap.release()

    @app.post("/benchmark/start/{clip}")
    def start(clip: str):
        if clip in evaluations:
            evaluations[clip].close()
        evaluations[clip] = Evaluation(records[clip], False)
        return {"ok": True}

    @app.post("/benchmark/pose/{clip}/{sample}")
    def pose(clip: str, sample: int, message: dict):
        start = time.perf_counter()
        evaluator = evaluations[clip]
        ts, inference_ms, result = decode_pose(message)
        expected = sample / FPS
        if abs(ts/1000 - expected) > 0.0001:
            raise ValueError("timestamp mismatch")
        if evaluator.sampling_clock is None or evaluator.sampling_clock.accept(ts/1000):
            signal, _ = evaluator.extractor.process_pose_result(result, evaluator.signal_index, ts/1000)
            evaluator.step(signal, evaluator.signal_index)
            evaluator.signal_index += 1
        server_ms = (time.perf_counter()-start)*1000
        evaluator.timings.append(dict(server_ms=server_ms, inference_ms=inference_ms, total_ms=server_ms+inference_ms))
        return {"count": evaluator.count, "serverMs": server_ms}

    @app.post("/benchmark/finish/{clip}")
    def finish(clip: str):
        evaluator = evaluations.pop(clip)
        result = evaluator.output()
        evaluator.close()
        (OUT / f"{clip}_gpu.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        return result

    app.mount("/", StaticFiles(directory=ROOT / "static", html=True))
    return app


if __name__ == "__main__":
    import uvicorn
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8766)
    args = parser.parse_args()
    OUT.mkdir(exist_ok=True)
    (OUT / "protocol.json").write_text(json.dumps(dict(
        fps=FPS, counter_input_hz=20, size=[512, 288], model="pose_landmarker_full float16 version 1", warmup_excluded=10,
        legacy_jpeg_quality=62, legacy_fps=20, reused_legacy=os.environ.get("BENCHMARK_REUSE_LEGACY") == "1", gpu_input="lossless PNG of the same resized source frame (no JPEG in the new app)",
        sampling="60Hz input clock with nearest original frame (30/24fps source frames repeat); counter samples 20Hz before signal conversion",
        window="first nonempty KVA event - 0.5s through last + 0.5s; preceding frames warm up engine",
        gate="readiness/countdown bypassed equally for labelled-window comparison; production tested separately",
        labels="one unique nonempty KVA keyframe is one jump; multiple pencils are not multiple jumps",
        timing="legacy: JPEG decode+pose+engine; GPU: worker inference+server adapter+engine. Disk/video decode/network excluded; first 10 frames per clip excluded from timing only",
        python=sys.version, platform=platform.platform(),
        source_hashes={str(p.relative_to(ROOT)): sha(p) for p in [ROOT / 'static/pose-worker.js', ROOT / 'app/pose_protocol.py', *[ROOT / f'{m}_jump/counter_engine.py' for m in ('basic','alternating','double')]]}
    ), indent=2), encoding="utf-8")
    uvicorn.run(create_app(), host="127.0.0.1", port=args.port, log_level="warning")
