"""Extract MediaPipe PoseLandmarker landmarks once per video and cache them as .npz.

Uses the same model file as the browser app (static/vendor/mediapipe/pose_landmarker_full.task)
in VIDEO mode with numPoses=1 and 0.5 confidences, at the video's native frame rate and
native aspect ratio (frames are only downscaled so the long side is at most --max-side).

Output per video: t (seconds), lm (frames, 33, 5: x, y, z, visibility, presence; NaN when no pose),
plus fps, width, height, and provenance fields.

Run with the Python 3.11 + mediapipe 0.10.21 environment, e.g.
    python tools/external_data/cache_landmarks.py --bench
    python tools/external_data/cache_landmarks.py --videos external_data/media/*.mp4 --out external_data/landmarks/cc
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import sys
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
MODEL = ROOT / "static" / "vendor" / "mediapipe" / "pose_landmarker_full.task"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def extract(job: tuple[str, str, int]) -> str:
    video_path, out_path, max_side = job
    video, out = Path(video_path), Path(out_path)
    if out.exists():
        return f"skip {out.name}"
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions
    from mediapipe.tasks.python.vision import PoseLandmarker, PoseLandmarkerOptions, RunningMode

    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(MODEL)),
        running_mode=RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    capture = cv2.VideoCapture(str(video))
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 30.0)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    scale = min(1.0, max_side / max(width, height))
    times, landmarks = [], []
    index = 0
    with PoseLandmarker.create_from_options(options) as landmarker:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if scale < 1.0:
                frame = cv2.resize(frame, (round(width * scale), round(height * scale)), interpolation=cv2.INTER_AREA)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            timestamp_ms = int(round(index * 1000.0 / fps))
            result = landmarker.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), timestamp_ms)
            row = np.full((33, 5), np.nan, dtype=np.float32)
            if result.pose_landmarks:
                row[:] = [[p.x, p.y, p.z, p.visibility, p.presence] for p in result.pose_landmarks[0]]
            times.append(index / fps)
            landmarks.append(row)
            index += 1
    capture.release()
    out.parent.mkdir(parents=True, exist_ok=True)
    partial = out.with_name(out.stem + ".partial.npz")  # write then rename: no half-written cache if two runs overlap
    np.savez_compressed(
        partial,
        t=np.asarray(times, dtype=np.float64),
        lm=np.stack(landmarks) if landmarks else np.zeros((0, 33, 5), np.float32),
        fps=fps,
        width=width,
        height=height,
        inference_scale=scale,
        video_sha256=_sha256(video),
        model_sha256=_sha256(MODEL),
        mediapipe_version=mp.__version__,
    )
    partial.replace(out)
    detected = float(np.mean(~np.isnan(np.stack(landmarks)[:, 0, 0]))) if landmarks else 0.0
    return f"done {out.name}: {index} frames @ {fps:.2f}fps, detected {detected:.1%}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bench", action="store_true", help="cache benchmark_results/cache/*/*.mp4")
    parser.add_argument("--videos", nargs="*", default=[])
    parser.add_argument("--out", default=str(ROOT / "external_data" / "landmarks" / "bench"))
    parser.add_argument("--max-side", type=int, default=1280)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    jobs = []
    if args.bench:
        for video in sorted((ROOT / "benchmark_results" / "cache").glob("*/*.mp4")):
            jobs.append((str(video), str(Path(args.out) / f"{video.parent.name}_{video.stem}.npz"), args.max_side))
    for pattern in args.videos:
        for name in sorted(glob.glob(pattern)):
            video = Path(name)
            jobs.append((str(video), str(Path(args.out) / f"{video.stem}.npz"), args.max_side))
    if not jobs:
        sys.exit("no videos selected")
    with Pool(args.workers) as pool:
        for message in pool.imap_unordered(extract, jobs):
            print(message, flush=True)


if __name__ == "__main__":
    main()
