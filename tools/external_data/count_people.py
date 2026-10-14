"""Flag videos in which more than one person appears (only single-person videos are used).

Samples one frame every --step seconds, runs the same PoseLandmarker model in IMAGE mode with up to
four poses, and records how many sampled frames contain two or more people. Results go to
external_data/screening/people.json; a video is "multi_person" when any two sampled frames show
two or more people (one stray frame is tolerated as a detection glitch).

    python tools/external_data/count_people.py
"""
from __future__ import annotations

import json
from multiprocessing import Pool
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[2]
MODEL = ROOT / "static" / "vendor" / "mediapipe" / "pose_landmarker_full.task"
OUT = ROOT / "external_data" / "screening" / "people.json"


def count(job: tuple[str, float]) -> tuple[str, dict]:
    video_path, step = job
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions
    from mediapipe.tasks.python.vision import PoseLandmarker, PoseLandmarkerOptions, RunningMode

    options = PoseLandmarkerOptions(base_options=BaseOptions(model_asset_path=str(MODEL)), running_mode=RunningMode.IMAGE,
                                    num_poses=4, min_pose_detection_confidence=0.5, min_pose_presence_confidence=0.5)
    capture = cv2.VideoCapture(video_path)
    duration = (capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0) / (capture.get(cv2.CAP_PROP_FPS) or 30.0)
    samples, multi, times = 0, 0, []
    with PoseLandmarker.create_from_options(options) as landmarker:
        at = 0.0
        while at < duration:
            capture.set(cv2.CAP_PROP_POS_MSEC, at * 1000)
            ok, frame = capture.read()
            if not ok:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            people = len(landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)).pose_landmarks)
            samples += 1
            if people >= 2:
                multi += 1
                times.append(round(at, 1))
            at += step
    capture.release()
    return Path(video_path).stem, dict(samples=samples, multi_person_samples=multi,
                                       multi_person=multi >= 2, example_times=times[:10])


def main() -> None:
    results = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {}
    jobs = [(str(p), 0.5) for p in sorted((ROOT / "external_data" / "media").glob("*.mp4"))
            if p.stem not in results and "." not in p.stem]
    with Pool(4) as pool:
        for stem, result in pool.imap_unordered(count, jobs):
            results[stem] = result
            OUT.write_text(json.dumps(results, indent=1), encoding="utf-8")
            print(stem, result["multi_person_samples"], "/", result["samples"], "MULTI" if result["multi_person"] else "single", flush=True)


if __name__ == "__main__":
    main()
