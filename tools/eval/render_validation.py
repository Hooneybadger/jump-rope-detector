"""Render validation videos: ground-truth labels vs the app's counts, overlaid on the source video at 60 fps.

For every labelled benchmark video and every human-verified external clip the engines run exactly as in the
app (basic and double 30 Hz, alternating every frame, plausible_pose filter, RopeTripPolicy) on the cached
landmarks. Labels and counts are matched 1:1 within +-0.25 s (tools/eval/evaluate_engines.match rule):
  white ring   ground-truth jump (label)          green dot   counted and matched
  red dot      counted without a label (false)    orange ring label not counted (missed)
  purple -1    rope-catch deduction shown to the user
The header shows running totals, the bottom bar the whole window with a playhead, and external clips carry
their CC BY attribution. Output: validation_video/{benchmark,external}/<id>.mp4 (60 fps, no audio).
These files are for verification only; nothing here is shown by the app.

    python tools/eval/render_validation.py [--only basic_05 ...] [--ids-file ids.txt] [--workers 4]
"""
from __future__ import annotations

import argparse
import csv
import importlib
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools" / "external_data"))
from app.pose_protocol import PoseSamplingClock, plausible_pose  # noqa: E402
from app.rope_trip import RopeTripPolicy  # noqa: E402
from app.signal_resampler import SignalResampler  # noqa: E402
from tools.eval.evaluate_engines import HOOKS, WARMUP_SEC, bench_records, json_records, pose_result  # noqa: E402

OUT = ROOT / "validation_video"
FPS_OUT = 60.0
APP_RATE = {"basic": 30.0, "alternating": None, "double": 30.0}  # app/jump_service.py COUNTER_INPUT_HZ
MODE_NAMES = {"basic": "모아뛰기", "alternating": "번갈아뛰기", "double": "이중뛰기"}
TOL, FLASH_SEC, END_PAD, LEAD_SEC = 0.25, 0.30, 3.0, 1.0
FONT = "C:/Windows/Fonts/malgun.ttf"
EDGES = [(11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (11, 23), (12, 24), (23, 24), (23, 25), (25, 27),
         (27, 29), (29, 31), (27, 31), (24, 26), (26, 28), (28, 30), (30, 32), (28, 32)]
WHITE, GREEN, RED, ORANGE, PURPLE = (255, 255, 255), (60, 200, 90), (60, 60, 235), (0, 160, 255), (200, 80, 200)  # BGR


def run_app(record: dict) -> tuple[list[tuple[float, int, int]], float, float]:
    """Engine run as in the app. Returns [(time, displayed_count, delta)] for every change of the shown count."""
    mode = record["mode"]
    module = importlib.import_module(f"{mode}_jump.counter_engine")
    config = module.EngineConfig()
    extractor = module.PoseSignalExtractor(config, initialize_pose=False)
    engine = module.RealtimeCounterEngine(config, enable_realtime_compensation=True)
    resampler = SignalResampler() if mode == "double" else None
    clock = PoseSamplingClock(APP_RATE[mode]) if APP_RATE[mode] else None
    data = np.load(record["npz"])
    times, landmarks = data["t"], data["lm"]
    if hasattr(engine, "frame_height_px"):
        engine.frame_height_px = float(data["height"]) * float(data["inference_scale"])
    labels = record["labels_sec"]
    seg_start, seg_end = record.get("segment") or (0.0, float(times[-1]))
    start, end = max(seg_start, labels[0] - 0.5), min(seg_end, labels[-1] + END_PAD)
    policy, armed, shown, index, changes = RopeTripPolicy(), False, 0, 0, []
    for t, row in zip(times, landmarks):
        if t < seg_start - WARMUP_SEC:
            continue
        if t > end:
            break
        if clock is not None and not clock.accept(float(t)):
            continue
        signal, _ = extractor.process_pose_result(plausible_pose(pose_result(row)), index, float(t))
        index += 1
        if not armed and t >= start:
            if HOOKS[mode]:
                getattr(engine, HOOKS[mode])()
            armed = True
        for frame in ([signal] if resampler is None else resampler.push(signal)):
            if not armed:
                engine.warmup(frame)
                continue
            policy.expire(frame.time_sec)
            event = engine.step(frame)
            if event is not None:
                new = policy.observe(event, frame.time_sec)
                if new != shown:
                    changes.append((float(frame.time_sec), new, new - shown))
                    shown = new
    return changes, start, end


def match(labels: list[float], counts: list[float]) -> tuple[dict[int, int], dict[int, int]]:
    pairs = sorted((abs(p - g), i, j) for i, p in enumerate(counts) for j, g in enumerate(labels) if abs(p - g) <= TOL)
    by_count, by_label = {}, {}
    for _, i, j in pairs:
        if i not in by_count and j not in by_label:
            by_count[i], by_label[j] = j, i
    return by_count, by_label


def foot_point(data, time_sec: float, width: int, height: int) -> tuple[int, int] | None:
    index = int(np.argmin(np.abs(data["t"] - time_sec)))
    row = data["lm"][index]
    if not np.isfinite(row[[31, 32], :2]).all():
        return None
    x, y = (row[31, :2] + row[32, :2]) / 2
    return int(x * width), int(y * height)


def text(draw: ImageDraw.ImageDraw, xy, value: str, size: int, fill=(255, 255, 255)) -> None:
    draw.text(xy, value, font=ImageFont.truetype(FONT, size), fill=fill, stroke_width=max(1, size // 12), stroke_fill=(0, 0, 0))


def ffmpeg_exe() -> str:
    from propose_labels import _ffmpeg

    return _ffmpeg()


def render(job: dict) -> str:
    record, video, out, credit = job["record"], Path(job["video"]), Path(job["out"]), job.get("credit")
    changes, start, end = run_app(record)
    labels = list(record["labels_sec"])
    counts = [t for t, _, d in changes if d > 0]
    by_count, by_label = match(labels, counts)
    data = np.load(record["npz"])
    capture = cv2.VideoCapture(str(video))
    src_fps = capture.get(cv2.CAP_PROP_FPS) or float(data["fps"])
    src_w, src_h = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    scale = min(1.0, 1280 / max(src_w, src_h))
    out_w, out_h = int(src_w * scale) // 2 * 2, int(src_h * scale) // 2 * 2
    small = max(12, out_w // 75)
    font_px = max(14, min(30, out_w // 30))  # sized by width so portrait videos fit
    head_h, bar_h = 3 * font_px + 18, 36 + 2 * small + (2 * (small + 4) if credit else 0)
    canvas_h = (out_h + head_h + bar_h + 1) // 2 * 2  # yuv420p needs even dimensions
    out.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.Popen([ffmpeg_exe(), "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{out_w}x{canvas_h}",
                             "-r", str(FPS_OUT), "-i", "-", "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
                             "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out)], stdin=subprocess.PIPE)
    t0, t1 = max(0.0, start - LEAD_SEC), end
    events = ([(t, "count_ok" if i in by_count else "count_false") for i, t in enumerate(counts)]
              + [(t, "label") for t in labels] + [(labels[j] + TOL, "missed") for j in range(len(labels)) if j not in by_label]
              + [(t, "minus") for t, _, d in changes if d < 0])
    src_index, frame = -1, None
    for k in range(int((t1 - t0) * FPS_OUT) + 1):
        now = t0 + k / FPS_OUT
        want = int(now * src_fps)  # nearest earlier source frame -> exact 60 fps timing for 24/25/30/60 fps sources
        if want - src_index > 5:  # first frame or a gap: seek
            capture.set(cv2.CAP_PROP_POS_FRAMES, want)
            src_index = want - 1
        while src_index < want:
            ok, image = capture.read()
            src_index += 1
            if ok:
                frame = image
        if frame is None:
            continue
        view = cv2.resize(frame, (out_w, out_h))
        # skeleton (light)
        row = data["lm"][int(np.argmin(np.abs(data["t"] - now)))]
        if np.isfinite(row[:, :2]).all():
            pts = [(int(x * out_w), int(y * out_h)) for x, y in row[:, :2]]
            for a, b in EDGES:
                cv2.line(view, pts[a], pts[b], (230, 230, 230), 1, cv2.LINE_AA)
        # event flashes at the feet
        for t, kind in events:
            if not (t <= now <= t + FLASH_SEC):
                continue
            point = foot_point(data, t, out_w, out_h)
            if point is None:
                continue
            fade = 1.0 - (now - t) / FLASH_SEC
            radius = max(8, out_h // 40)
            if kind == "label":
                cv2.circle(view, point, int(radius * 1.6), WHITE, 2, cv2.LINE_AA)
            elif kind == "count_ok":
                cv2.circle(view, point, int(radius * (0.6 + 0.4 * fade)), GREEN, -1, cv2.LINE_AA)
            elif kind == "count_false":
                cv2.circle(view, point, int(radius * (0.6 + 0.4 * fade)), RED, -1, cv2.LINE_AA)
            elif kind == "missed":
                cv2.circle(view, point, int(radius * 2.1), ORANGE, 3, cv2.LINE_AA)
            elif kind == "minus":
                cv2.putText(view, "-1", (point[0] + radius * 2, point[1] - radius * 2), cv2.FONT_HERSHEY_SIMPLEX, 1.2, PURPLE, 3, cv2.LINE_AA)
        canvas = np.zeros((canvas_h, out_w, 3), np.uint8)
        canvas[head_h:head_h + out_h] = view
        # bottom timeline
        y0 = head_h + out_h + 8
        span = max(1e-6, t1 - t0)
        cv2.rectangle(canvas, (8, y0), (out_w - 8, y0 + 26), (40, 40, 40), -1)
        tx = lambda t: int(8 + (out_w - 16) * (t - t0) / span)  # noqa: E731
        for t in labels:
            cv2.line(canvas, (tx(t), y0), (tx(t), y0 + 10), WHITE, 1)
        for i, t in enumerate(counts):
            cv2.line(canvas, (tx(t), y0 + 13), (tx(t), y0 + 25), GREEN if i in by_count else RED, 1 if i in by_count else 2)
        for j, t in enumerate(labels):
            if j not in by_label:
                cv2.line(canvas, (tx(t), y0), (tx(t), y0 + 25), ORANGE, 2)
        cv2.line(canvas, (tx(now), y0 - 3), (tx(now), y0 + 29), (0, 255, 255), 2)
        # header and credit (PIL for Korean text)
        image = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(image)
        shown = next((c for t, c, _ in reversed(changes) if t <= now), 0)
        gt_now = sum(1 for t in labels if t <= now)
        ok_now = sum(1 for i, t in enumerate(counts) if t <= now and i in by_count)
        false_now = sum(1 for i, t in enumerate(counts) if t <= now and i not in by_count)
        miss_now = sum(1 for j, t in enumerate(labels) if t + TOL <= now and j not in by_label)
        rate = "30Hz" if APP_RATE[record["mode"]] else f"{src_fps:.0f}fps 전체"
        text(draw, (8, 4), f"[검증용] {MODE_NAMES[record['mode']]} · {record['id']} · 입력 {rate}", font_px - 4)
        state = "측정 전" if now < start else ("측정 종료" if now > end - 1e-6 else "측정 중")
        text(draw, (8, 6 + font_px), f"정답 {gt_now} · 앱 표시 {shown} · {state}", font_px)
        text(draw, (8, 8 + 2 * font_px), f"맞음 {ok_now} · 오탐 {false_now} · 누락 {miss_now}", font_px)
        text(draw, (8, y0 + 30), "흰선/흰 고리 정답 · 초록 맞음 · 빨강 오탐 · 주황 누락 · 보라 -1", small, (220, 220, 220))
        for n, line in enumerate(credit or []):
            text(draw, (8, y0 + 36 + small + n * (small + 4)), line, small, (220, 220, 220))
        canvas = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
        proc.stdin.write(canvas.tobytes())
    capture.release()
    proc.stdin.close()
    proc.wait()
    final = changes[-1][1] if changes else 0
    return f"{record['id']}: labels {len(labels)} shown {final} ok {len(by_count)} false {len(counts) - len(by_count)} missed {len(labels) - len(by_label)}"


def jobs(only: set[str]) -> list[dict]:
    items = []
    for record in bench_records(ROOT / "external_data" / "landmarks" / "bench"):
        mode, number = record["id"].split("_")
        items.append(dict(record=record, video=str(ROOT / "benchmark_results" / "cache" / mode / f"{number}.mp4"),
                          out=str(OUT / "benchmark" / f"{record['id']}.mp4")))
    with (ROOT / "external_data" / "manifest" / "sources.csv").open(encoding="utf-8-sig") as handle:
        sources = {row["id"]: row for row in csv.DictReader(handle)}
    for record in json_records(ROOT / "external_data" / "labels" / "verified.json", ROOT / "external_data" / "landmarks" / "cc"):
        source = sources.get(record["video_id"], {})
        credit = [f"출처: {source.get('title', '')[:50]} · {source.get('channel', '')}", f"라이선스: CC BY · {source.get('url', '')}"]
        items.append(dict(record=record, video=str(ROOT / "external_data" / "media" / f"{record['video_id']}.mp4"),
                          out=str(OUT / "external" / f"{record['id']}.mp4"), credit=credit))
    return [j for j in items if not only or j["record"]["id"] in only]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", nargs="*", default=[])
    parser.add_argument("--ids-file", help="one id per line (ids may start with '-', which --only cannot take)")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    only = set(args.only) | (set(Path(args.ids_file).read_text(encoding="utf-8").split()) if args.ids_file else set())
    work = jobs(only)
    with ProcessPoolExecutor(args.workers) as pool:
        for line in pool.map(render, work):
            print(line, flush=True)


if __name__ == "__main__":
    main()
