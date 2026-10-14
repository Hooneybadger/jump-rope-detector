"""Build a Kinovea annotation package from visually accepted segments.

For every segment marked basic/alternating/double in external_data/screening/review.json:
  * cut a clip (at most --max-seconds, re-encoded so frame times are exact) to
    external_data/annotation/clips/<id>_seg<k>.mp4,
  * write <id>_seg<k>.kva next to it, using an existing benchmark KVA as the structural template,
    with one keyframe per touchdown candidate named by its review tag (ok / check / trip? / double? / single? / first),
    to be corrected by a person. Each keyframe carries dots at the landing position taken from the cached
    landmarks: basic = midpoint of the two toe tips, alternating = toe tip of the landing foot (the name gets
    "왼발"/"오른발", the person's own side), double = one dot on each toe tip,
  * list everything (with source URL, uploader and license for attribution) in manifest.csv.

Times inside the clip KVA are clip-relative; import_verified_labels.py adds the segment start back.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import uuid
from pathlib import Path

import numpy as np

from propose_labels import _ffmpeg, candidate_events

ROOT = Path(__file__).resolve().parents[2]
TEMPLATE = ROOT / "benchmark_results" / "cache" / "basic" / "02.kva"
OUT = ROOT / "external_data" / "annotation"
# ARGB colours: ok/double? = green, check = red, trip? = purple, single? = yellow, first = orange
TAG_COLORS = {"ok": "255;46;204;113", "double?": "255;46;204;113", "check": "255;231;76;60", "trip?": "255;155;89;182",
              "single?": "255;241;196;15", "first": "255;255;140;0"}


L_TOE, R_TOE = 31, 32  # MediaPipe foot_index (toe tip), the person's own left/right
FOOT_NAMES = {"left": "왼발", "right": "오른발"}


def marker_points(data, time_sec: float, mode: str, foot: str) -> list[tuple[int, int]]:
    """Pixel positions of the dots for one touchdown, from the cached landmarks at that frame:
    basic = midpoint of the two toe tips, alternating = toe tip of the landing foot, double = both toe tips."""
    width, height = int(data["width"]), int(data["height"])
    index = int(np.argmin(np.abs(data["t"] - time_sec)))
    for offset in (0, -1, 1, -2, 2, -3, 3):  # nearest frame with a pose
        row = data["lm"][min(max(index + offset, 0), len(data["lm"]) - 1)]
        if np.isfinite(row[[L_TOE, R_TOE], :2]).all():
            break
    else:
        return [(width // 2, int(height * 0.93))]
    left, right = (row[L_TOE, :2] * (width, height), row[R_TOE, :2] * (width, height))
    if mode == "alternating" and foot in ("left", "right"):
        points = [left if foot == "left" else right]
    elif mode == "double":
        points = [left, right]
    else:
        points = [(left + right) / 2]
    return [(int(round(x)), int(round(y))) for x, y in points]


def pencil_xml(name: str, argb: str, x: int, y: int) -> str:
    return f"""        <Pencil id="{uuid.uuid4()}" name="{name}">
          <PointList Count="2">
            <Point>{x};{y}</Point>
            <Point>{x + 1};{y}</Point>
          </PointList>
          <DrawingStyle>
            <Color Key="color">
              <Value>{argb}</Value>
            </Color>
            <PenSize Key="pen size">
              <Value>10</Value>
            </PenSize>
            <PenShape Key="pen shape">
              <Value>Solid</Value>
            </PenShape>
          </DrawingStyle>
          <InfosFading>
            <Enabled>true</Enabled>
            <UseDefault>true</UseDefault>
            <AlwaysVisible>false</AlwaysVisible>
            <MasterFactor>1</MasterFactor>
            <OpaqueFrames>1</OpaqueFrames>
            <Frames>20</Frames>
          </InfosFading>
        </Pencil>"""


def keyframe_xml(time_ms: int, name: str, tag: str, points: list[tuple[int, int]]) -> str:
    argb = TAG_COLORS.get(tag, TAG_COLORS["first"])
    pencils = "\n".join(pencil_xml(name, argb, x, y) for x, y in points)
    return f"""    <Keyframe id="{uuid.uuid4()}">
      <Timestamp>{time_ms}</Timestamp>
      <Name>{name}</Name>
      <Color>{argb.split(";", 1)[1]}</Color>
      <Drawings>
{pencils}
      </Drawings>
    </Keyframe>"""


def write_kva(path: Path, clip: Path, fps: float, width: int, height: int, duration: float, events: list[dict]) -> None:
    template = TEMPLATE.read_text(encoding="utf-8-sig")
    head = template[: template.index("<Keyframes>")]
    tail = template[template.index("</Keyframes>"):]
    head = re.sub(r"<OriginalFilename>.*?</OriginalFilename>", f"<OriginalFilename>{clip.stem}</OriginalFilename>", head)
    head = re.sub(r"<FullPath>.*?</FullPath>", f"<FullPath>{clip.name}</FullPath>", head)
    head = re.sub(r"<ImageSize>.*?</ImageSize>", f"<ImageSize>{width};{height}</ImageSize>", head)
    head = re.sub(r"<AverageTimeStampsPerFrame>.*?</AverageTimeStampsPerFrame>",
                  f"<AverageTimeStampsPerFrame>{int(round(1000 / fps))}</AverageTimeStampsPerFrame>", head)
    head = re.sub(r"<CaptureFramerate>.*?</CaptureFramerate>", f"<CaptureFramerate>{fps:.2f}</CaptureFramerate>", head)
    head = re.sub(r"<UserFramerate>.*?</UserFramerate>", f"<UserFramerate>{fps:.2f}</UserFramerate>", head)
    head = re.sub(r"<SelectionEnd>.*?</SelectionEnd>", f"<SelectionEnd>{int(duration * 1000)}</SelectionEnd>", head)
    head = re.sub(r"<A>.*?</A>", f"<A>{width // 2};{height // 2}</A>", head)
    head = re.sub(r"<B>.*?</B>", f"<B>{width // 2 + 100};{height // 2}</B>", head)
    frames = [keyframe_xml(int(round(e["time_sec"] * 1000)), e["name"], e["tag"], e["points"]) for e in events]
    path.write_text(head + "<Keyframes>\n" + "\n".join(frames) + "\n  " + tail, encoding="utf-8")


def clip_seconds(path: Path) -> float:
    import cv2

    capture = cv2.VideoCapture(str(path))
    seconds = (capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0) / (capture.get(cv2.CAP_PROP_FPS) or 30.0)
    capture.release()
    return seconds


def cut_clip(video: Path, start: float, end: float, out: Path) -> None:
    cmd = [_ffmpeg(), "-v", "error", "-y", "-ss", f"{start:.3f}", "-i", str(video), "-t", f"{end - start:.3f}",
           "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-an", "-movflags", "+faststart", str(out)]
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-seconds", type=float, default=30.0)
    args = parser.parse_args()
    review = json.loads((ROOT / "external_data" / "screening" / "review.json").read_text(encoding="utf-8"))
    segments = {item["id"]: item for item in json.loads((ROOT / "external_data" / "screening" / "segments.json").read_text(encoding="utf-8"))}
    with (ROOT / "external_data" / "manifest" / "sources.csv").open(encoding="utf-8-sig") as handle:
        sources = {row["id"]: row for row in csv.DictReader(handle)}
    clips = OUT / "clips"
    clips.mkdir(parents=True, exist_ok=True)
    previous = {}
    if (OUT / "manifest.csv").exists():  # keep human work: verified clips keep their KVA, verified flag and notes
        with (OUT / "manifest.csv").open(encoding="utf-8-sig") as handle:
            previous = {row["clip"]: row for row in csv.DictReader(handle)}
    rows = []
    for key, decision in review.items():
        if key.startswith("_") or decision not in {"basic", "alternating", "double"}:
            continue
        video_id, seg = key.rsplit("_seg", 1)
        item = segments.get(video_id)
        if item is None or int(seg) >= len(item["segments"]) or video_id not in sources:
            continue
        segment = item["segments"][int(seg)]
        start = segment["start_sec"]
        end = min(segment["end_sec"], start + args.max_seconds)
        clip = clips / f"{key}.mp4"
        if not clip.exists() or abs(clip_seconds(clip) - (end - start)) > 0.5:  # re-cut stale clips (segments renumbered)
            cut_clip(ROOT / "external_data" / "media" / f"{video_id}.mp4", start, end, clip)
        old = previous.get(f"clips/{key}.mp4", {})
        if old.get("verified", "").strip().lower() == "yes" and (clips / f"{key}.kva").exists():
            rows.append(dict(old, video_id=video_id))  # repair ids a spreadsheet may have mangled
            print(key, decision, "verified, kept as is", flush=True)
            continue
        npz = ROOT / "external_data" / "landmarks" / "cc" / f"{video_id}.npz"
        events, _ = candidate_events(ROOT / "external_data" / "media" / f"{video_id}.mp4", npz, start, end, decision)
        data = np.load(npz)
        events = [dict(e, time_sec=round(e["time_sec"] - start, 3), points=marker_points(data, e["time_sec"], decision, e["foot"]),
                       name=f"{e['tag']} {FOOT_NAMES[e['foot']]}" if decision == "alternating" and e["foot"] in FOOT_NAMES else e["tag"])
                  for e in events if start <= e["time_sec"] <= end]
        write_kva(clips / f"{key}.kva", clip, item["fps"], item["width"], item["height"], end - start, events)
        source = sources[video_id]
        rows.append(dict(clip=f"clips/{key}.mp4", mode=decision, video_id=video_id, segment_start_sec=start,
                         segment_end_sec=round(end, 3), candidates=len(events),
                         needs_check=sum(e["tag"] in ("check", "single?", "trip?") for e in events), verified=old.get("verified", "no"), notes=old.get("notes", ""),
                         use_until_sec=old.get("use_until_sec", ""),
                         url=source["url"], title=source["title"], channel=source["channel"], license=source["license"]))
        print(key, decision, f"{end - start:.1f}s", f"{len(events)} candidates", flush=True)
    # double-under clips first: they are the only external check of the hand-ratio rule
    rows.sort(key=lambda row: row["mode"] != "double")
    for number, row in enumerate(rows, 1):
        row["priority"] = number
    with (OUT / "manifest.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=["priority"] + [k for k in rows[0] if k != "priority"] if rows else ["clip"])
        writer.writeheader()
        writer.writerows(rows)
    total = sum(float(r["segment_end_sec"]) - float(r["segment_start_sec"]) for r in rows)
    print(f"{len(rows)} clips, {total / 60:.1f} min of video", flush=True)


if __name__ == "__main__":
    np.seterr(all="ignore")
    main()
