"""Turn human-verified annotation clips into evaluation labels.

Reads external_data/annotation/manifest.csv; only rows with verified=yes are used. Each clip KVA
is read with the benchmark rule (one non-empty keyframe = one counted jump), except keyframes named
"trip" or "걸림" by the person, which are recorded as rope-catch events and not counted. The automatic hint
"trip?" (a touchdown after a pause, see propose_labels.py) is a jump unless the person renamed it.
Clip-relative times are shifted back by the segment start so they line up with the cached landmarks.
An optional use_until_sec (clip-relative) ends the clip early, e.g. where the camera or the person turns side-on.

Output: external_data/labels/verified.json for tools/eval/evaluate_engines.py --labels-json.
"""
from __future__ import annotations

import csv
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ANNOTATION = ROOT / "external_data" / "annotation"
OUT = ROOT / "external_data" / "labels" / "verified.json"


def read_clip_kva(path: Path) -> tuple[list[float], list[float]]:
    root = ET.parse(path).getroot()
    jumps, trips = set(), set()
    for keyframe in root.findall("./Keyframes/Keyframe"):
        if not any(p.find("./PointList/Point") is not None for p in keyframe.findall("./Drawings/Pencil")):
            continue
        time_sec = round(int(keyframe.findtext("Timestamp")) / 1000.0, 3)  # ms timestamps (template format)
        name = (keyframe.findtext("Name") or "").lower()
        is_trip = "걸림" in name or re.search(r"\btrip\b(?!\?)", name) is not None
        (trips if is_trip else jumps).add(time_sec)
    return sorted(jumps), sorted(trips)


def main() -> None:
    with (ANNOTATION / "manifest.csv").open(encoding="utf-8-sig") as handle:
        rows = [row for row in csv.DictReader(handle) if row.get("verified", "").strip().lower() == "yes"]
    records = []
    for row in rows:
        clip = ANNOTATION / row["clip"]
        jumps, trips = read_clip_kva(clip.with_suffix(".kva"))
        until = float(row["use_until_sec"]) if (row.get("use_until_sec") or "").strip() else None
        if until is not None:
            jumps, trips = [t for t in jumps if t <= until], [t for t in trips if t <= until]
        if not jumps:
            continue
        offset = float(row["segment_start_sec"])
        end = offset + until if until is not None else float(row["segment_end_sec"])
        video_id = clip.stem.rsplit("_seg", 1)[0]  # not row["video_id"]: spreadsheets turn ids like "-caOW..." into "#NAME?"
        records.append(dict(
            id=clip.stem, video_id=video_id, mode=row["mode"], group=video_id, tier="human_verified",
            segment=[offset, end],
            labels_sec=[round(offset + t, 3) for t in jumps],
            trips_sec=[round(offset + t, 3) for t in trips],
        ))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(records, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"{len(records)} verified clips, {sum(len(r['labels_sec']) for r in records)} jumps, "
          f"{sum(len(r['trips_sec']) for r in records)} rope catches -> {OUT}")


if __name__ == "__main__":
    main()
