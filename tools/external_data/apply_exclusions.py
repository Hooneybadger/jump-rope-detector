"""Apply the external-data exclusion rules and clean everything derived from excluded material.

Rules (user decisions, 2026-10-14):
  * only single-person videos: any video with two or more people in >= 2 sampled frames
    (count_people.py, people.json) is excluded;
  * only jumping scenes longer than 15 s: segments.json must be produced with --min-seconds 15;
    a video without any such scene is excluded and its media, landmarks and contact sheet are deleted
    and it is dropped from sources.csv / ATTRIBUTION.md.

Excluded ids go to manifest/excluded_ids.txt ("<id> <reason>") so they are never downloaded again; lines
added by hand (e.g. "<id> multi_person_visual" for a passer-by the detector missed) are kept.
review.json entries are re-keyed to the new segment numbering by segment start time; entries for
excluded videos or for scenes of 15 s or less are removed, as are their annotation clips, KVA labels
and proposals.

    python tools/external_data/apply_exclusions.py --old-segments external_data/screening/segments_min5.json
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "external_data"


def load_json(path: Path, default):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--old-segments", required=True, help="segments.json the current review.json was keyed on")
    args = parser.parse_args()
    people = load_json(DATA / "screening" / "people.json", {})
    old_segments = {i["id"]: i["segments"] for i in load_json(Path(args.old_segments), [])}
    new_segments = {i["id"]: i["segments"] for i in load_json(DATA / "screening" / "segments.json", [])}
    review = load_json(DATA / "screening" / "review.json", {})

    media_ids = sorted(p.stem for p in (DATA / "media").glob("*.mp4") if "." not in p.stem)
    excluded: dict[str, str] = {}
    for line in (DATA / "manifest" / "excluded_ids.txt").read_text(encoding="utf-8").splitlines() if (DATA / "manifest" / "excluded_ids.txt").exists() else []:
        if line.strip():
            vid, reason = (line.split(maxsplit=1) + ["?"])[:2]
            excluded[vid] = reason
    for vid in media_ids:
        if vid in people and people[vid]["multi_person"]:
            excluded.setdefault(vid, "multi_person")
        elif vid in new_segments and not new_segments[vid]:
            excluded.setdefault(vid, "no_jumping_scene_over_15s")

    # delete short videos entirely (media, landmarks, sheet) and drop them from the source list
    short = {vid for vid, reason in excluded.items() if reason == "no_jumping_scene_over_15s"}
    for vid in short:
        for path in [DATA / "media" / f"{vid}.mp4", DATA / "landmarks" / "cc" / f"{vid}.npz", DATA / "screening" / f"{vid}.jpg"]:
            path.unlink(missing_ok=True)
    import sys

    sys.path.insert(0, str(Path(__file__).parent))
    from download_cc_videos import SOURCES, write_sources

    with SOURCES.open(encoding="utf-8-sig") as handle:
        all_rows = list(csv.DictReader(handle))
    rows = {row["id"]: row for row in all_rows if row["id"] not in short}
    if len(rows) != len(all_rows):  # rewrite only when something was dropped (the downloader also writes this file)
        write_sources(rows)

    # re-key the visual review onto the >15 s segments; drop excluded videos and short scenes
    new_review = {k: v for k, v in review.items() if k.startswith("_")}
    dropped = []
    for key, decision in review.items():
        if key.startswith("_"):
            continue
        if "_seg" not in key:  # video-level note
            if key not in excluded:
                new_review[key] = decision
            continue
        vid, number = key.rsplit("_seg", 1)
        old = old_segments.get(vid, [])
        if vid in excluded or int(number) >= len(old):
            dropped.append(key)
            continue
        start = old[int(number)]["start_sec"]
        match = [i for i, s in enumerate(new_segments.get(vid, [])) if abs(s["start_sec"] - start) < 0.05]
        if not match:
            dropped.append(key)
            continue
        new_review[f"{vid}_seg{match[0]}"] = decision
    (DATA / "screening" / "review.json").write_text(json.dumps(new_review, ensure_ascii=False, indent=1), encoding="utf-8")

    # remove annotation material and proposals that are no longer valid
    keep = {k for k, v in new_review.items() if not k.startswith("_") and v in {"basic", "alternating", "double"}}
    removed = []
    for folder in (DATA / "annotation" / "clips", DATA / "labels" / "proposals"):
        for path in folder.glob("*") if folder.exists() else []:
            if path.stem not in keep:
                path.unlink()
                removed.append(path.name)
    (DATA / "manifest" / "excluded_ids.txt").write_text(
        "".join(f"{vid} {reason}\n" for vid, reason in sorted(excluded.items())), encoding="utf-8")
    print(json.dumps(dict(excluded_multi_person=sum(r == "multi_person" for r in excluded.values()),
                          excluded_short=len(short), review_kept=len(keep), review_dropped=dropped,
                          annotation_files_removed=len(removed)), ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
