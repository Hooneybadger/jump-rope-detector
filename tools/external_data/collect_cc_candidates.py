"""Collect YouTube jump-rope videos whose license is Creative Commons Attribution (CC BY).

Only metadata is fetched here. Every candidate is re-checked against the video's own
license field, so the search filter alone is never trusted. Results are appended to
external_data/manifest/candidates.jsonl with the provenance needed for attribution.

Run with the isolated tooling venv:
    .data-venv/Scripts/python.exe tools/external_data/collect_cc_candidates.py
"""
from __future__ import annotations

import json
import re
import time
import urllib.parse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import yt_dlp

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "external_data" / "manifest" / "candidates.jsonl"
SEARCH_CACHE = ROOT / "external_data" / "manifest" / "search_hits.json"
CC_FILTER = "EgIwAQ%3D%3D"  # YouTube search filter: Creative Commons
CC_BY_LICENSE = "Creative Commons Attribution license (reuse allowed)"
PER_QUERY = 40
MIN_DURATION_SEC = 8
MAX_DURATION_SEC = 1800
NC_PATTERN = re.compile(r"non[- ]?commercial|\bCC[- ]?BY[- ]?NC\b|비영리|비상업", re.IGNORECASE)

# (query, mode hint). The hint only orders later screening; it is not a label.
QUERIES = [
    ("jump rope basic bounce", "basic"),
    ("jump rope tutorial beginner", "basic"),
    ("jump rope workout", "basic"),
    ("skipping rope workout", "basic"),
    ("줄넘기", "basic"),
    ("줄넘기 모아뛰기", "basic"),
    ("줄넘기 양발모아뛰기", "basic"),
    ("なわとび", "basic"),
    ("跳绳", "basic"),
    ("saltar la comba", "basic"),
    ("corde à sauter", "basic"),
    ("Seilspringen", "basic"),
    ("jump rope alternate foot step", "alternating"),
    ("jump rope running step", "alternating"),
    ("speed step jump rope", "alternating"),
    ("줄넘기 번갈아뛰기", "alternating"),
    ("줄넘기 구보", "alternating"),
    ("かけあし跳び なわとび", "alternating"),
    ("double unders", "double"),
    ("double under jump rope", "double"),
    ("double unders tutorial", "double"),
    ("줄넘기 이중뛰기", "double"),
    ("줄넘기 2단뛰기", "double"),
    ("二重跳び", "double"),
    ("跳绳 双摇", "double"),
    ("jump rope fail", "trip"),
    ("double under fails", "trip"),
    ("jump rope mistakes", "trip"),
    ("줄넘기 실패", "trip"),
]


def search_ids(query: str) -> list[str]:
    url = (
        "https://www.youtube.com/results?search_query="
        + urllib.parse.quote_plus(query)
        + "&sp="
        + CC_FILTER
    )
    opts = {"quiet": True, "extract_flat": True, "playlistend": PER_QUERY, "skip_download": True}
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=False)
    return [entry["id"] for entry in info.get("entries") or [] if entry.get("id")]


def fetch_metadata(video_id: str) -> dict | None:
    opts = {"quiet": True, "skip_download": True, "noplaylist": True, "js_runtimes": {"node": {}}}
    with yt_dlp.YoutubeDL(opts) as ydl:
        try:
            info = ydl.extract_info(f"https://www.youtube.com/watch?v={video_id}", download=False)
        except yt_dlp.utils.DownloadError as error:
            return {"id": video_id, "error": str(error)[:300]}
    return info


def load_seen() -> dict[str, dict]:
    seen: dict[str, dict] = {}
    if MANIFEST.exists():
        for line in MANIFEST.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                seen[row["id"]] = row
    return seen


def search_all() -> dict[str, list[tuple[str, str]]]:
    """Search once and cache the hits so an interrupted run resumes without re-searching."""
    if SEARCH_CACHE.exists():
        return {k: [tuple(h) for h in v] for k, v in json.loads(SEARCH_CACHE.read_text(encoding="utf-8")).items()}
    query_hits: dict[str, list[tuple[str, str]]] = {}
    for query, hint in QUERIES:
        try:
            ids = search_ids(query)
        except Exception as error:  # network hiccup: keep the other queries
            print(f"search failed: {query!r}: {error}", flush=True)
            continue
        print(f"{query!r}: {len(ids)} results", flush=True)
        for video_id in ids:
            query_hits.setdefault(video_id, []).append((query, hint))
        time.sleep(1.0)
    SEARCH_CACHE.write_text(json.dumps(query_hits, ensure_ascii=False, indent=1), encoding="utf-8")
    return query_hits


def main() -> None:
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    seen = load_seen()
    query_hits = search_all()
    pending = [video_id for video_id in query_hits if video_id not in seen]
    print(f"{len(query_hits)} unique ids, {len(pending)} need metadata", flush=True)

    with MANIFEST.open("a", encoding="utf-8") as out, ThreadPoolExecutor(max_workers=6) as pool:
        futures = {pool.submit(fetch_metadata, video_id): video_id for video_id in pending}
        for future in as_completed(futures):
            video_id = futures[future]
            hits = query_hits[video_id]
            info = future.result()
            if info is None or "error" in info:
                row = {"id": video_id, "status": "error", "error": (info or {}).get("error")}
            else:
                license_text = info.get("license") or ""
                description = info.get("description") or ""
                duration = info.get("duration") or 0
                reasons = []
                if license_text != CC_BY_LICENSE:
                    reasons.append("license_not_cc_by")
                if NC_PATTERN.search(description) or NC_PATTERN.search(info.get("title") or ""):
                    reasons.append("noncommercial_wording")
                if info.get("live_status") not in (None, "not_live", "was_live"):
                    reasons.append("live")
                if not MIN_DURATION_SEC <= duration <= MAX_DURATION_SEC:
                    reasons.append("duration")
                row = {
                    "id": video_id,
                    "status": "candidate" if not reasons else "excluded",
                    "exclude_reasons": reasons,
                    "url": f"https://www.youtube.com/watch?v={video_id}",
                    "title": info.get("title"),
                    "channel": info.get("channel") or info.get("uploader"),
                    "channel_url": info.get("channel_url") or info.get("uploader_url"),
                    "upload_date": info.get("upload_date"),
                    "license": license_text,
                    "duration_sec": duration,
                    "width": info.get("width"),
                    "height": info.get("height"),
                    "fps": info.get("fps"),
                    "queries": [query for query, _ in hits],
                    "mode_hints": sorted({hint for _, hint in hits}),
                    "metadata_retrieved_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                }
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            out.flush()
            print(video_id, row.get("status"), row.get("exclude_reasons", ""), flush=True)


if __name__ == "__main__":
    main()
