"""Download screened CC BY candidates and write the source list (attribution file).

Each video's license is re-read from YouTube immediately before download; anything that is no
longer "Creative Commons Attribution" is skipped. Files go to external_data/media/<id>.mp4
(360p video only, no audio track), and every download is recorded in
external_data/manifest/sources.csv and external_data/ATTRIBUTION.md.

    .data-venv/Scripts/python.exe tools/external_data/download_cc_videos.py --ids-file external_data/manifest/selected_ids.txt
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import time
from datetime import datetime, timezone
from pathlib import Path

import imageio_ffmpeg
import yt_dlp

ROOT = Path(__file__).resolve().parents[2]
MEDIA = ROOT / "external_data" / "media"
SOURCES = ROOT / "external_data" / "manifest" / "sources.csv"
FAILED = ROOT / "external_data" / "manifest" / "failed_ids.txt"
EXCLUDED = ROOT / "external_data" / "manifest" / "excluded_ids.txt"
MIN_DURATION_SEC = 15  # a video this short cannot hold a jumping scene longer than 15 s
ATTRIBUTION = ROOT / "external_data" / "ATTRIBUTION.md"
CC_BY_LICENSE = "Creative Commons Attribution license (reuse allowed)"
FIELDS = ["id", "url", "title", "channel", "channel_url", "upload_date", "license", "license_checked_at",
          "duration_sec", "width", "height", "fps", "file", "sha256", "downloaded_at", "mode_hints", "notes"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_sources() -> dict[str, dict]:
    if not SOURCES.exists():
        return {}
    with SOURCES.open(encoding="utf-8-sig") as handle:
        return {row["id"]: row for row in csv.DictReader(handle)}


def write_sources(rows: dict[str, dict]) -> None:
    with SOURCES.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        for row in sorted(rows.values(), key=lambda r: r["id"]):
            writer.writerow({k: row.get(k, "") for k in FIELDS})
    lines = [
        "# 외부 영상 출처 (Creative Commons Attribution, CC BY)",
        "",
        "아래 영상은 YouTube에서 업로더가 'Creative Commons Attribution license (reuse allowed)'로 공개한 영상이며,",
        "줄넘기 카운팅 엔진의 연구·검증 목적으로 사용합니다. 라이선스는 다운로드 직전에 영상 메타데이터에서 다시 확인했습니다.",
        "영상 파일과 파생 관절 데이터는 저장소에 포함하지 않습니다(.gitignore).",
        "",
        "| # | 제목 | 업로더 | URL | 업로드일 | 라이선스 확인 시각(UTC) |",
        "|---:|---|---|---|---|---|",
    ]
    for number, row in enumerate(sorted(rows.values(), key=lambda r: r["id"]), start=1):
        title = (row.get("title") or "").replace("|", "/")
        channel = (row.get("channel") or "").replace("|", "/")
        lines.append(f"| {number} | {title} | [{channel}]({row.get('channel_url')}) | {row['url']} | "
                     f"{row.get('upload_date')} | {row.get('license_checked_at')} |")
    ATTRIBUTION.write_text("\n".join(lines) + "\n", encoding="utf-8")


def download(video_id: str, hints: str) -> dict | None:
    url = f"https://www.youtube.com/watch?v={video_id}"
    base = {"quiet": True, "noplaylist": True, "js_runtimes": {"node": {}}}
    with yt_dlp.YoutubeDL({**base, "skip_download": True}) as ydl:
        info = ydl.extract_info(url, download=False)
    checked_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    if (info.get("license") or "") != CC_BY_LICENSE:
        print(f"skip {video_id}: license is now {info.get('license')!r}", flush=True)
        return None
    MEDIA.mkdir(parents=True, exist_ok=True)
    options = {
        **base,
        # Video only (no audio is collected or used): 360p/480p H.264 over plain HTTPS. HLS (m3u8)
        # streams stalled on timeouts.
        "format": "134/135/bv*[height<=480][ext=mp4][protocol=https][vcodec^=avc1]",
        "noprogress": True,
        "socket_timeout": 30,
        "retries": 3,
        "fragment_retries": 3,
        "merge_output_format": "mp4",
        "outtmpl": str(MEDIA / "%(id)s.%(ext)s"),
        "ffmpeg_location": imageio_ffmpeg.get_ffmpeg_exe(),
    }
    with yt_dlp.YoutubeDL(options) as ydl:
        result = ydl.extract_info(url, download=True)
    path = MEDIA / f"{video_id}.mp4"
    if not path.exists():
        print(f"missing file after download: {video_id}", flush=True)
        return None
    return dict(
        id=video_id, url=url, title=info.get("title"), channel=info.get("channel") or info.get("uploader"),
        channel_url=info.get("channel_url") or info.get("uploader_url"), upload_date=info.get("upload_date"),
        license=info.get("license"), license_checked_at=checked_at, duration_sec=info.get("duration"),
        width=result.get("width"), height=result.get("height"), fps=result.get("fps"),
        file=str(path.relative_to(ROOT)).replace("\\", "/"), sha256=sha256(path),
        downloaded_at=datetime.now(timezone.utc).isoformat(timespec="seconds"), mode_hints=hints, notes="",
    )


def discard_failed(video_id: str) -> None:
    """Delete partial files of a failed download and never retry it."""
    for path in MEDIA.glob(f"{video_id}.*"):
        path.unlink(missing_ok=True)
    with FAILED.open("a", encoding="utf-8") as handle:
        handle.write(video_id + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ids-file", required=True)
    parser.add_argument("--limit", type=int, default=0, help="max downloads in this batch (0 = all)")
    args = parser.parse_args()
    candidates = {}
    for line in (ROOT / "external_data" / "manifest" / "candidates.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            candidates[row["id"]] = row
    rows = load_sources()
    failed = set(FAILED.read_text(encoding="utf-8").split()) if FAILED.exists() else set()
    if EXCLUDED.exists():  # "<id> <reason>" per line: multi-person or no jumping scene over 15 s
        failed |= {line.split()[0] for line in EXCLUDED.read_text(encoding="utf-8").splitlines() if line.strip()}
    done = 0
    for video_id in Path(args.ids_file).read_text(encoding="utf-8").split():
        if video_id in rows or video_id in failed:
            continue
        if args.limit and done >= args.limit:
            break
        candidate = candidates.get(video_id, {})
        if candidate.get("status") != "candidate":
            print(f"skip {video_id}: not a screened CC BY candidate", flush=True)
            continue
        if (candidate.get("duration_sec") or 0) <= MIN_DURATION_SEC:
            print(f"skip {video_id}: {candidate.get('duration_sec')} s video, no room for a >15 s jumping scene", flush=True)
            continue
        if done:
            time.sleep(random.uniform(20, 40))  # stay well under YouTube's rate limits
        done += 1
        try:
            row = download(video_id, ",".join(candidate.get("mode_hints", [])))
        except Exception as error:
            message = str(error)
            print(f"failed {video_id}: {message[:200]}", flush=True)
            discard_failed(video_id)
            if "not a bot" in message or "429" in message:
                print("rate limited by YouTube; stopping this batch", flush=True)
                break
            continue
        if not row:
            discard_failed(video_id)
        else:
            rows[video_id] = row
            write_sources(rows)
            print(f"ok {video_id} {row['width']}x{row['height']}@{row['fps']}", flush=True)


if __name__ == "__main__":
    main()
