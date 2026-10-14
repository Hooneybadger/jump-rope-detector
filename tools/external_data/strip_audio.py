"""Remove the audio track from already downloaded media and annotation clips (audio is not used).

Video streams are copied bit-for-bit (-c:v copy), so cached landmarks stay valid. sources.csv gets
the new file hash and a note.
"""
from __future__ import annotations

import csv
import json
import subprocess
from pathlib import Path

from download_cc_videos import FIELDS, SOURCES, sha256
from imageio_ffmpeg import get_ffmpeg_exe as _ffmpeg

ROOT = Path(__file__).resolve().parents[2]


def has_audio(path: Path) -> bool:
    probe = subprocess.run([_ffmpeg(), "-hide_banner", "-i", str(path)], capture_output=True, text=True, errors="ignore")
    return "Audio:" in probe.stderr


def strip(path: Path) -> bool:
    if not has_audio(path):
        return False
    tmp = path.with_suffix(".noaudio.mp4")
    subprocess.run([_ffmpeg(), "-v", "error", "-y", "-i", str(path), "-map", "0:v:0", "-c:v", "copy", "-an",
                    "-movflags", "+faststart", str(tmp)], check=True)
    tmp.replace(path)
    return True


def main() -> None:
    changed = [p for p in sorted((ROOT / "external_data" / "media").glob("*.mp4")) if strip(p)]
    changed += [p for p in sorted((ROOT / "external_data" / "annotation" / "clips").glob("*.mp4")) if strip(p)]
    with SOURCES.open(encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    names = {p.stem for p in changed}
    for row in rows:
        if row["id"] in names:
            row["sha256"] = sha256(ROOT / row["file"])
            row["notes"] = "audio track removed (video stream unchanged)"
    with SOURCES.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"audio_removed": [p.name for p in changed]}, indent=1))


if __name__ == "__main__":
    main()
