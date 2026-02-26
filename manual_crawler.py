import argparse
import os
import subprocess


def parse_line(line):
    # Format: url|start|end|output
    parts = [p.strip() for p in line.split("|")]
    if len(parts) != 4:
        raise ValueError("Expected 4 fields: url|start|end|output")
    return parts[0], parts[1], parts[2], parts[3]


def ensure_parent(path):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def normalize_mp4_path(out_path):
    base, _ext = os.path.splitext(out_path)
    return base + ".mp4"


def download_section(url, start, end, out_path):
    ensure_parent(out_path)
    section = f"*{start}-{end}"
    out_path = normalize_mp4_path(out_path)
    cmd = [
        "yt-dlp",
        "--download-sections", section,
        "--remux-video", "mp4",
        "--merge-output-format", "mp4",
        url,
        "-o", out_path
    ]
    subprocess.run(cmd, check=False)


def main():
    ap = argparse.ArgumentParser(description="Manual multi-clip downloader with yt-dlp.")
    ap.add_argument("--input", "-i", required=True, help="Path to txt file")
    args = ap.parse_args()

    if not os.path.exists(args.input):
        raise SystemExit(f"Input file not found: {args.input}")

    with open(args.input, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            url, start, end, out_path = parse_line(line)
            print(f"[DOWNLOAD] {url} {start}-{end} -> {out_path}")
            download_section(url, start, end, out_path)


if __name__ == "__main__":
    main()
