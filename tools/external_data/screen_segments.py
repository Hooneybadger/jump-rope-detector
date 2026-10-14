"""Find continuous rope-jumping segments in cached landmarks and write review montages.

A 3 s window (0.5 s hop) is "jumping" when the full body is visible, and the torso-normalised
hip height oscillates with a clear period between 1.2 and 5 Hz. Neighbouring windows are merged
into segments of at least --min-seconds. For each segment the left/right ankle phase correlation
is reported (in-phase => two-foot jumps, anti-phase => alternate-foot steps) as a hint for the
human/visual type check; it is not a label.

    python tools/external_data/screen_segments.py --landmarks external_data/landmarks/cc --media external_data/media
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
NOSE, L_SH, R_SH, L_HIP, R_HIP, L_ANK, R_ANK = 0, 11, 12, 23, 24, 27, 28
WINDOW_SEC, HOP_SEC = 3.0, 0.5
F_MIN, F_MAX = 1.0, 6.0  # search band; accepted cadence is 1.3-4.6 Hz (edge picks = no clear period)
ACCEPT_MIN, ACCEPT_MAX = 1.3, 4.6


def _interp_nan(values: np.ndarray) -> np.ndarray:
    values = values.copy()
    bad = np.isnan(values)
    if bad.all():
        return values
    idx = np.arange(len(values))
    values[bad] = np.interp(idx[bad], idx[~bad], values[~bad])
    return values


def _period(signal: np.ndarray, fps: float) -> tuple[float, float]:
    """Dominant frequency (Hz) and normalised autocorrelation at that lag."""
    x = signal - signal.mean()
    denom = float(np.dot(x, x))
    if denom <= 1e-12:
        return 0.0, 0.0
    lags = range(max(1, int(fps / F_MAX)), min(len(x) - 2, int(fps / F_MIN)) + 1)
    best_lag, best_r = 0, -1.0
    for lag in lags:
        r = float(np.dot(x[:-lag], x[lag:])) / denom * len(x) / (len(x) - lag)
        if r > best_r:
            best_lag, best_r = lag, r
    return (fps / best_lag if best_lag else 0.0), best_r


def screen(npz_path: Path, min_seconds: float) -> dict:
    data = np.load(npz_path)
    t, lm, fps = data["t"], data["lm"], float(data["fps"])
    vis = lm[:, :, 3]
    full_body = np.all(np.nan_to_num(vis[:, [NOSE, L_HIP, R_HIP, L_ANK, R_ANK]]) >= 0.5, axis=1)
    in_frame = np.nan_to_num(lm[:, [L_ANK, R_ANK], 1], nan=2.0).max(axis=1) <= 1.0
    valid = full_body & in_frame
    torso = np.abs((lm[:, L_HIP, 1] + lm[:, R_HIP, 1]) / 2 - (lm[:, L_SH, 1] + lm[:, R_SH, 1]) / 2)
    hip = _interp_nan(((lm[:, L_HIP, 1] + lm[:, R_HIP, 1]) / 2) / np.nanmedian(torso))
    la, ra = _interp_nan(lm[:, L_ANK, 1]), _interp_nan(lm[:, R_ANK, 1])

    win, hop = int(round(WINDOW_SEC * fps)), max(1, int(round(HOP_SEC * fps)))
    flags = np.zeros(len(t), dtype=bool)
    for start in range(0, max(0, len(t) - win), hop):
        sl = slice(start, start + win)
        if valid[sl].mean() < 0.9:
            continue
        freq, r = _period(hip[sl], fps)
        x = np.arange(win)
        amplitude = float(np.std(hip[sl] - np.polyval(np.polyfit(x, hip[sl], 1), x)))
        if r >= 0.5 and ACCEPT_MIN <= freq <= ACCEPT_MAX and amplitude >= 0.02:
            flags[sl] = True

    segments, i = [], 0
    while i < len(flags):
        if not flags[i]:
            i += 1
            continue
        j = i
        while j < len(flags) and flags[j]:
            j += 1
        if (j - i) / fps > min_seconds:
            sl = slice(i, j)
            freq, r = _period(hip[sl][: int(10 * fps)], fps)
            phase_corr = float(np.corrcoef(np.diff(la[sl]), np.diff(ra[sl]))[0, 1])
            body_height = float(np.nanmedian(np.nan_to_num(lm[sl, [L_ANK, R_ANK], 1]).max(axis=1) - lm[sl, NOSE, 1]))
            segments.append(dict(start_sec=round(float(t[i]), 3), end_sec=round(float(t[j - 1]), 3),
                                 duration_sec=round((j - i) / fps, 2), hip_freq_hz=round(freq, 2), periodicity=round(r, 2),
                                 ankle_phase_corr=round(phase_corr, 2), body_height_frac=round(body_height, 2),
                                 hint="two_foot" if phase_corr > 0.3 else ("alternate_foot" if phase_corr < -0.3 else "unclear")))
        i = j
    return dict(id=npz_path.stem, fps=fps, frames=len(t), width=int(data["width"]), height=int(data["height"]),
                valid_fraction=round(float(valid.mean()), 3), segments=segments)


def contact_sheet(video: Path, item: dict, out: Path, tiles: int = 5, row_height: int = 150) -> None:
    """One image per video: one row per segment (label 'seg k  start-end s'), tiles spread over it."""
    if not item["segments"]:
        return
    capture = cv2.VideoCapture(str(video))
    rows = []
    for number, segment in enumerate(item["segments"]):
        images = []
        for k in range(tiles):
            at = segment["start_sec"] + (segment["end_sec"] - segment["start_sec"]) * (k + 0.5) / tiles
            capture.set(cv2.CAP_PROP_POS_MSEC, at * 1000)
            ok, frame = capture.read()
            if not ok:
                frame = np.zeros((row_height, row_height, 3), np.uint8)
            scale = row_height / frame.shape[0]
            images.append(cv2.resize(frame, (max(1, int(frame.shape[1] * scale)), row_height)))
        row = np.hstack(images)
        label = f"seg{number} {segment['start_sec']:.0f}-{segment['end_sec']:.0f}s {segment['hint']}"
        cv2.rectangle(row, (0, 0), (len(label) * 11 + 8, 22), (0, 0, 0), -1)
        cv2.putText(row, label, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        rows.append(row)
    capture.release()
    width = max(r.shape[1] for r in rows)
    rows = [cv2.copyMakeBorder(r, 0, 2, 0, width - r.shape[1], cv2.BORDER_CONSTANT) for r in rows]
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 60])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--landmarks", default=str(ROOT / "external_data" / "landmarks" / "cc"))
    parser.add_argument("--media", default=str(ROOT / "external_data" / "media"))
    parser.add_argument("--out", default=str(ROOT / "external_data" / "screening"))
    parser.add_argument("--min-seconds", type=float, default=15.0, help="jumping scenes must be longer than this")
    parser.add_argument("--no-montage", action="store_true")
    args = parser.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    report = []
    for npz in sorted(Path(args.landmarks).glob("*.npz")):
        item = screen(npz, args.min_seconds)
        report.append(item)
        if not args.no_montage:
            sheet = out_dir / f"{item['id']}.jpg"
            if not sheet.exists():
                contact_sheet(Path(args.media) / f"{item['id']}.mp4", item, sheet)
        print(item["id"], f"valid={item['valid_fraction']:.0%}", [(s["start_sec"], s["duration_sec"], s["hip_freq_hz"], s["hint"]) for s in item["segments"]], flush=True)
    (out_dir / "segments.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
