"""Propose per-jump label candidates for screened segments, for human verification in Kinovea.

Counting candidates come from the feet only (no audio, no rope vision): foot touchdowns are local
maxima of the torso-normalised foot height (image y) at native fps, one per two-foot jump or one per
foot landing for alternate-foot steps. Tags only guide the person who verifies:
  ok       interval to the previous touchdown is within 0.6-1.6x the segment median
  check    irregular interval (pause, missed or doubled touchdown)
  trip?    a pause (>1.6x median) during which the floor band shows no rope pass (rope_motion.py);
           rope vision is used only for this catch hint, never for counting
  double? / single?   double-under mode: hand-turn rate / jump rate over the past 2 s >= 1.5 or not
                       (tools/eval/hand_turns.py; the same rule as the double-under engine)
  first    first touchdown of the clip
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
L_SH, R_SH, L_HIP, R_HIP, L_ANK, R_ANK, L_HEEL, R_HEEL, L_TOE, R_TOE = 11, 12, 23, 24, 27, 28, 29, 30, 31, 32


def _interp(values: np.ndarray) -> np.ndarray:
    values = values.copy()
    bad = np.isnan(values)
    if bad.all():
        return values
    idx = np.arange(len(values))
    values[bad] = np.interp(idx[bad], idx[~bad], values[~bad])
    return values


def _peaks(x: np.ndarray, min_distance: int, prominence: float) -> list[int]:
    peaks = []
    for i in range(1, len(x) - 1):
        if x[i] >= x[i - 1] and x[i] > x[i + 1]:
            lo, hi = max(0, i - min_distance), min(len(x), i + min_distance + 1)
            if x[i] == x[lo:hi].max() and x[i] - min(x[lo:i].min(), x[i + 1:hi].min()) >= prominence:
                if not peaks or i - peaks[-1] >= min_distance:
                    peaks.append(i)
    return peaks


def _period_frames(x: np.ndarray, fps: float, lo_sec: float = 0.2, hi_sec: float = 0.8) -> int:
    """Dominant period (frames) from the power spectrum within [lo_sec, hi_sec]."""
    x = x - np.mean(x)
    n = 1 << int(np.ceil(np.log2(max(len(x), 16) * 4)))
    power = np.abs(np.fft.rfft(x * np.hanning(len(x)), n)) ** 2
    freqs = np.fft.rfftfreq(n, 1.0 / fps)
    band = (freqs >= 1.0 / hi_sec) & (freqs <= 1.0 / lo_sec)
    if not band.any():
        return max(1, int(round(0.4 * fps)))
    peak = freqs[band][int(np.argmax(power[band]))]
    return max(1, int(round(fps / peak)))


def _bandpass(x: np.ndarray, fps: float, period: int) -> np.ndarray:
    """Keep 0.5x-1.8x the step frequency (FFT mask) so peaks follow the jump rhythm."""
    f0 = fps / period
    spectrum = np.fft.rfft(x - np.mean(x))
    freqs = np.fft.rfftfreq(len(x), 1.0 / fps)
    spectrum[(freqs < 0.5 * f0) | (freqs > 1.8 * f0)] = 0
    return np.fft.irfft(spectrum, len(x))


def pose_touchdowns(npz: Path, start: float, end: float, alternate: bool) -> tuple[list[float], float]:
    """Foot touchdowns (local maxima of torso-normalised foot height, image y) and the step period (s)."""
    times, step, _ = pose_touchdowns_with_feet(npz, start, end, alternate)
    return times, step


def pose_touchdowns_with_feet(npz: Path, start: float, end: float, alternate: bool) -> tuple[list[float], float, list[str]]:
    """pose_touchdowns plus the landing foot of each touchdown: "left"/"right" (the person's own side,
    MediaPipe convention) for alternate-foot steps, "both" for two-foot jumps."""
    data = np.load(npz)
    t, lm = data["t"], data["lm"]
    sel = (t >= start) & (t <= end)
    t, lm = t[sel], lm[sel]
    fps = float(data["fps"])
    torso = np.nanmedian(np.abs((lm[:, L_HIP, 1] + lm[:, R_HIP, 1]) / 2 - (lm[:, L_SH, 1] + lm[:, R_SH, 1]) / 2))
    left = _interp(np.nanmean(lm[:, [L_ANK, L_HEEL, L_TOE], 1], axis=1)) / torso
    right = _interp(np.nanmean(lm[:, [R_ANK, R_HEEL, R_TOE], 1], axis=1)) / torso
    if alternate:
        diff = left - right  # anti-phase: + when the left foot is the lower (landed) one
        stride = _period_frames(diff, fps, 0.3, 1.6)  # one stride = two steps
        smooth = _bandpass(diff, fps, stride)
        distance = max(1, int(0.6 * stride))
        prominence = 0.3 * float(np.std(smooth))
        sides = {i: "left" for i in _peaks(smooth, distance, prominence)}
        sides.update({i: "right" for i in _peaks(-smooth, distance, prominence)})
        order = sorted(sides)
        return [float(t[i]) for i in order], stride / fps / 2, [sides[i] for i in order]
    feet = (left + right) / 2
    period = _period_frames(feet, fps)
    smooth = _bandpass(feet, fps, period)
    peaks = _peaks(smooth, max(1, int(0.6 * period)), 0.3 * float(np.std(smooth)))
    return [float(t[i]) for i in peaks], period / fps, ["both"] * len(peaks)


def _ffmpeg() -> str:
    """FFMPEG env var, else imageio-ffmpeg's bundled binary (installed in .data-venv)."""
    if os.environ.get("FFMPEG"):
        return os.environ["FFMPEG"]
    bundled = sorted((ROOT / ".data-venv" / "Lib" / "site-packages" / "imageio_ffmpeg" / "binaries").glob("ffmpeg*"))
    if bundled:
        return str(bundled[0])
    import imageio_ffmpeg

    return imageio_ffmpeg.get_ffmpeg_exe()


def candidate_events(video: Path, npz: Path, start: float, end: float, mode: str) -> tuple[list[dict], float]:
    """Foot touchdowns with review tags (see module docstring)."""
    import sys

    from rope_motion import floor_motion, passes_between, rope_passes

    sys.path.insert(0, str(ROOT))
    from tools.eval.hand_turns import hand_matrix, ratio_at

    touchdowns, step, feet = pose_touchdowns_with_feet(npz, start, end, alternate=(mode == "alternating"))
    intervals = np.diff(touchdowns) if len(touchdowns) > 1 else np.asarray([])
    median = float(np.median(intervals)) if len(intervals) else step
    times, motion = floor_motion(video, npz, start, end)
    passes = [None] + passes_between(touchdowns, rope_passes(times, motion, 0.5 * step))
    data = np.load(npz)
    hands = hand_matrix(data["lm"]) if mode == "double" else None
    events = []
    for k, time_sec in enumerate(touchdowns):
        if k == 0:
            tag = "first"
        else:
            gap = touchdowns[k] - touchdowns[k - 1]
            if gap > 1.6 * median and passes[k] == 0:
                tag = "trip?"
            elif not 0.6 * median <= gap <= 1.6 * median:
                tag = "check"
            elif mode == "double":
                ratio, _ = ratio_at(hands, data["t"], float(data["fps"]), 1.0 / median, time_sec, 2.0)
                tag = "double?" if ratio >= 1.5 else "single?"
            else:
                tag = "ok"
        events.append(dict(time_sec=round(time_sec, 3), tag=tag, foot=feet[k]))
    return events, step


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--segments", default=str(ROOT / "external_data" / "screening" / "segments.json"))
    parser.add_argument("--review", default=str(ROOT / "external_data" / "screening" / "review.json"),
                        help='visual type check: {"<id>_seg<k>": "basic"|"alternating"|"double"|"reject:..."}')
    parser.add_argument("--landmarks", default=str(ROOT / "external_data" / "landmarks" / "cc"))
    parser.add_argument("--media", default=str(ROOT / "external_data" / "media"))
    parser.add_argument("--out", default=str(ROOT / "external_data" / "labels" / "proposals"))
    args = parser.parse_args()
    review = json.loads(Path(args.review).read_text(encoding="utf-8"))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for item in json.loads(Path(args.segments).read_text(encoding="utf-8")):
        for number, segment in enumerate(item["segments"]):
            key = f"{item['id']}_seg{number}"
            mode = review.get(key)
            if mode not in {"basic", "alternating", "double"}:
                continue
            events, step = candidate_events(Path(args.media) / f"{item['id']}.mp4", Path(args.landmarks) / f"{item['id']}.npz",
                                            segment["start_sec"], segment["end_sec"], mode)
            ok = sum(e["tag"] in ("ok", "double?") for e in events)
            proposal = dict(id=key, video_id=item["id"], mode=mode, segment=[segment["start_sec"], segment["end_sec"]],
                            tier="proposal_unverified", step_sec=round(step, 3), events=events)
            (out / f"{key}.json").write_text(json.dumps(proposal, indent=1), encoding="utf-8")
            print(f"{key} {mode}: step={step:.3f}s touchdowns={len(events)} regular={ok / max(1, len(events) - 1):.0%}",
                  flush=True)


if __name__ == "__main__":
    main()
