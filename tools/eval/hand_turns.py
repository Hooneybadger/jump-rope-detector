"""Causal per-jump hand turns: how many hand circles happened between the previous and this landing.

At each labelled jump, take only the past --window seconds of landmarks (as a live counter would),
project the eight wrist-relative hand-point coordinates onto their first principal component,
band-pass it around 0.6x-2.6x the recent jump rate, and measure the unwrapped phase advance of its
analytic signal between the previous and the current landing. One circle per jump => ~1 turn
(single), two => ~2 turns (double-under). The ratio of hand-band power peak sharpness is reported
as a quality score so small/blurred hands can fall back to another classifier.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.eval.evaluate_engines import bench_records  # noqa: E402
from tools.eval.jump_features import _interp  # noqa: E402

HAND = ((15, 17), (15, 19), (16, 18), (16, 20))  # (wrist, pinky/index)


def hand_matrix(lm: np.ndarray) -> np.ndarray:
    hip_to_foot = np.nanmedian(np.nanmean(lm[:, 27:33, 1], axis=1) - (lm[:, 23, 1] + lm[:, 24, 1]) / 2)
    cols = [_interp(lm[:, p, a] - lm[:, w, a]) / hip_to_foot for w, p in HAND for a in (0, 1)]
    return np.stack(cols, axis=1)


def analytic_phase(x: np.ndarray, fps: float, lo: float, hi: float) -> np.ndarray:
    n = len(x)
    spectrum = np.fft.fft(x - x.mean())
    freqs = np.fft.fftfreq(n, 1.0 / fps)
    keep = (freqs >= lo) & (freqs <= hi)  # positive band only => analytic signal
    spectrum[~keep] = 0
    return np.unwrap(np.angle(np.fft.ifft(2 * spectrum)))


def turns_at(hands: np.ndarray, t: np.ndarray, fps: float, prev_landing: float, landing: float,
             jump_hz: float, window: float, now: float | None = None) -> tuple[float, float]:
    """Turns between prev_landing and landing, using only landmarks up to `now` (default: landing)."""
    now = landing if now is None else now
    sel = (t > now - window) & (t <= now)
    if sel.sum() < 10:
        return float("nan"), 0.0
    block = hands[sel] - hands[sel].mean(axis=0)
    _, singular, vt = np.linalg.svd(block, full_matrices=False)
    pc = block @ vt[0]
    phase = analytic_phase(pc, fps, 0.6 * jump_hz, min(2.6 * jump_hz, fps / 2 - 0.5))
    tt = t[sel]
    i0, i1 = int(np.argmin(np.abs(tt - prev_landing))), int(np.argmin(np.abs(tt - landing)))
    turns = abs(phase[i1] - phase[i0]) / (2 * np.pi)
    quality = float(singular[0] ** 2 / max(1e-12, np.sum(singular ** 2)))
    return float(turns), quality


def ratio_at(hands: np.ndarray, t: np.ndarray, fps: float, jump_hz: float, now: float,
             window: float) -> tuple[float, float]:
    """(dominant hand frequency / jump frequency, spectral concentration) over the window ending at `now`."""
    sel = (t > now - window) & (t <= now)
    if sel.sum() < 10:
        return float("nan"), 0.0
    block = hands[sel] - hands[sel].mean(axis=0)
    taper = np.hanning(len(block))[:, None]
    n = 1 << int(np.ceil(np.log2(len(block) * 8)))
    power = np.sum(np.abs(np.fft.rfft(block * taper, n, axis=0)) ** 2, axis=1)
    freqs = np.fft.rfftfreq(n, 1.0 / fps)
    band = (freqs >= 0.6 * jump_hz) & (freqs <= min(2.6 * jump_hz, fps / 2 - 0.5))
    peak = int(np.argmax(power[band]))
    f_peak = freqs[band][peak]
    near = band & (np.abs(freqs - f_peak) <= 0.15 * f_peak)
    concentration = float(power[near].sum() / max(1e-12, power[band].sum()))
    return float(f_peak / jump_hz), concentration


def main() -> None:
    np.seterr(all="ignore")
    window = 2.0
    lag = "--lag" in sys.argv
    for record in bench_records(ROOT / "external_data" / "landmarks" / "bench"):
        data = np.load(record["npz"])
        t, lm, fps = data["t"], data["lm"], float(data["fps"])
        hands = hand_matrix(lm)
        labels = record["labels_sec"]
        values = []
        for k in range(3, len(labels)):
            jump_hz = 1.0 / max(0.15, float(np.median(np.diff(labels[k - 3:k + 1]))))
            if "--ratio" in sys.argv:
                ratio, concentration = ratio_at(hands, t, fps, jump_hz, labels[k], window)
                turns = concentration if "--quality" in sys.argv else ratio
            elif lag:
                # decide jump k-1 once landing k has arrived: its cycle sits one jump away from the window edge
                turns, _ = turns_at(hands, t, fps, labels[k - 2], labels[k - 1], jump_hz, window, labels[k])
            else:
                turns, _ = turns_at(hands, t, fps, labels[k - 1], labels[k], jump_hz, window)
            values.append(turns)
        v = np.asarray(values)
        v = v[np.isfinite(v)]
        if len(v):
            print(f"{record['id']:16s} n={len(v):3d} turns/jump p10={np.percentile(v, 10):.2f} p50={np.median(v):.2f} "
                  f"p90={np.percentile(v, 90):.2f}  share>=1.5: {np.mean(v >= 1.5):.0%}")


if __name__ == "__main__":
    main()
