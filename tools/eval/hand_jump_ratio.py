"""Hand-turn rate relative to jump rate, per labelled clip window.

Each rope turn is driven by one circle of the hands. In a single (basic) jump the hands make one
circle per jump; in a double-under two. MediaPipe's forearm barely rotates, but the hand points
(wrist -> index / pinky) trace the wrist circle, so their x/y oscillate once per turn.

For sliding 3 s windows inside the labelled span: jump frequency from the foot-height spectrum,
hand frequency from the spectrum of the wrist-relative hand point positions (both hands summed),
and their ratio (about 1 for singles, about 2 for double-unders).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.eval.evaluate_engines import bench_records  # noqa: E402
from tools.eval.jump_features import _interp  # noqa: E402

L_WR, R_WR, L_PINKY, R_PINKY, L_INDEX, R_INDEX = 15, 16, 17, 18, 19, 20
L_FOOT, R_FOOT = (27, 29, 31), (28, 30, 32)
L_HIP, R_HIP = 23, 24


def _spectrum(x: np.ndarray, fps: float) -> tuple[np.ndarray, np.ndarray]:
    x = x - np.mean(x)
    n = 1 << int(np.ceil(np.log2(max(len(x), 16) * 8)))
    return np.fft.rfftfreq(n, 1.0 / fps), np.abs(np.fft.rfft(x * np.hanning(len(x)), n)) ** 2


def window_ratios(npz: Path, start: float, end: float, window: float = 3.0, hop: float = 1.0) -> list[tuple[float, float, float]]:
    data = np.load(npz)
    t, lm, fps = data["t"], data["lm"], float(data["fps"])
    feet = _interp(np.nanmean(lm[:, L_FOOT + R_FOOT, 1], axis=1))
    hip = _interp((lm[:, L_HIP, 1] + lm[:, R_HIP, 1]) / 2)
    scale = np.nanmedian(feet - hip)
    hands = []
    for wrist, pinky, index in ((L_WR, L_PINKY, L_INDEX), (R_WR, R_PINKY, R_INDEX)):
        for point in (pinky, index):
            for axis in (0, 1):
                hands.append(_interp(lm[:, point, axis] - lm[:, wrist, axis]) / scale)
    out = []
    a = start
    while a + window <= end + 1e-9:
        sel = (t >= a) & (t < a + window)
        if sel.sum() < 0.8 * window * fps:
            a += hop
            continue
        freqs, foot_power = _spectrum(feet[sel], fps)
        band = (freqs >= 1.2) & (freqs <= 5.0)
        f_jump = freqs[band][int(np.argmax(foot_power[band]))]
        hand_power = sum(_spectrum(h[sel], fps)[1] for h in hands)
        hand_band = (freqs >= 0.6 * f_jump) & (freqs <= min(2.6 * f_jump, fps / 2 - 0.5))
        f_hand = freqs[hand_band][int(np.argmax(hand_power[hand_band]))]
        out.append((a, float(f_jump), float(f_hand / f_jump)))
        a += hop
    return out


def main() -> None:
    np.seterr(all="ignore")
    summary = {}
    for record in bench_records(ROOT / "external_data" / "landmarks" / "bench"):
        labels = record["labels_sec"]
        ratios = window_ratios(Path(record["npz"]), labels[0], labels[-1])
        if not ratios:
            continue
        values = np.array([r for _, _, r in ratios])
        jump = np.median([f for _, f, _ in ratios])
        summary.setdefault(record["mode"], []).extend(values.tolist())
        print(f"{record['id']:16s} windows={len(values):3d} jump={jump:.2f}Hz hand/jump ratio p10={np.percentile(values, 10):.2f} "
              f"p50={np.median(values):.2f} p90={np.percentile(values, 90):.2f}  share>=1.5: {np.mean(values >= 1.5):.0%}")
    for mode, values in summary.items():
        values = np.asarray(values)
        print(f"{mode:11s} windows={len(values):4d} median ratio={np.median(values):.2f} share>=1.5: {np.mean(values >= 1.5):.0%}")


if __name__ == "__main__":
    main()
