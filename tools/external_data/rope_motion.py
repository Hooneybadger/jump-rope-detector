"""Visual rope-pass cue from pixels near the floor (independent of the pose-based touchdown cue).

The rope sweeps under the feet once per turn. For each frame, take the absolute frame difference
inside a floor band around the feet, mask the feet and shins (their own motion follows the jump),
and keep the mean of the remaining moving pixels. Peaks of this "floor motion" series mark rope
passes; between two foot touchdowns a basic jump shows one pass and a double-under two.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

L_HIP, R_HIP, L_KNEE, R_KNEE = 23, 24, 25, 26
FEET = (27, 28, 29, 30, 31, 32)  # ankles, heels, toes
WORK_WIDTH = 640


def floor_motion(video: Path, npz: Path, start: float, end: float) -> tuple[np.ndarray, np.ndarray]:
    """Return (times, motion) for frames in [start, end]."""
    data = np.load(npz)
    t_all, lm_all = data["t"], data["lm"]
    capture = cv2.VideoCapture(str(video))
    fps = capture.get(cv2.CAP_PROP_FPS) or float(data["fps"])
    capture.set(cv2.CAP_PROP_POS_MSEC, max(0.0, start - 0.1) * 1000)
    prev, times, motion = None, [], []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        now = capture.get(cv2.CAP_PROP_POS_MSEC) / 1000.0 - 1.0 / fps
        if now > end:
            break
        scale = WORK_WIDTH / frame.shape[1]
        gray = cv2.cvtColor(cv2.resize(frame, (WORK_WIDTH, int(frame.shape[0] * scale))), cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        if prev is None or now < start:
            prev = gray
            continue
        index = int(np.argmin(np.abs(t_all - now)))
        row = lm_all[index]
        if np.isnan(row[0, 0]):
            prev = gray
            continue
        h, w = gray.shape
        xy = row[:, :2] * [w, h]
        leg = max(8.0, float(np.mean([abs(row[L_HIP, 1] - row[27, 1]), abs(row[R_HIP, 1] - row[28, 1])]) * h))
        floor = float(np.max(xy[list(FEET), 1]))
        center = float((xy[L_HIP, 0] + xy[R_HIP, 0]) / 2)
        y0, y1 = int(max(0, floor - 0.30 * leg)), int(min(h, floor + 0.10 * leg))
        x0, x1 = int(max(0, center - 0.9 * leg)), int(min(w, center + 0.9 * leg))
        diff = cv2.absdiff(gray, prev)
        prev = gray
        if y1 - y0 < 3 or x1 - x0 < 3:
            continue
        mask = np.zeros_like(gray)
        mask[y0:y1, x0:x1] = 255
        radius = max(2, int(0.10 * leg))
        for joint in FEET:
            cv2.circle(mask, (int(xy[joint, 0]), int(xy[joint, 1])), radius, 0, -1)
        for knee, ankle in ((L_KNEE, 27), (R_KNEE, 28)):
            cv2.line(mask, tuple(map(int, xy[knee])), tuple(map(int, xy[ankle])), 0, max(2, int(0.12 * leg)))
        moving = (diff > 18) & (mask > 0)
        area = max(1, int(np.count_nonzero(mask)))
        times.append(now)
        motion.append(np.count_nonzero(moving) / area)
    capture.release()
    return np.asarray(times), np.asarray(motion)


def rope_passes(times: np.ndarray, motion: np.ndarray, min_gap: float) -> list[float]:
    """Peaks of the floor-motion series at least min_gap seconds apart and above its median + 1 MAD."""
    if len(motion) < 5:
        return []
    fps = 1.0 / max(1e-6, float(np.median(np.diff(times))))
    distance = max(1, int(min_gap * fps))
    level = np.median(motion) + np.median(np.abs(motion - np.median(motion)))
    peaks = []
    for i in range(1, len(motion) - 1):
        if motion[i] < level or motion[i] < motion[i - 1] or motion[i] <= motion[i + 1]:
            continue
        lo, hi = max(0, i - distance), min(len(motion), i + distance + 1)
        if motion[i] == motion[lo:hi].max() and (not peaks or times[i] - peaks[-1] >= min_gap):
            peaks.append(float(times[i]))
    return peaks


def passes_between(touchdowns: list[float], passes: list[float]) -> list[int]:
    """Number of rope passes in each interval between consecutive touchdowns."""
    p = np.asarray(passes)
    return [int(np.sum((p > a) & (p <= b))) for a, b in zip(touchdowns, touchdowns[1:])]
