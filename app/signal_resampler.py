from __future__ import annotations

from dataclasses import fields, replace
from typing import Any

ENGINE_RATE_HZ = 30.0
MAX_INTERPOLATION_GAP_SEC = 0.25


def _is_rate_field(name: str) -> bool:
    # Frame-to-frame motion measured between two real frames; it is split across the
    # synthetic frames instead of being interpolated.
    return name.endswith("_flow_ratio") or name.endswith("_rotation_ratio")


def _is_interval_field(name: str) -> bool:
    # Describes the whole interval between two real frames, so every synthetic frame
    # inside that interval carries the same value.
    return name.endswith("_sync_ratio")


class SignalResampler:
    """Feeds counting engines at the frame rate their thresholds were tuned for.

    The engines count airborne frames, refractory gaps and EMA steps in 30 fps units,
    while the browser analysis stream runs at 12-20 fps. Each real pose frame is
    expanded into the 30 Hz frames that fall between it and the previous real frame,
    with landmark coordinates linearly interpolated.
    """

    def __init__(
        self,
        rate_hz: float = ENGINE_RATE_HZ,
        max_gap_sec: float = MAX_INTERPOLATION_GAP_SEC,
    ) -> None:
        self.rate_hz = rate_hz
        self.max_gap_sec = max_gap_sec
        self.previous: Any = None
        self.last_index = -1

    def reset(self) -> None:
        self.previous = None
        self.last_index = -1

    def push(self, signal: Any) -> list[Any]:
        target_index = int(round(signal.time_sec * self.rate_hz))
        if target_index <= self.last_index:
            self.previous = signal
            return []

        previous = self.previous
        self.previous = signal
        gap = 0.0 if previous is None else signal.time_sec - previous.time_sec
        if (
            previous is None
            or not previous.detected
            or not signal.detected
            or gap <= 0.0
            or gap > self.max_gap_sec
        ):
            self.last_index = target_index
            return [replace(signal, frame_idx=target_index)]

        first_index = self.last_index + 1
        steps = target_index - first_index + 1
        frames = []
        for index in range(first_index, target_index + 1):
            if index == target_index:
                time_sec = signal.time_sec
                weight = 1.0
            else:
                time_sec = index / self.rate_hz
                weight = min(1.0, max(0.0, (time_sec - previous.time_sec) / gap))
            values: dict[str, Any] = {}
            for field in fields(signal):
                name = field.name
                if name in {"frame_idx", "time_sec", "detected"}:
                    continue
                start = getattr(previous, name)
                end = getattr(signal, name)
                if _is_rate_field(name) and isinstance(end, float):
                    values[name] = end / steps
                elif _is_interval_field(name):
                    values[name] = end
                elif isinstance(start, float) and isinstance(end, float):
                    values[name] = start + (end - start) * weight
                else:
                    values[name] = end
            frames.append(replace(signal, frame_idx=index, time_sec=time_sec, **values))
        self.last_index = target_index
        return frames
