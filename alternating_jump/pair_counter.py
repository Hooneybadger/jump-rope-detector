"""Count one alternate-foot repetition when two observed opposite steps complete."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class AlternatingPairCounter:
    max_gap_sec: float = 1.0
    count: int = 0
    pending_side: str | None = None
    pending_time: float | None = None

    def clear_pending(self) -> None:
        self.pending_side = None
        self.pending_time = None

    def add(self, side: str, time_sec: float) -> bool:
        if side not in {"left", "right"}:
            raise ValueError("step side must be left or right")
        if self.pending_time is not None and time_sec - self.pending_time > self.max_gap_sec:
            self.clear_pending()
        if self.pending_side is not None and side != self.pending_side:
            self.count += 1
            self.clear_pending()
            return True
        # A repeated foot replaces the unfinished step; it never invents its opposite.
        self.pending_side, self.pending_time = side, time_sec
        return False

    def snapshot(self) -> tuple[int, str | None, float | None]:
        return self.count, self.pending_side, self.pending_time

    def restore(self, state: tuple[int, str | None, float | None]) -> None:
        self.count, self.pending_side, self.pending_time = state


def paired_label_times(times: list[float], feet: list[str] | None = None,
                       max_gap_sec: float = 1.0) -> list[float]:
    """Pair step labels at the second step. Legacy labels assume alternating feet.

    No detector output is used to convert ground truth. Original timestamps stay intact.
    """
    if feet is not None and len(feet) != len(times):
        raise ValueError("one foot label is required per step timestamp")
    counter = AlternatingPairCounter(max_gap_sec=max_gap_sec)
    pairs = []
    for i, time_sec in enumerate(times):
        side = feet[i] if feet is not None else ("left" if i % 2 == 0 else "right")
        if counter.add(side, time_sec):
            pairs.append(time_sec)
    return pairs
