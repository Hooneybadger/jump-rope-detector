"""Conservative rope-catch (-1) policy applied on top of any counting engine.

Each engine emits a -1 event when, after a counted jump, the feet stay on the floor while the
body still moves (its "rope stuck" verdict). On the 20 labelled benchmark clips every such -1
was wrong: three came right after the player stopped at the end, two in the middle of steady
jumping when the engine briefly missed an airborne phase. A real catch is followed by a reset
and a restart, so this policy only commits the -1 when jumping resumes after a real pause:

  * pause = time from the last counted jump before the verdict to the next counted jump,
  * commit when min_pause_sec <= pause <= resume_window_sec and the pause is at least
    pause_interval_factor x the recent jump interval,
  * otherwise (immediate continuation, no restart, session end) the -1 is dropped.

The displayed count is engine count + withheld decrements, so the engines stay unchanged.
Known limitation: a voluntary rest followed by a restart is indistinguishable here and is
treated as a catch; separating the two needs labelled catch data.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from statistics import median


@dataclass(frozen=True)
class RopeTripConfig:
    min_pause_sec: float = 1.0
    resume_window_sec: float = 10.0
    pause_interval_factor: float = 3.0


class RopeTripPolicy:
    def __init__(self, config: RopeTripConfig | None = None) -> None:
        self.config = config or RopeTripConfig()
        self.withheld = 0
        self.pending_since: float | None = None  # time of the last +1 before a withheld -1
        self.last_count_time: float | None = None
        self.intervals: deque[float] = deque(maxlen=6)
        self.committed = 0
        self.dropped = 0

    def reset(self) -> None:
        self.__init__(self.config)

    def displayed(self, engine_count: int) -> int:
        return max(0, engine_count + self.withheld)

    def observe(self, event, time_sec: float) -> int:
        """Feed one engine event (count_delta +1 or -1); returns the count to display."""
        if event.count_delta < 0:
            if self.pending_since is None and self.last_count_time is not None:
                self.withheld += 1
                self.pending_since = self.last_count_time
            elif self.pending_since is not None:
                self.withheld += 1  # second verdict during the same pause: never deduct twice
                self.dropped += 1
            return self.displayed(event.running_count)

        if self.pending_since is not None:
            pause = time_sec - self.pending_since
            typical = median(self.intervals) if self.intervals else 0.0
            if (self.config.min_pause_sec <= pause <= self.config.resume_window_sec
                    and pause >= self.config.pause_interval_factor * typical):
                self.withheld -= 1
                self.committed += 1
            else:
                self.dropped += 1
            self.pending_since = None
        elif self.last_count_time is not None:
            self.intervals.append(time_sec - self.last_count_time)
        self.last_count_time = time_sec
        return self.displayed(event.running_count)

    def expire(self, time_sec: float) -> None:
        """Drop a withheld -1 whose restart window has passed (keeps the last jump counted)."""
        if self.pending_since is not None and time_sec - self.pending_since > self.config.resume_window_sec:
            self.pending_since = None
            self.dropped += 1
