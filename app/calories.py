"""Estimated energy cost of a jump-rope session and the cheer tier it earns.

MET values come from the 2024 Adult Compendium of Physical Activities
(Herrmann SD et al., J Sport Health Sci 2024;13(1):6-12, doi:10.1016/j.jshs.2023.10.010):
  15552  8.3 MET  rope jumping, slow pace, < 100 skips/min, 2 foot skip
  15551 11.8 MET  rope jumping, moderate pace, 100 to 120 skips/min, 2 foot skip
  15550 12.3 MET  rope jumping, fast pace, 120-160 skips/min
  15554 10.0 MET  rope jumping, double under or more
The Compendium has no alternate-foot code. Choi DH (2004, Exercise Science 13(1):25-34)
found no difference in oxygen uptake between two-foot and alternate-foot skipping at the
same rope rate, so alternating jumps use the same pace bands as basic jumps.
Calories follow the ACSM equation: kcal/min = MET x 3.5 x body mass (kg) / 200.
"""
from __future__ import annotations

from dataclasses import dataclass

REFERENCE_WEIGHT_KG = 60
WEIGHT_TABLE_KG = (40, 50, 60, 70, 80)


@dataclass(frozen=True)
class MetBasis:
    met: float
    code: str
    label: str


SLOW = MetBasis(8.3, "15552", "느린 속도 (분당 100회 미만)")
MODERATE = MetBasis(11.8, "15551", "보통 속도 (분당 100~120회)")
FAST = MetBasis(12.3, "15550", "빠른 속도 (분당 120회 이상)")
DOUBLE_UNDER = MetBasis(10.0, "15554", "이중뛰기 (double under)")


@dataclass(frozen=True)
class CalorieEstimate:
    basis: MetBasis | None
    pace_per_min: float
    minutes: float
    weight_kg: float
    kcal: float

    def kcal_for(self, minutes: float, weight_kg: float | None = None) -> float:
        if self.basis is None:
            return 0.0
        return kcal(self.basis.met, weight_kg or self.weight_kg, minutes)


def kcal(met: float, weight_kg: float, minutes: float) -> float:
    return met * 3.5 * weight_kg / 200 * minutes


def met_basis(mode: str, pace_per_min: float) -> MetBasis:
    if mode == "double":
        return DOUBLE_UNDER
    if pace_per_min >= 120:
        return FAST
    if pace_per_min >= 100:
        return MODERATE
    return SLOW


def estimate(mode: str, count: int, duration_seconds: int, weight_kg: float = REFERENCE_WEIGHT_KG) -> CalorieEstimate:
    minutes = max(0, duration_seconds) / 60
    if count <= 0 or minutes <= 0:
        return CalorieEstimate(None, 0.0, minutes, weight_kg, 0.0)
    pace = count / minutes
    basis = met_basis(mode, pace)
    return CalorieEstimate(basis, pace, minutes, weight_kg, kcal(basis.met, weight_kg, minutes))


@dataclass(frozen=True)
class CheerTier:
    level: int
    threshold: int
    title: str
    message: str
    next_threshold: int | None


# Count thresholds per session. Double unders take far more effort per count.
TIER_THRESHOLDS = {
    "basic": (0, 30, 100, 200, 400),
    "alternating": (0, 30, 100, 200, 400),
    "double": (0, 10, 30, 60, 120),
}
TIER_COPY = (
    ("첫걸음을 뗐어요", "줄을 잡은 오늘이 시작입니다. 다음에는 첫 단계 목표를 넘겨 봐요."),
    ("좋은 출발이에요", "몸이 리듬을 기억하기 시작했어요. 이 박자를 조금만 더 길게 이어 가요."),
    ("리듬을 탔어요", "끊기지 않는 점프가 쌓이고 있어요. 호흡을 고르게 유지하면 더 멀리 갑니다."),
    ("지치지 않는 점프", "꾸준한 페이스가 돋보이는 기록이에요. 착지를 가볍게 유지해 보세요."),
    ("오늘의 챔피언", "최고 단계를 달성했어요. 이 기록을 다음 측정의 기준으로 삼아 보세요."),
)


def cheer_tier(mode: str, count: int) -> CheerTier:
    thresholds = TIER_THRESHOLDS.get(mode, TIER_THRESHOLDS["basic"])
    level = sum(1 for threshold in thresholds[1:] if count >= threshold)
    title, message = TIER_COPY[level]
    next_threshold = thresholds[level + 1] if level + 1 < len(thresholds) else None
    return CheerTier(level, thresholds[level], title, message, next_threshold)
