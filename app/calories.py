"""Estimated energy cost of a jump-rope session, the cheer tier it earns, and a daily time goal.

Energy cost follows the age range of each Compendium of Physical Activities.

6-18 y: Youth Compendium (Butte NF et al., Med Sci Sports Exerc 2018;50(2):246-256). Jump rope
  (10260X) uses the smoothed METy values that NCCOR recommends for energy estimates:
  6.9/7.1/7.2/7.4 at 6-9/10-12/13-15/16-18 y. kcal = METy x BMR x time, with BMR from the
  age-, sex- and mass-specific Schofield equations (Schofield WN, Hum Nutr Clin Nutr
  1985;39 Suppl 1:5-41), as NCCOR advises for consistency.
19-59 y: 2024 Adult Compendium (Herrmann SD et al., J Sport Health Sci 2024;13(1):6-12):
  15552  8.3 MET  rope jumping, slow pace, < 100 skips/min, 2 foot skip
  15551 11.8 MET  rope jumping, moderate pace, 100 to 120 skips/min, 2 foot skip
  15550 12.3 MET  rope jumping, fast pace, 120-160 skips/min
  15554 10.0 MET  rope jumping, double under or more
  kcal/min = MET x 3.5 x body mass (kg) / 200. The Compendium has no alternate-foot code.
  Choi DH (2004, Exercise Science 13(1):25-34) found no difference in oxygen uptake between
  two-foot and alternate-foot skipping at the same rope rate, so both use the pace bands.
60+ y: the Older Adult Compendium (Willis EA et al., J Sport Health Sci 2024;13(1):13-17)
  rates intensity against a resting rate of 2.7 mL/kg/min (MET60+) but lists no rope
  jumping, so the adult oxygen cost is kept and only reported as MET60+.

A MET multiplies a standard resting rate of 3.5 mL/kg/min. Resting energy really depends on
sex, age, height and body mass, so the net cost (gross minus resting) subtracts the user's own
resting rate: Harris-Benedict for adults, the method the Compendium uses for corrected METs
(Kozey S et al., J Phys Act Health 2010;7(4):508-516), and Schofield for youth.

The recommended daily time follows the Korean physical activity guidelines (Ministry of Health
and Welfare, 2023), which match the WHO 2020 guidelines: 60 min/day at 6-18 y; 75-150 min/week
of vigorous activity at 19-64 y; 75-100 min/week at 65+ y. Rope jumping is vigorous, so the
lower bound is spread over 7 days, and the upper bound applies at BMI 25 or more, the Korean
obesity threshold. The Korean Society for the Study of Obesity suggests 250-300 min/week of
aerobic exercise for meaningful weight loss (Kim KK et al., J Obes Metab Syndr 2023;32(1):1-24),
and 150 vigorous minutes equal 300 moderate minutes.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

REFERENCE_WEIGHT_KG = 60
WEIGHT_TABLE_KG = (40, 50, 60, 70, 80)
# Youth without a body mass: a reference and table that fit children rather than adults.
YOUTH_REFERENCE_WEIGHT_KG = 40
YOUTH_WEIGHT_TABLE_KG = (20, 30, 40, 50, 60)

SEX_NAMES = {"male": "남성", "female": "여성"}
PROFILE_AGE_MIN, PROFILE_AGE_MAX = 6, 100
YOUTH_AGE_MIN, YOUTH_AGE_MAX = 6, 18
OLDER_ADULT_AGE = 60
GUIDELINE_OLDER_AGE = 65
STANDARD_RESTING_ML = 3.5
OLDER_RESTING_ML = 2.7
OBESITY_BMI = 25
# Youth Compendium jump rope (10260X), smoothed METy by age group: (upper age, METy, label).
YOUTH_JUMP_ROPE = ((9, 6.9, "6~9세"), (12, 7.1, "10~12세"), (15, 7.2, "13~15세"), (18, 7.4, "16~18세"))
# Schofield (1985) BMR in kcal/day = a x weight + b: (upper age, a, b).
SCHOFIELD = {
    "male": ((9, 22.706, 504.3), (18, 17.686, 658.2)),
    "female": ((9, 20.315, 485.9), (18, 13.384, 692.6)),
}
# Harris-Benedict RMR in kcal/day = a + b x weight + c x height - d x age, as the Compendium gives it.
HARRIS_BENEDICT = {
    "male": (66.4730, 13.7516, 5.0033, 6.7550),
    "female": (655.0955, 9.5634, 1.8496, 4.6756),
}

GUIDELINE_SOURCE = "보건복지부, 한국인을 위한 신체활동 지침서(2023)"
OBESITY_SOURCE = "대한비만학회 비만 진료지침 2022(J Obes Metab Syndr 2023;32(1):1-24)"


def _average(values: list[float]) -> float:
    return sum(values) / len(values)


def _sexes(sex: str | None, table: dict) -> list[str]:
    """Without a sex, both equations are averaged."""
    return [sex] if sex in table else list(table)


def is_youth(age: int | None) -> bool:
    return age is not None and YOUTH_AGE_MIN <= age <= YOUTH_AGE_MAX


def schofield_bmr(sex: str | None, age: int, weight_kg: float) -> float:
    """Basal metabolic rate in kcal/day for 6-18 y."""
    values = []
    for key in _sexes(sex, SCHOFIELD):
        a, b = next((a, b) for upper, a, b in SCHOFIELD[key] if age <= upper)
        values.append(a * weight_kg + b)
    return _average(values)


def harris_benedict_rmr(sex: str | None, age: int, height_cm: float, weight_kg: float) -> float:
    """Resting metabolic rate in kcal/day for adults."""
    return _average([a + b * weight_kg + c * height_cm - d * age
                     for a, b, c, d in (HARRIS_BENEDICT[key] for key in _sexes(sex, HARRIS_BENEDICT))])


@dataclass(frozen=True)
class MetBasis:
    met: float
    code: str
    label: str
    youth: bool = False


@dataclass(frozen=True)
class BodyProfile:
    sex: str | None = None
    age: int | None = None
    height_cm: float | None = None
    weight_kg: float | None = None

    @property
    def bmi(self) -> float | None:
        if not self.height_cm or not self.weight_kg:
            return None
        return self.weight_kg / (self.height_cm / 100) ** 2

    @property
    def is_empty(self) -> bool:
        return self.sex is None and self.age is None and self.height_cm is None and self.weight_kg is None

    @property
    def uses_youth_compendium(self) -> bool:
        return is_youth(self.age)

    @property
    def is_older_adult(self) -> bool:
        return self.age is not None and self.age >= OLDER_ADULT_AGE

    def summary(self) -> str:
        parts = []
        if self.sex in SEX_NAMES:
            parts.append(SEX_NAMES[self.sex])
        if self.age is not None:
            parts.append(f"{self.age}세")
        if self.height_cm:
            parts.append(f"{self.height_cm:g}cm")
        if self.weight_kg:
            parts.append(f"{self.weight_kg:g}kg")
        return " · ".join(parts)


SLOW = MetBasis(8.3, "15552", "느린 속도 (분당 100회 미만)")
MODERATE = MetBasis(11.8, "15551", "보통 속도 (분당 100~120회)")
FAST = MetBasis(12.3, "15550", "빠른 속도 (분당 120회 이상)")
DOUBLE_UNDER = MetBasis(10.0, "15554", "이중뛰기 (double under)")


def resting_kcal_per_min(profile: BodyProfile | None, weight_kg: float) -> tuple[float, str]:
    """The user's resting energy and the method behind it: schofield, harris-benedict or standard (1 MET)."""
    if profile is not None and is_youth(profile.age):
        return schofield_bmr(profile.sex, profile.age, weight_kg) / 1440, "schofield"
    if profile is not None and profile.age is not None and profile.height_cm:
        return harris_benedict_rmr(profile.sex, profile.age, profile.height_cm, weight_kg) / 1440, "harris-benedict"
    return STANDARD_RESTING_ML * weight_kg / 200, "standard"


@dataclass(frozen=True)
class CalorieEstimate:
    basis: MetBasis | None
    pace_per_min: float
    minutes: float
    weight_kg: float
    kcal: float
    profile: BodyProfile | None = None

    @property
    def personal(self) -> bool:
        """True when the user's own body mass drives the estimate."""
        return bool(self.profile and self.profile.weight_kg)

    @property
    def resting_method(self) -> str:
        return resting_kcal_per_min(self.profile, self.weight_kg)[1]

    @property
    def net_kcal(self) -> float:
        return self.net_for(self.minutes)

    @property
    def met60(self) -> float | None:
        """Intensity against the 2.7 mL/kg/min resting rate of adults aged 60 and older."""
        if self.basis is None or self.basis.youth or self.profile is None or not self.profile.is_older_adult:
            return None
        return self.basis.met * STANDARD_RESTING_ML / OLDER_RESTING_ML

    def kcal_for(self, minutes: float, weight_kg: float | None = None) -> float:
        if self.basis is None:
            return 0.0
        weight = weight_kg or self.weight_kg
        if self.basis.youth and self.profile is not None and self.profile.age is not None:
            return youth_kcal(self.basis.met, self.profile.sex, self.profile.age, weight, minutes)
        return kcal(self.basis.met, weight, minutes)

    def net_for(self, minutes: float, weight_kg: float | None = None) -> float:
        """Energy above rest: gross minus the user's own resting energy over the same time."""
        if self.basis is None:
            return 0.0
        weight = weight_kg or self.weight_kg
        resting = resting_kcal_per_min(self.profile, weight)[0] * minutes
        return max(0.0, self.kcal_for(minutes, weight) - resting)


def kcal(met: float, weight_kg: float, minutes: float) -> float:
    return met * STANDARD_RESTING_ML * weight_kg / 200 * minutes


def youth_kcal(mety: float, sex: str | None, age: int, weight_kg: float, minutes: float) -> float:
    return mety * schofield_bmr(sex, age, weight_kg) / 1440 * minutes


def youth_basis(age: int) -> MetBasis:
    _, mety, band = next(item for item in YOUTH_JUMP_ROPE if age <= item[0])
    return MetBasis(mety, "10260X", f"청소년 줄넘기 ({band})", youth=True)


def met_basis(mode: str, pace_per_min: float) -> MetBasis:
    if mode == "double":
        return DOUBLE_UNDER
    if pace_per_min >= 120:
        return FAST
    if pace_per_min >= 100:
        return MODERATE
    return SLOW


def reference_weight(profile: BodyProfile | None) -> float:
    if profile is not None and profile.weight_kg:
        return profile.weight_kg
    return YOUTH_REFERENCE_WEIGHT_KG if profile is not None and is_youth(profile.age) else REFERENCE_WEIGHT_KG


def weight_table(profile: BodyProfile | None) -> tuple[float, ...]:
    """Body masses for the conversion table when the user has not entered their own."""
    return YOUTH_WEIGHT_TABLE_KG if profile is not None and is_youth(profile.age) else WEIGHT_TABLE_KG


def estimate(mode: str, count: int, duration_seconds: int, weight_kg: float | None = None,
             profile: BodyProfile | None = None) -> CalorieEstimate:
    weight_kg = profile.weight_kg if profile is not None and profile.weight_kg else weight_kg or reference_weight(profile)
    minutes = max(0, duration_seconds) / 60
    if count <= 0 or minutes <= 0:
        return CalorieEstimate(None, 0.0, minutes, weight_kg, 0.0, profile)
    pace = count / minutes
    basis = youth_basis(profile.age) if profile is not None and profile.uses_youth_compendium else met_basis(mode, pace)
    result = CalorieEstimate(basis, pace, minutes, weight_kg, 0.0, profile)
    return replace(result, kcal=result.kcal_for(minutes))


@dataclass(frozen=True)
class Recommendation:
    seconds: int
    reason: str
    source: str = GUIDELINE_SOURCE

    @property
    def minutes(self) -> float:
        return self.seconds / 60

    @property
    def label(self) -> str:
        return duration_label(self.seconds)


def duration_label(seconds: int) -> str:
    """100 -> 1분 40초, 1800 -> 30분, 45 -> 45초."""
    minutes, rest = divmod(max(0, int(seconds)), 60)
    parts = [f"{minutes}분"] if minutes else []
    if rest or not minutes:
        parts.append(f"{rest}초")
    return " ".join(parts)


def weekly_to_daily_seconds(weekly_minutes: int) -> int:
    return round(weekly_minutes * 60 / 7)


def recommendation(profile: BodyProfile | None) -> Recommendation | None:
    """Daily jump-rope time from the Korean physical activity guidelines; None until a profile has been entered.

    The guidelines do not differ by sex. Without an age the adult rule applies.
    """
    if profile is None or profile.is_empty:
        return None
    if is_youth(profile.age):
        return Recommendation(60 * 60, "만 6~18세 권장량: 매일 60분 이상 유산소 신체활동")
    older = profile.age is not None and profile.age >= GUIDELINE_OLDER_AGE
    group, upper = ("만 65세 이상", 100) if older else ("만 19~64세", 150)
    balance = " · 균형 운동도 주 3일 이상 권장" if older else ""
    bmi = profile.bmi
    if bmi is not None and bmi >= OBESITY_BMI:
        return Recommendation(
            weekly_to_daily_seconds(upper),
            f"체질량지수 {bmi:.1f}(비만 기준 {OBESITY_BMI} 이상)이라 {group} 고강도 권장량 상한 주 {upper}분을 7일로 나눈 값{balance}",
            f"{GUIDELINE_SOURCE}; {OBESITY_SOURCE}",
        )
    return Recommendation(weekly_to_daily_seconds(75), f"{group} 권장량: 고강도 주 75분(하한)을 7일로 나눈 값{balance}")


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
