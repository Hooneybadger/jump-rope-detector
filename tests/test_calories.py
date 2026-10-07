import pytest

from app import calories


@pytest.mark.parametrize(
    ("mode", "count", "seconds", "met", "code"),
    [
        ("basic", 90, 60, 8.3, "15552"),
        ("basic", 110, 60, 11.8, "15551"),
        ("basic", 130, 60, 12.3, "15550"),
        ("alternating", 110, 60, 11.8, "15551"),
        ("alternating", 300, 120, 12.3, "15550"),
        ("double", 40, 60, 10.0, "15554"),
        ("double", 150, 60, 10.0, "15554"),
    ],
)
def test_compendium_met_follows_mode_and_measured_pace(mode, count, seconds, met, code):
    estimate = calories.estimate(mode, count, seconds)
    assert estimate.basis.met == met
    assert estimate.basis.code == code


def test_acsm_equation_with_reference_weight():
    # 11.8 MET x 3.5 x 60 kg / 200 = 12.39 kcal per minute
    estimate = calories.estimate("basic", 110, 60)
    assert estimate.kcal == pytest.approx(12.39)
    assert estimate.kcal_for(10) == pytest.approx(123.9)
    assert estimate.kcal_for(1, weight_kg=80) == pytest.approx(16.52)


def test_no_jumps_means_no_calorie_estimate():
    for count, seconds in ((0, 60), (10, 0)):
        estimate = calories.estimate("basic", count, seconds)
        assert estimate.basis is None and estimate.kcal == 0


@pytest.mark.parametrize(
    ("mode", "count", "level", "next_threshold"),
    [
        ("basic", 0, 0, 30),
        ("basic", 29, 0, 30),
        ("basic", 30, 1, 100),
        ("alternating", 250, 3, 400),
        ("basic", 400, 4, None),
        ("double", 9, 0, 10),
        ("double", 30, 2, 60),
        ("double", 125, 4, None),
    ],
)
def test_cheer_tier_changes_at_count_thresholds(mode, count, level, next_threshold):
    tier = calories.cheer_tier(mode, count)
    assert tier.level == level
    assert tier.next_threshold == next_threshold
    assert tier.title and tier.message
