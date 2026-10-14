"""Experiment only (not used by the app): re-express a frame-tuned EngineConfig for another input rate.

Result on the 20 labelled benchmark videos (2026-10-14): scaling made counts worse (basic 90.2% -> 81.6% at
20 Hz, alternating 98.4% -> 80.0% at native fps), so the app keeps the frame-tuned configs and instead
stops down-sampling the counter input. Kept for future experiments with independent data.

The basic and alternating engines count durations in frames, smooth with per-frame EMA
coefficients and compare per-frame velocities. Those values were tuned on video at a
reference rate, so feeding the engine at another rate silently changes every time constant.
scale_config() keeps the physical meaning (seconds, time constants, velocity per second):

  * frame counts                 -> round(value * rate / reference), at least 1
  * per-frame EMA coefficients   -> 1 - (1 - alpha) ** (reference / rate)
  * per-frame velocities / decay -> value * reference / rate

Counts of events (interval history lengths) and dimensionless ratios are unchanged.
The double-under engine is not scaled: SignalResampler always feeds it at 30 Hz.
"""
from __future__ import annotations

from dataclasses import fields, replace

FRAME_FIELDS = {
    "basic": {
        "min_refractory_frames", "fast_min_refractory_frames", "fast_mode_cadence_threshold",
        "motion_window_frames", "recent_window_frames", "min_count_gap_frames", "adaptive_gap_floor_frames",
        "rope_stuck_window_frames", "rope_stuck_min_hold_frames", "floor_reacquire_frames",
    },
    "alternating": {
        "support_streak_frames", "motion_window_frames", "recent_window_frames", "min_count_gap_frames",
        "adaptive_gap_floor_frames", "rearm_after_gap_min_frames", "rearm_fast_interval_min",
        "rearm_fast_interval_max", "rearm_interval_spread_max", "miss_recovery_fast_interval_min",
        "miss_recovery_fast_interval_max", "miss_recovery_interval_spread_max", "relaxed_contact_fast_interval_min",
        "relaxed_contact_fast_interval_max", "relaxed_contact_interval_spread_max", "alternation_reset_gap_frames",
        "alternation_recovery_extra_count_min_interval", "dual_air_window_frames", "arm_motion_window_frames",
        "arm_missing_dual_air_min_active_frames", "weak_support_followup_min_gap_frames",
        "weak_support_bonus_timeout_frames", "weak_support_dual_air_active_frames", "bootstrap_max_gap_frames",
        "rope_stuck_window_frames", "rope_stuck_min_hold_frames", "floor_reacquire_frames",
    },
}
ALPHA_FIELDS = {
    "basic": {"ema_alpha_hip", "ema_alpha_foot", "fast_ema_alpha_hip", "fast_ema_alpha_foot",
              "baseline_alpha_hip", "baseline_alpha_foot"},
    "alternating": {"ema_alpha_hip", "ema_alpha_foot", "baseline_alpha_hip"},
}
PER_FRAME_FIELDS = {
    "basic": {"descend_velocity_ratio", "ascend_velocity_ratio", "fast_descend_velocity_ratio",
              "fast_ascend_velocity_ratio", "floor_decay_ratio", "rope_stuck_attempt_velocity_ratio"},
    "alternating": {"floor_decay_ratio"},
}


def scale_config(config, mode: str, rate_hz: float, reference_hz: float):
    if mode not in FRAME_FIELDS or rate_hz <= 0 or abs(rate_hz - reference_hz) < 1e-6:
        return config
    ratio = rate_hz / reference_hz
    names = {field.name for field in fields(config)}
    changes = {}
    for name in FRAME_FIELDS[mode] & names:
        changes[name] = max(1, int(round(getattr(config, name) * ratio)))
    for name in ALPHA_FIELDS[mode] & names:
        changes[name] = 1.0 - (1.0 - getattr(config, name)) ** (1.0 / ratio)
    for name in PER_FRAME_FIELDS[mode] & names:
        value = getattr(config, name)
        if value > 0:  # negative values are "disabled" sentinels
            changes[name] = value / ratio
    return replace(config, **changes)
