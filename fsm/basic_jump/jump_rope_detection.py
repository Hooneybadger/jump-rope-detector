import numpy as np

from jump_rope_settings import *


def get_true_ratio(flags, end_idx, window_size):
    if not flags:
        return 0.0
    start_idx = max(0, end_idx - window_size + 1)
    window = flags[start_idx:end_idx + 1]
    if not window:
        return 0.0
    return float(sum(1 for flag in window if flag)) / float(len(window))


def precompute_true_ratio_series(flags, window_size):
    if not flags:
        return []
    if window_size <= 0:
        window_size = 1
    flag_arr = np.array([1 if flag else 0 for flag in flags], dtype=np.int32)
    cumsum = np.cumsum(flag_arr, dtype=np.int64)
    ratio_series = np.zeros(len(flag_arr), dtype=np.float32)
    for idx in range(len(flag_arr)):
        start_idx = max(0, idx - window_size + 1)
        total = int(cumsum[idx]) - (int(cumsum[start_idx - 1]) if start_idx > 0 else 0)
        ratio_series[idx] = float(total) / float(idx - start_idx + 1)
    return ratio_series.tolist()


def get_local_min_prominence(series, candidate_idx, window_size):
    if candidate_idx < 1 or (candidate_idx + 1) >= len(series):
        return None
    window_size = max(1, int(window_size))
    center = float(series[candidate_idx])
    if not np.isfinite(center):
        return None
    left_start = max(0, candidate_idx - window_size)
    right_end = min(len(series), candidate_idx + 1 + window_size)
    left_neighbors = np.array(series[left_start:candidate_idx], dtype=np.float32)
    right_neighbors = np.array(series[candidate_idx + 1:right_end], dtype=np.float32)
    left_neighbors = left_neighbors[np.isfinite(left_neighbors)]
    right_neighbors = right_neighbors[np.isfinite(right_neighbors)]
    if left_neighbors.size == 0 or right_neighbors.size == 0:
        return None
    prominence_left = float(np.max(left_neighbors) - center)
    prominence_right = float(np.max(right_neighbors) - center)
    return float(min(prominence_left, prominence_right))


def get_local_range(series, candidate_idx, window_size):
    if not series:
        return None
    if candidate_idx < 0 or candidate_idx >= len(series):
        return None
    window_size = max(1, int(window_size))
    start_idx = max(0, candidate_idx - window_size)
    end_idx = min(len(series), candidate_idx + 1 + window_size)
    window = np.array(series[start_idx:end_idx], dtype=np.float32)
    window = window[np.isfinite(window)]
    if window.size < 2:
        return None
    return float(np.max(window) - np.min(window))


def get_local_velocity_correlation(series_a, series_b, candidate_idx, window_size):
    if not series_a or not series_b:
        return None
    max_len = min(len(series_a), len(series_b))
    if candidate_idx < 1 or candidate_idx >= max_len:
        return None
    window_size = max(2, int(window_size))
    start_idx = max(0, candidate_idx - window_size)
    end_idx = min(max_len, candidate_idx + 1 + window_size)
    series_a_window = np.array(series_a[start_idx:end_idx], dtype=np.float32)
    series_b_window = np.array(series_b[start_idx:end_idx], dtype=np.float32)
    finite_mask = np.isfinite(series_a_window) & np.isfinite(series_b_window)
    if int(np.sum(finite_mask)) < 4:
        return None
    series_a_window = series_a_window[finite_mask]
    series_b_window = series_b_window[finite_mask]
    if series_a_window.size < 4 or series_b_window.size < 4:
        return None
    vel_a = np.diff(series_a_window)
    vel_b = np.diff(series_b_window)
    finite_vel_mask = np.isfinite(vel_a) & np.isfinite(vel_b)
    vel_a = vel_a[finite_vel_mask]
    vel_b = vel_b[finite_vel_mask]
    if vel_a.size < 3 or vel_b.size < 3:
        return None
    if float(np.std(vel_a)) < 1e-6 or float(np.std(vel_b)) < 1e-6:
        return None
    corr = float(np.corrcoef(vel_a, vel_b)[0, 1])
    if not np.isfinite(corr):
        return None
    return corr


def evaluate_foot_motion_signature(
    candidate_idx,
    left_foot_y,
    right_foot_y,
    foot_center_y,
    foot_center_x=None,
    prominence_window=LOCAL_PROMINENCE_WINDOW,
    foot_lift_min_prominence=STRICT_FOOT_LIFT_MIN_PROMINENCE,
    both_feet_min_prominence=STRICT_BOTH_FEET_MIN_PROMINENCE,
    feet_symmetry_min_ratio=STRICT_FEET_SYMMETRY_MIN_RATIO,
    foot_sync_window=STRICT_FOOT_SYNC_WINDOW,
    foot_sync_min_corr=STRICT_FOOT_SYNC_MIN_CORR,
    require_inplace=STRICT_REQUIRE_INPLACE,
    inplace_window=STRICT_INPLACE_WINDOW,
    inplace_max_center_drift=STRICT_INPLACE_MAX_CENTER_DRIFT,
    require_advanced_motion=STRICT_REQUIRE_ADVANCED_MOTION,
    advanced_min_checks=STRICT_MOTION_ADVANCED_MIN_CHECKS,
):
    foot_prominence = get_local_min_prominence(
        foot_center_y,
        candidate_idx,
        prominence_window,
    )
    left_foot_prominence = get_local_min_prominence(
        left_foot_y,
        candidate_idx,
        prominence_window,
    )
    right_foot_prominence = get_local_min_prominence(
        right_foot_y,
        candidate_idx,
        prominence_window,
    )
    foot_motion_strict_ok = (
        (foot_prominence is not None)
        and (foot_prominence >= float(foot_lift_min_prominence))
    )

    both_feet_lift_ok = (
        (left_foot_prominence is not None)
        and (right_foot_prominence is not None)
        and (left_foot_prominence >= float(both_feet_min_prominence))
        and (right_foot_prominence >= float(both_feet_min_prominence))
    )

    feet_symmetry_ratio = 0.0
    if both_feet_lift_ok:
        max_prominence = max(float(left_foot_prominence), float(right_foot_prominence))
        min_prominence = min(float(left_foot_prominence), float(right_foot_prominence))
        feet_symmetry_ratio = min_prominence / max(max_prominence, 1e-9)
    feet_symmetry_ok = feet_symmetry_ratio >= float(feet_symmetry_min_ratio)

    foot_sync_corr = get_local_velocity_correlation(
        left_foot_y,
        right_foot_y,
        candidate_idx,
        foot_sync_window,
    )
    foot_sync_ok = (
        (foot_sync_corr is not None)
        and (foot_sync_corr >= float(foot_sync_min_corr))
    )
    if foot_sync_corr is None and both_feet_lift_ok and feet_symmetry_ok:
        foot_sync_ok = True

    center_x_drift = None
    if foot_center_x is not None:
        center_x_drift = get_local_range(
            foot_center_x,
            candidate_idx,
            inplace_window,
        )
    if bool(require_inplace):
        inplace_ok = (
            (center_x_drift is not None)
            and (center_x_drift <= float(inplace_max_center_drift))
        )
    else:
        inplace_ok = True

    advanced_ok_count = (
        int(bool(both_feet_lift_ok))
        + int(bool(feet_symmetry_ok))
        + int(bool(foot_sync_ok))
        + int(bool(inplace_ok) and bool(require_inplace))
    )
    strict_motion_ok = bool(foot_motion_strict_ok)
    if bool(require_advanced_motion):
        strict_motion_ok = (
            strict_motion_ok
            and advanced_ok_count >= max(1, int(advanced_min_checks))
        )

    return {
        "foot_prominence": (float(foot_prominence) if foot_prominence is not None else None),
        "left_foot_prominence": (
            float(left_foot_prominence) if left_foot_prominence is not None else None
        ),
        "right_foot_prominence": (
            float(right_foot_prominence) if right_foot_prominence is not None else None
        ),
        "foot_motion_strict_ok": bool(foot_motion_strict_ok),
        "both_feet_lift_ok": bool(both_feet_lift_ok),
        "feet_symmetry_ratio": float(feet_symmetry_ratio),
        "feet_symmetry_ok": bool(feet_symmetry_ok),
        "foot_sync_corr": (float(foot_sync_corr) if foot_sync_corr is not None else None),
        "foot_sync_ok": bool(foot_sync_ok),
        "center_x_drift": (float(center_x_drift) if center_x_drift is not None else None),
        "inplace_ok": bool(inplace_ok),
        "advanced_ok_count": int(advanced_ok_count),
        "require_advanced_motion": bool(require_advanced_motion),
        "strict_motion_ok": bool(strict_motion_ok),
    }


def has_stable_entry_cadence(pending_candidates, min_events, max_gap_frames, max_cv):
    if len(pending_candidates) < min_events:
        return False
    recent = pending_candidates[-min_events:]
    frames = np.array([item["candidate_frame"] for item in recent], dtype=np.float32)
    if frames.size < 2:
        return False
    intervals = np.diff(frames)
    if intervals.size == 0:
        return False
    if np.any(intervals <= 0):
        return False
    if float(np.max(intervals)) > float(max_gap_frames):
        return False
    mean_interval = float(np.mean(intervals))
    if mean_interval <= 0.0:
        return False
    cadence_cv = float(np.std(intervals) / mean_interval)
    return cadence_cv <= float(max_cv)


def choose_entry_backfill_tail_count(pending_candidates, active_enter_min_events=None):
    default_tail = max(0, int(ACTIVE_ENTER_CONFIRM_TAIL_EVENTS))
    if not ENABLE_ADAPTIVE_ENTRY_BACKFILL:
        return default_tail
    if not pending_candidates:
        return default_tail

    if active_enter_min_events is None:
        active_enter_min_events = ACTIVE_ENTER_MIN_EVENTS
    recent_count = max(1, int(active_enter_min_events))
    recent = pending_candidates[-recent_count:]
    rope_values = [
        float(item["rope_ratio"])
        for item in recent
        if item.get("rope_ratio") is not None
    ]
    if not rope_values:
        return default_tail
    dual_values = [
        float(item["rope_dual_ratio"])
        for item in recent
        if item.get("rope_dual_ratio") is not None
    ]
    strength_values = [
        float(item["strength_ratio"])
        for item in recent
        if item.get("strength_ratio") is not None
    ]
    median_rope = float(np.median(rope_values))
    median_dual = float(np.median(dual_values)) if dual_values else 1.0
    median_strength = float(np.median(strength_values)) if strength_values else 0.0
    max_rope = max(rope_values) if rope_values else 0.0
    very_low_conf_entry = (
        median_rope <= float(ENTRY_BACKFILL_VERY_LOW_ROPE_RATIO)
        and median_dual <= float(ENTRY_BACKFILL_VERY_LOW_DUAL_RATIO)
    )
    if very_low_conf_entry:
        tail_count = max(0, int(ENTRY_BACKFILL_VERY_LOW_CONF_TAIL_EVENTS))
        if bool(ENTRY_BACKFILL_STRONG_ENTRY_FORCE_ENABLED):
            strong_entry = (
                median_strength >= float(ENTRY_BACKFILL_STRONG_ENTRY_MIN_STRENGTH_RATIO)
                and max_rope >= float(ENTRY_BACKFILL_STRONG_ENTRY_MIN_ROPE_RATIO)
            )
            if strong_entry:
                force_tail = max(1, int(ENTRY_BACKFILL_STRONG_ENTRY_FORCE_TAIL_EVENTS))
                tail_count = max(tail_count, min(len(recent), force_tail))
        return int(tail_count)
    high_conf_min_rope = float(ENTRY_BACKFILL_HIGH_CONF_MIN_ROPE_RATIO)
    high_conf_min_dual = float(ENTRY_BACKFILL_HIGH_CONF_MIN_DUAL_RATIO)
    high_conf_entry = (
        median_rope >= high_conf_min_rope
        and median_dual >= high_conf_min_dual
    )
    if high_conf_entry:
        return max(1, int(ENTRY_BACKFILL_HIGH_CONF_TAIL_EVENTS))
    return max(0, int(ENTRY_BACKFILL_LOW_CONF_TAIL_EVENTS))


def build_detected_jump_event(
    candidate_idx,
    landing_offset_frames,
    event_time_bias_ms,
    frame_numbers,
    frame_timestamps_ms,
    left_foot_x,
    left_foot_y,
    right_foot_x,
    right_foot_y,
    fps,
    width,
    height,
    count,
    strength_ratio,
    rope_ratio=None,
    rope_dual_ratio=None,
    foot_prominence=None,
    left_foot_prominence=None,
    right_foot_prominence=None,
    both_feet_lift_ok=None,
    feet_symmetry_ratio=None,
    feet_symmetry_ok=None,
    foot_sync_corr=None,
    foot_sync_ok=None,
    inplace_ok=None,
):
    landing_idx = min(candidate_idx + landing_offset_frames, len(frame_numbers) - 1)
    event_frame = frame_numbers[landing_idx]
    if frame_timestamps_ms and landing_idx < len(frame_timestamps_ms):
        base_timestamp_ms = float(frame_timestamps_ms[landing_idx])
    else:
        base_timestamp_ms = (event_frame * 1000.0) / fps
    event_timestamp_ms = int(round(base_timestamp_ms + event_time_bias_ms))
    detected_feet = sorted(
        [
            (left_foot_x[landing_idx] * width, left_foot_y[landing_idx] * height),
            (right_foot_x[landing_idx] * width, right_foot_y[landing_idx] * height),
        ],
        key=lambda pt: pt[0],
    )
    return {
        "count": count,
        "frame": event_frame,
        "timestamp_ms": event_timestamp_ms,
        "left_point": detected_feet[0],
        "right_point": detected_feet[1],
        "strength_ratio": strength_ratio,
        "rope_ratio": rope_ratio,
        "rope_dual_ratio": rope_dual_ratio,
        "foot_prominence": foot_prominence,
        "left_foot_prominence": left_foot_prominence,
        "right_foot_prominence": right_foot_prominence,
        "both_feet_lift_ok": both_feet_lift_ok,
        "feet_symmetry_ratio": feet_symmetry_ratio,
        "feet_symmetry_ok": feet_symmetry_ok,
        "foot_sync_corr": foot_sync_corr,
        "foot_sync_ok": foot_sync_ok,
        "inplace_ok": inplace_ok,
    }


def get_adaptive_jump_threshold_at_index(readings, end_idx):
    if end_idx < 0 or end_idx >= len(readings):
        return ADAPTIVE_THRESHOLD_WARMUP
    if (end_idx + 1) < 10:
        return ADAPTIVE_THRESHOLD_WARMUP

    start_idx = max(0, end_idx - ADAPTIVE_THRESHOLD_LOOKBACK + 1)
    recent = np.array(readings[start_idx:end_idx + 1], dtype=np.float32)
    recent_diffs = np.abs(np.diff(recent))
    if recent_diffs.size == 0:
        return ADAPTIVE_THRESHOLD_WARMUP

    median = np.median(recent_diffs)
    mad = np.median(np.abs(recent_diffs - median))
    robust_spread = max(mad, median * 0.5)
    threshold = ADAPTIVE_THRESHOLD_SCALE * robust_spread
    threshold = np.clip(threshold, ADAPTIVE_THRESHOLD_MIN, LEGACY_JUMP_THRESHOLD)
    return float(threshold)


def precompute_adaptive_threshold_series(readings):
    thresholds = []
    for idx in range(len(readings)):
        thresholds.append(get_adaptive_jump_threshold_at_index(readings, idx))
    return thresholds


def get_adaptive_jump_threshold(readings):
    if not readings:
        return ADAPTIVE_THRESHOLD_WARMUP
    return get_adaptive_jump_threshold_at_index(readings, len(readings) - 1)
