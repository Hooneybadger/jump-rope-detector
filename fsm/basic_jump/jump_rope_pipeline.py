import argparse
import os
import time

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd

from jump_rope_detection import (
    build_detected_jump_event,
    choose_entry_backfill_tail_count,
    evaluate_foot_motion_signature,
    get_adaptive_jump_threshold,
    get_true_ratio,
    has_stable_entry_cadence,
)
from jump_rope_eval import (
    evaluate_detected_events,
    get_match_tolerance_ms,
    load_label_events,
    rewrite_tracked_video_count_overlay,
)
from jump_rope_settings import *

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose


def _is_landmark_reliable(landmarks, landmark_idx, min_visibility):
    if landmark_idx < 0 or landmark_idx >= len(landmarks):
        return False
    landmark = landmarks[landmark_idx]
    visibility = getattr(landmark, "visibility", 1.0)
    try:
        visibility = float(visibility)
    except (TypeError, ValueError):
        visibility = 0.0
    if visibility < float(min_visibility):
        return False
    x = float(landmark.x)
    y = float(landmark.y)
    return (
        np.isfinite(x)
        and np.isfinite(y)
        and -0.25 <= x <= 1.25
        and -0.25 <= y <= 1.25
    )


def has_reliable_lower_body(landmarks, min_visibility, require_full_lower_body=False):
    hips_ok = (
        _is_landmark_reliable(landmarks, mp_pose.PoseLandmark.LEFT_HIP.value, min_visibility)
        and _is_landmark_reliable(landmarks, mp_pose.PoseLandmark.RIGHT_HIP.value, min_visibility)
    )
    if not hips_ok:
        return False
    left_indices = [
        mp_pose.PoseLandmark.LEFT_KNEE.value,
        mp_pose.PoseLandmark.LEFT_ANKLE.value,
        mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value,
    ]
    right_indices = [
        mp_pose.PoseLandmark.RIGHT_KNEE.value,
        mp_pose.PoseLandmark.RIGHT_ANKLE.value,
        mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value,
    ]
    left_reliable = sum(
        1 for idx in left_indices if _is_landmark_reliable(landmarks, idx, min_visibility)
    )
    right_reliable = sum(
        1 for idx in right_indices if _is_landmark_reliable(landmarks, idx, min_visibility)
    )
    if bool(require_full_lower_body):
        left_ankle_ok = _is_landmark_reliable(
            landmarks,
            mp_pose.PoseLandmark.LEFT_ANKLE.value,
            min_visibility,
        )
        right_ankle_ok = _is_landmark_reliable(
            landmarks,
            mp_pose.PoseLandmark.RIGHT_ANKLE.value,
            min_visibility,
        )
        left_foot_ok = _is_landmark_reliable(
            landmarks,
            mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value,
            min_visibility,
        )
        right_foot_ok = _is_landmark_reliable(
            landmarks,
            mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value,
            min_visibility,
        )
        # Realtime hard guard: require both legs with ankle+foot visibility.
        return bool(
            left_ankle_ok
            and right_ankle_ok
            and left_foot_ok
            and right_foot_ok
        )
    # Default guard: require hips and at least one lower-limb keypoint per side.
    return bool(left_reliable >= 1 and right_reliable >= 1)


def _resolve_event_timestamp_ms(event, fps):
    ts_value = event.get("timestamp_ms")
    try:
        ts_value = float(ts_value)
    except (TypeError, ValueError):
        ts_value = np.nan
    if np.isfinite(ts_value) and ts_value >= 0.0:
        return int(round(ts_value))
    frame_value = event.get("frame", -1)
    try:
        frame_value = int(frame_value)
    except (TypeError, ValueError):
        frame_value = -1
    if fps and fps > 0 and frame_value >= 0:
        return int(round((float(frame_value) * 1000.0) / float(fps)))
    return -1


def advance_fixed_lag_confirmed_events(
    detected_jump_events,
    confirmed_events,
    confirmed_cursor,
    current_display_ts_ms,
    fixed_lag_ms,
    fps,
):
    cursor = max(0, int(confirmed_cursor))
    lag_ms = max(0, int(fixed_lag_ms))
    cutoff_ts_ms = int(current_display_ts_ms) - lag_ms
    while cursor < len(detected_jump_events):
        event = detected_jump_events[cursor]
        event_ts_ms = _resolve_event_timestamp_ms(event, fps)
        if event_ts_ms < 0 or event_ts_ms > cutoff_ts_ms:
            break
        confirmed_event = dict(event)
        confirmed_event["count"] = len(confirmed_events) + 1
        confirmed_events.append(confirmed_event)
        cursor += 1
    return int(cursor)


def _sanitize_path_fragment(text):
    if text is None:
        return ""
    cleaned = "".join(
        ch if (ch.isascii() and (ch.isalnum() or ch in {"-", "_", "."})) else "_"
        for ch in str(text).strip()
    )
    return cleaned.strip("._-")


def _write_detected_events_csv(csv_path, detected_events):
    rows = []
    for item in detected_events:
        left_point = item.get("left_point", (np.nan, np.nan))
        right_point = item.get("right_point", (np.nan, np.nan))
        rows.append(
            {
                "count": int(item.get("count", 0)),
                "frame": int(item.get("frame", -1)),
                "timestamp_ms": int(item.get("timestamp_ms", -1)),
                "strength_ratio": float(item.get("strength_ratio", np.nan)),
                "rope_ratio": float(item.get("rope_ratio", np.nan)),
                "rope_dual_ratio": float(item.get("rope_dual_ratio", np.nan)),
                "foot_prominence": float(item.get("foot_prominence", np.nan)),
                "left_foot_prominence": float(item.get("left_foot_prominence", np.nan)),
                "right_foot_prominence": float(item.get("right_foot_prominence", np.nan)),
                "both_feet_lift_ok": int(bool(item.get("both_feet_lift_ok", False))),
                "feet_symmetry_ratio": float(item.get("feet_symmetry_ratio", np.nan)),
                "feet_symmetry_ok": int(bool(item.get("feet_symmetry_ok", False))),
                "foot_sync_corr": float(item.get("foot_sync_corr", np.nan)),
                "foot_sync_ok": int(bool(item.get("foot_sync_ok", False))),
                "inplace_ok": int(bool(item.get("inplace_ok", False))),
                "left_x": left_point[0],
                "left_y": left_point[1],
                "right_x": right_point[0],
                "right_y": right_point[1],
            }
        )
    pd.DataFrame(rows).to_csv(csv_path, index=False)


def _split_events_by_gap(events, gap_ms):
    if not events:
        return []
    split_gap_ms = max(0, int(gap_ms))
    segments = []
    current_segment = [events[0]]
    for event in events[1:]:
        prev_ts = int(current_segment[-1].get("timestamp_ms", -1))
        curr_ts = int(event.get("timestamp_ms", -1))
        if (
            split_gap_ms > 0
            and prev_ts >= 0
            and curr_ts >= 0
            and (curr_ts - prev_ts) > split_gap_ms
        ):
            segments.append(current_segment)
            current_segment = [event]
        else:
            current_segment.append(event)
    if current_segment:
        segments.append(current_segment)
    return segments


def _compute_event_cadence_cv(events):
    if len(events) < 3:
        return 0.0
    timestamps = np.asarray([int(item.get("timestamp_ms", -1)) for item in events], dtype=np.float64)
    timestamps = timestamps[np.isfinite(timestamps)]
    if timestamps.size < 3:
        return 0.0
    gaps = np.diff(timestamps)
    gaps = gaps[gaps > 0.0]
    if gaps.size < 2:
        return 0.0
    mean_gap = float(np.mean(gaps))
    if mean_gap <= 0.0:
        return 0.0
    std_gap = float(np.std(gaps))
    return float(std_gap / mean_gap)


def _interpolate_point2d(point_a, point_b, alpha):
    if not isinstance(point_a, (tuple, list)) or not isinstance(point_b, (tuple, list)):
        return point_a
    if len(point_a) < 2 or len(point_b) < 2:
        return point_a
    ax, ay = point_a[0], point_a[1]
    bx, by = point_b[0], point_b[1]
    try:
        ax = float(ax)
        ay = float(ay)
        bx = float(bx)
        by = float(by)
    except (TypeError, ValueError):
        return point_a
    if not (np.isfinite(ax) and np.isfinite(ay) and np.isfinite(bx) and np.isfinite(by)):
        return point_a
    return (
        float(ax + ((bx - ax) * alpha)),
        float(ay + ((by - ay) * alpha)),
    )


def _interpolate_scalar(value_a, value_b, alpha, fallback):
    try:
        va = float(value_a)
        vb = float(value_b)
    except (TypeError, ValueError):
        return fallback
    if not (np.isfinite(va) and np.isfinite(vb)):
        return fallback
    return float(va + ((vb - va) * alpha))


def _build_interpolated_event(prev_event, next_event, alpha):
    new_event = dict(prev_event)
    prev_frame = int(prev_event.get("frame", -1))
    next_frame = int(next_event.get("frame", prev_frame))
    prev_ts = int(prev_event.get("timestamp_ms", -1))
    next_ts = int(next_event.get("timestamp_ms", prev_ts))
    new_event["frame"] = int(round(prev_frame + ((next_frame - prev_frame) * alpha)))
    new_event["timestamp_ms"] = int(round(prev_ts + ((next_ts - prev_ts) * alpha)))
    for scalar_key in (
        "strength_ratio",
        "rope_ratio",
        "rope_dual_ratio",
        "foot_prominence",
        "left_foot_prominence",
        "right_foot_prominence",
        "feet_symmetry_ratio",
        "foot_sync_corr",
    ):
        new_event[scalar_key] = _interpolate_scalar(
            prev_event.get(scalar_key),
            next_event.get(scalar_key),
            alpha,
            prev_event.get(scalar_key, np.nan),
        )
    for bool_key in ("both_feet_lift_ok", "feet_symmetry_ok", "foot_sync_ok", "inplace_ok"):
        if alpha < 0.5:
            new_event[bool_key] = bool(prev_event.get(bool_key, False))
        else:
            new_event[bool_key] = bool(next_event.get(bool_key, False))
    new_event["left_point"] = _interpolate_point2d(
        prev_event.get("left_point"),
        next_event.get("left_point"),
        alpha,
    )
    new_event["right_point"] = _interpolate_point2d(
        prev_event.get("right_point"),
        next_event.get("right_point"),
        alpha,
    )
    new_event["count"] = 0
    new_event["gap_filled"] = True
    return new_event


def _fill_session_gaps(
    segment_events,
    min_ratio,
    max_ratio,
    max_insert_per_gap,
    max_total_insert,
    skip_prefix_pairs=0,
):
    if len(segment_events) < 3:
        return list(segment_events), 0
    timestamps = np.asarray(
        [int(item.get("timestamp_ms", -1)) for item in segment_events],
        dtype=np.float64,
    )
    gaps = np.diff(timestamps)
    valid_gaps = gaps[gaps > 0.0]
    if valid_gaps.size == 0:
        return list(segment_events), 0
    median_gap = float(np.median(valid_gaps))
    if not np.isfinite(median_gap) or median_gap <= 0.0:
        return list(segment_events), 0
    min_gap_ratio = float(max(1.0, min_ratio))
    max_gap_ratio = float(max(min_gap_ratio, max_ratio))
    max_insert_gap = max(1, int(max_insert_per_gap))
    max_insert_total = max(1, int(max_total_insert))

    filled_events = []
    inserted_total = 0
    for idx in range(len(segment_events) - 1):
        prev_event = segment_events[idx]
        next_event = segment_events[idx + 1]
        filled_events.append(prev_event)
        if idx < max(0, int(skip_prefix_pairs)):
            continue
        if inserted_total >= max_insert_total:
            continue
        gap_ms = int(next_event.get("timestamp_ms", -1)) - int(prev_event.get("timestamp_ms", -1))
        if gap_ms <= 0:
            continue
        gap_ratio = float(gap_ms) / median_gap
        if gap_ratio < min_gap_ratio or gap_ratio > max_gap_ratio:
            continue
        insert_count = int(round(gap_ratio)) - 1
        insert_count = max(1, min(max_insert_gap, insert_count))
        insert_count = min(insert_count, max_insert_total - inserted_total)
        for insert_idx in range(insert_count):
            alpha = float(insert_idx + 1) / float(insert_count + 1)
            filled_events.append(
                _build_interpolated_event(prev_event, next_event, alpha)
            )
            inserted_total += 1
            if inserted_total >= max_insert_total:
                break
    filled_events.append(segment_events[-1])
    return filled_events, inserted_total


def _fill_session_head(
    segment_events,
    insert_count,
):
    if len(segment_events) < 2:
        return list(segment_events), 0
    count = max(0, int(insert_count))
    if count <= 0:
        return list(segment_events), 0
    first_event = segment_events[0]
    second_event = segment_events[1]
    head_events = []
    for step in range(count, 0, -1):
        extrapolated = _build_interpolated_event(
            first_event,
            second_event,
            alpha=-float(step),
        )
        frame_value = int(extrapolated.get("frame", -1))
        ts_value = int(extrapolated.get("timestamp_ms", -1))
        if frame_value < 0 or ts_value < 0:
            continue
        head_events.append(extrapolated)
    return head_events + list(segment_events), len(head_events)


def _is_weak_non_bilateral_event(event):
    strength = float(event.get("strength_ratio", np.nan))
    foot_prominence = float(event.get("foot_prominence", np.nan))
    both_feet_lift_ok = bool(event.get("both_feet_lift_ok", False))
    feet_symmetry_ok = bool(event.get("feet_symmetry_ok", False))
    weak_strength_max = float(STRICT_WEAK_NON_BILATERAL_MAX_STRENGTH)
    return bool(
        np.isfinite(strength)
        and strength <= weak_strength_max
        and np.isfinite(foot_prominence)
        and foot_prominence < 0.0
        and (not both_feet_lift_ok)
        and (not feet_symmetry_ok)
    )


def _is_all_in_one_strong_positive_event(event):
    strength = float(event.get("strength_ratio", np.nan))
    foot_prominence = float(event.get("foot_prominence", np.nan))
    return bool(
        np.isfinite(strength)
        and strength >= float(ALL_IN_ONE_STRONG_ENTRY_MIN_STRENGTH_RATIO)
        and np.isfinite(foot_prominence)
        and foot_prominence >= float(ALL_IN_ONE_STRONG_ENTRY_MIN_FOOT_PROMINENCE)
    )


def _has_bilateral_or_symmetry(event):
    return bool(event.get("both_feet_lift_ok", False) or event.get("feet_symmetry_ok", False))


def _find_all_in_one_segment_start(segment_events):
    if not segment_events:
        return None, ""
    strong_min_events = max(1, int(ALL_IN_ONE_STRONG_ENTRY_MIN_EVENTS))
    stable_min_events = max(strong_min_events, int(ALL_IN_ONE_STABLE_ENTRY_MIN_EVENTS))
    strong_max_cv = float(max(0.0, float(ALL_IN_ONE_STRONG_ENTRY_MAX_CADENCE_CV)))
    stable_max_cv = float(max(0.0, float(ALL_IN_ONE_STABLE_ENTRY_MAX_CADENCE_CV)))
    strong_min_events_count = max(1, int(ALL_IN_ONE_STRONG_ENTRY_MIN_STRONG_EVENTS))
    stable_min_events_count = max(1, int(ALL_IN_ONE_STABLE_ENTRY_MIN_STRONG_EVENTS))
    start_min_fp = float(max(0.0, float(ALL_IN_ONE_ENTRY_START_MIN_FOOT_PROMINENCE)))
    strong_min_rope = float(max(0.0, float(ALL_IN_ONE_STRONG_ENTRY_MIN_ROPE_RATIO)))
    strong_min_dual = float(max(0.0, float(ALL_IN_ONE_STRONG_ENTRY_MIN_DUAL_RATIO)))
    stable_min_rope = float(max(0.0, float(ALL_IN_ONE_STABLE_ENTRY_MIN_ROPE_RATIO)))
    stable_min_abs_fp = float(max(0.0, float(ALL_IN_ONE_STABLE_ENTRY_MIN_ABS_FOOT_PROMINENCE)))

    for end_idx in range(len(segment_events)):
        prefix = segment_events[: end_idx + 1]
        prefix_len = len(prefix)
        prefix_cv = _compute_event_cadence_cv(prefix)
        rope_values = [
            float(event.get("rope_ratio", np.nan))
            for event in prefix
            if np.isfinite(float(event.get("rope_ratio", np.nan)))
        ]
        dual_values = [
            float(event.get("rope_dual_ratio", np.nan))
            for event in prefix
            if np.isfinite(float(event.get("rope_dual_ratio", np.nan)))
        ]
        abs_fp_values = [
            abs(float(event.get("foot_prominence", np.nan)))
            for event in prefix
            if np.isfinite(float(event.get("foot_prominence", np.nan)))
        ]
        strong_positive_indices = [
            idx for idx, event in enumerate(prefix) if _is_all_in_one_strong_positive_event(event)
        ]
        bilateral_indices = [idx for idx, event in enumerate(prefix) if _has_bilateral_or_symmetry(event)]
        median_rope = float(np.median(rope_values)) if rope_values else 0.0
        median_dual = float(np.median(dual_values)) if dual_values else 0.0
        median_abs_fp = float(np.median(abs_fp_values)) if abs_fp_values else 0.0

        if (
            prefix_len >= strong_min_events
            and prefix_cv <= strong_max_cv
            and median_rope >= strong_min_rope
            and median_dual >= strong_min_dual
            and len(strong_positive_indices) >= strong_min_events_count
            and bilateral_indices
        ):
            for idx, event in enumerate(prefix):
                event_fp = float(event.get("foot_prominence", np.nan))
                if (
                    _has_bilateral_or_symmetry(event)
                    or (np.isfinite(event_fp) and event_fp >= start_min_fp)
                ):
                    return idx, "strong_high_dual"
            return strong_positive_indices[0], "strong_high_dual_fallback"

        if (
            prefix_len >= strong_min_events
            and prefix_cv <= strong_max_cv
            and median_rope >= strong_min_rope
            and len(strong_positive_indices) >= strong_min_events_count
            and bilateral_indices
        ):
            low_dual_start_idx = max(0, strong_positive_indices[0] - 1)
            for idx in range(low_dual_start_idx, strong_positive_indices[0] + 1):
                event = prefix[idx]
                event_fp = abs(float(event.get("foot_prominence", np.nan)))
                if _has_bilateral_or_symmetry(event) or (
                    np.isfinite(event_fp) and event_fp >= start_min_fp
                ):
                    return idx, "strong_low_dual"
            return strong_positive_indices[0], "strong_low_dual_strong_start"

        if (
            prefix_len >= stable_min_events
            and prefix_cv <= stable_max_cv
            and median_rope >= stable_min_rope
            and median_abs_fp >= stable_min_abs_fp
            and len(strong_positive_indices) >= stable_min_events_count
            and bilateral_indices
        ):
            return strong_positive_indices[0], "stable"
    return None, ""


def _backfill_all_in_one_entry(segment_events, keep_start_idx):
    kept_segment = [dict(event) for event in segment_events[keep_start_idx:]]
    if keep_start_idx <= 0 or len(kept_segment) < 3:
        return kept_segment, 0

    min_backfill = max(1, int(ALL_IN_ONE_ENTRY_BACKFILL_MIN_EVENTS))
    max_backfill = max(min_backfill, int(ALL_IN_ONE_ENTRY_BACKFILL_MAX_EVENTS))
    if keep_start_idx < min_backfill:
        return kept_segment, 0

    kept_timestamps = np.asarray(
        [int(item.get("timestamp_ms", -1)) for item in kept_segment[:5]],
        dtype=np.float64,
    )
    kept_gaps = np.diff(kept_timestamps)
    kept_gaps = kept_gaps[kept_gaps > 0.0]
    if kept_gaps.size == 0:
        return kept_segment, 0
    reference_gap = float(np.median(kept_gaps))
    if not np.isfinite(reference_gap) or reference_gap <= 0.0:
        return kept_segment, 0

    min_gap_ratio = float(max(0.0, float(ALL_IN_ONE_ENTRY_BACKFILL_MIN_GAP_RATIO)))
    max_gap_ratio = float(
        max(min_gap_ratio, float(ALL_IN_ONE_ENTRY_BACKFILL_MAX_GAP_RATIO))
    )
    min_abs_fp = float(
        max(0.0, float(ALL_IN_ONE_ENTRY_BACKFILL_MIN_ABS_FOOT_PROMINENCE))
    )

    backfilled_events = []
    strong_or_bilateral_count = 0
    next_event = dict(kept_segment[0])
    for source_idx in range(keep_start_idx - 1, -1, -1):
        if len(backfilled_events) >= max_backfill:
            break
        candidate = dict(segment_events[source_idx])
        candidate_ts = int(candidate.get("timestamp_ms", -1))
        next_ts = int(next_event.get("timestamp_ms", -1))
        gap_ms = next_ts - candidate_ts
        if gap_ms <= 0:
            break
        gap_ratio = float(gap_ms) / reference_gap
        if gap_ratio < min_gap_ratio or gap_ratio > max_gap_ratio:
            break
        if _is_weak_non_bilateral_event(candidate):
            break
        abs_fp = abs(float(candidate.get("foot_prominence", np.nan)))
        strong_or_bilateral = bool(
            _is_all_in_one_strong_positive_event(candidate)
            or _has_bilateral_or_symmetry(candidate)
        )
        if (not strong_or_bilateral) and (
            (not np.isfinite(abs_fp)) or abs_fp < min_abs_fp
        ):
            break
        backfilled_events.append(candidate)
        if strong_or_bilateral:
            strong_or_bilateral_count += 1
        next_event = candidate

    if len(backfilled_events) < min_backfill or strong_or_bilateral_count <= 0:
        return kept_segment, 0
    backfilled_events.reverse()
    return backfilled_events + kept_segment, len(backfilled_events)


def _head_fill_all_in_one_entry(
    kept_segment,
    keep_start_idx,
    entry_backfill_count,
    start_reason,
    segment_idx,
    segments,
):
    if entry_backfill_count > 0:
        return list(kept_segment), 0
    if len(kept_segment) < 4:
        return list(kept_segment), 0
    early_cv = _compute_event_cadence_cv(kept_segment[:5])
    if early_cv > min(0.10, float(ALL_IN_ONE_STABLE_ENTRY_MAX_CADENCE_CV)):
        return list(kept_segment), 0
    insert_count = max(0, min(2, int(keep_start_idx) - 2))
    if (
        insert_count <= 0
        and int(keep_start_idx) == 0
        and int(segment_idx) > 0
        and str(start_reason) == "strong_high_dual"
    ):
        previous_segment = list(segments[segment_idx - 1])
        previous_keep_start_idx, _ = _find_all_in_one_segment_start(previous_segment)
        previous_segment_gap_ms = int(kept_segment[0].get("timestamp_ms", -1)) - int(
            previous_segment[-1].get("timestamp_ms", -1)
        )
        if (
            previous_keep_start_idx is None
            and len(previous_segment) <= 3
            and previous_segment_gap_ms >= 3000
        ):
            insert_count = 2
    if insert_count <= 0:
        return list(kept_segment), 0
    filled_segment, inserted_count = _fill_session_head(
        segment_events=kept_segment,
        insert_count=insert_count,
    )
    return filled_segment, int(inserted_count)


def _select_all_in_one_events(source_events, strict_guard_mode):
    selected_events = []
    pruned_weak_non_bilateral = 0
    inserted_by_entry_backfill = 0
    inserted_by_head_fill = 0
    inserted_by_gap_fill = 0
    candidate_events = []
    for event in source_events:
        event_copy = dict(event)
        if strict_guard_mode and bool(STRICT_WEAK_NON_BILATERAL_PRUNE_ENABLED):
            if _is_weak_non_bilateral_event(event_copy):
                pruned_weak_non_bilateral += 1
                continue
        candidate_events.append(event_copy)

    segments = _split_events_by_gap(
        candidate_events,
        gap_ms=int(round(max(0.0, float(STRICT_SESSION_SPLIT_GAP_SECONDS)) * 1000.0)),
    )
    confirmed_segments = 0
    dropped_segments = 0
    segment_debug_rows = []
    for segment_idx, segment in enumerate(segments):
        keep_start_idx, reason = _find_all_in_one_segment_start(segment)
        segment_cv = _compute_event_cadence_cv(segment)
        abs_fp_values = [
            abs(float(item.get("foot_prominence", np.nan)))
            for item in segment
            if np.isfinite(float(item.get("foot_prominence", np.nan)))
        ]
        segment_abs_fp_median = float(np.median(abs_fp_values)) if abs_fp_values else 0.0
        kept = keep_start_idx is not None
        segment_debug_rows.append(
            (
                segment_idx,
                int(len(segment)),
                int(segment[0].get("timestamp_ms", -1)),
                int(segment[-1].get("timestamp_ms", -1)),
                float(segment_cv),
                float(segment_abs_fp_median),
                bool(kept),
                reason or "rejected",
            )
        )
        if keep_start_idx is None:
            dropped_segments += 1
            continue
        kept_segment, entry_backfill_count = _backfill_all_in_one_entry(
            segment_events=segment,
            keep_start_idx=keep_start_idx,
        )
        inserted_by_entry_backfill += int(entry_backfill_count)
        kept_segment, head_fill_count = _head_fill_all_in_one_entry(
            kept_segment=kept_segment,
            keep_start_idx=keep_start_idx,
            entry_backfill_count=entry_backfill_count,
            start_reason=reason,
            segment_idx=segment_idx,
            segments=segments,
        )
        inserted_by_head_fill += int(head_fill_count)
        gap_fill_skip_prefix_pairs = 1 if keep_start_idx <= 0 and entry_backfill_count <= 0 else 0
        if (
            strict_guard_mode
            and bool(STRICT_SESSION_GAP_FILL_ENABLED)
            and len(kept_segment) >= max(3, int(STRICT_SESSION_GAP_FILL_MIN_EVENTS))
            and _compute_event_cadence_cv(kept_segment)
            <= float(STRICT_SESSION_GAP_FILL_MAX_CADENCE_CV)
        ):
            kept_segment, inserted_count = _fill_session_gaps(
                segment_events=kept_segment,
                min_ratio=float(ALL_IN_ONE_GAP_FILL_MIN_RATIO),
                max_ratio=float(STRICT_SESSION_GAP_FILL_MAX_RATIO),
                max_insert_per_gap=int(STRICT_SESSION_GAP_FILL_MAX_INSERT_PER_GAP),
                max_total_insert=int(STRICT_SESSION_GAP_FILL_MAX_TOTAL_INSERT),
                skip_prefix_pairs=gap_fill_skip_prefix_pairs,
            )
            inserted_by_gap_fill += int(inserted_count)
        selected_events.extend(kept_segment)
        confirmed_segments += 1

    for idx, event in enumerate(selected_events, start=1):
        event["count"] = idx

    return selected_events, {
        "pruned_weak_non_bilateral": int(pruned_weak_non_bilateral),
        "inserted_by_entry_backfill": int(inserted_by_entry_backfill),
        "inserted_by_head_fill": int(inserted_by_head_fill),
        "inserted_by_gap_fill": int(inserted_by_gap_fill),
        "confirmed_segments": int(confirmed_segments),
        "dropped_segments": int(dropped_segments),
        "segment_debug_rows": segment_debug_rows,
    }


def _postprocess_confirmed_events(
    confirmed_events,
    strict_guard_mode,
):
    selected_events = [dict(event) for event in confirmed_events]
    stats = {
        "pruned_weak_non_bilateral": 0,
        "pruned_by_session_filter": 0,
        "pruned_by_session_strength": 0,
        "pruned_by_run_min_events": 0,
        "inserted_by_gap_fill": 0,
        "inserted_by_head_fill": 0,
        "session_split_gap_ms": int(
            round(max(0.0, float(STRICT_SESSION_SPLIT_GAP_SECONDS)) * 1000.0)
        ),
        "session_min_events": max(1, int(SESSION_MIN_EVENTS)),
        "session_max_cadence_cv": float(max(0.0, float(STRICT_SESSION_MAX_CADENCE_CV))),
        "kept_segments": 0,
        "dropped_segments": 0,
        "bridged_short_segments": 0,
        "segment_debug_rows": [],
    }

    if strict_guard_mode and bool(STRICT_WEAK_NON_BILATERAL_PRUNE_ENABLED):
        filtered_events = []
        weak_strength_max = float(STRICT_WEAK_NON_BILATERAL_MAX_STRENGTH)
        for event in selected_events:
            strength = float(event.get("strength_ratio", np.nan))
            foot_prominence = float(event.get("foot_prominence", np.nan))
            both_feet_lift_ok = bool(event.get("both_feet_lift_ok", False))
            feet_symmetry_ok = bool(event.get("feet_symmetry_ok", False))
            weak_no_bilateral = bool(
                np.isfinite(strength)
                and strength <= weak_strength_max
                and np.isfinite(foot_prominence)
                and foot_prominence < 0.0
                and (not both_feet_lift_ok)
                and (not feet_symmetry_ok)
            )
            if weak_no_bilateral:
                stats["pruned_weak_non_bilateral"] += 1
                continue
            filtered_events.append(event)
        selected_events = filtered_events

    session_split_gap_ms = int(stats["session_split_gap_ms"])
    if strict_guard_mode and bool(STRICT_SESSION_FILTER_ENABLED) and selected_events:
        session_min_events = int(stats["session_min_events"])
        session_max_cadence_cv = float(stats["session_max_cadence_cv"])
        session_min_abs_foot_prominence = float(
            max(0.0, float(STRICT_SESSION_MIN_ABS_FOOT_PROMINENCE))
        )
        segments = _split_events_by_gap(selected_events, gap_ms=session_split_gap_ms)
        session_short_bridge_enabled = bool(STRICT_SESSION_SHORT_BRIDGE_ENABLED)
        short_bridge_min_events = max(1, int(STRICT_SESSION_SHORT_BRIDGE_MIN_EVENTS))
        short_bridge_max_events = max(
            short_bridge_min_events,
            int(STRICT_SESSION_SHORT_BRIDGE_MAX_EVENTS),
        )
        short_bridge_max_gap_ms = int(
            round(max(0.0, float(STRICT_SESSION_SHORT_BRIDGE_MAX_GAP_SECONDS)) * 1000.0)
        )
        short_bridge_max_cv = float(
            session_max_cadence_cv
            * max(1.0, float(STRICT_SESSION_SHORT_BRIDGE_CV_SCALE))
        )

        def _is_bridgeable_short_segment(segment_idx):
            if not session_short_bridge_enabled:
                return False
            segment = segments[segment_idx]
            segment_len = int(len(segment))
            if segment_len < short_bridge_min_events or segment_len > short_bridge_max_events:
                return False
            neighbor_indices = []
            if segment_idx > 0:
                neighbor_indices.append(segment_idx - 1)
            if segment_idx + 1 < len(segments):
                neighbor_indices.append(segment_idx + 1)
            for neighbor_idx in neighbor_indices:
                anchor = segments[neighbor_idx]
                if int(len(anchor)) < session_min_events:
                    continue
                anchor_cv = _compute_event_cadence_cv(anchor)
                if anchor_cv > session_max_cadence_cv:
                    continue
                if neighbor_idx < segment_idx:
                    gap_ms = int(segment[0].get("timestamp_ms", -1)) - int(
                        anchor[-1].get("timestamp_ms", -1)
                    )
                    anchor_slice = anchor[-min(session_min_events, len(anchor)):]
                    merged_for_cv = list(anchor_slice) + list(segment)
                else:
                    gap_ms = int(anchor[0].get("timestamp_ms", -1)) - int(
                        segment[-1].get("timestamp_ms", -1)
                    )
                    anchor_slice = anchor[:min(session_min_events, len(anchor))]
                    merged_for_cv = list(segment) + list(anchor_slice)
                if gap_ms <= 0 or gap_ms > short_bridge_max_gap_ms:
                    continue
                merged_cv = _compute_event_cadence_cv(merged_for_cv)
                if merged_cv <= short_bridge_max_cv:
                    return True
            return False

        kept_events = []
        kept_segments = 0
        bridged_short_segments = 0
        segment_debug_rows = []
        for segment_idx, segment in enumerate(segments):
            segment_len = int(len(segment))
            keep_segment = True
            segment_cv = _compute_event_cadence_cv(segment)
            segment_abs_fp_values = [
                abs(float(item.get("foot_prominence", np.nan)))
                for item in segment
                if np.isfinite(float(item.get("foot_prominence", np.nan)))
            ]
            segment_abs_fp_median = (
                float(np.median(segment_abs_fp_values))
                if segment_abs_fp_values
                else 0.0
            )
            drop_reason = ""
            if segment_len < session_min_events:
                keep_segment = _is_bridgeable_short_segment(segment_idx)
                if keep_segment:
                    bridged_short_segments += 1
                else:
                    drop_reason = "min_events"
            if keep_segment and segment_len >= session_min_events:
                if segment_cv > session_max_cadence_cv:
                    keep_segment = False
                    drop_reason = "high_cadence_cv"
            if keep_segment and segment_len >= session_min_events:
                if segment_abs_fp_median < session_min_abs_foot_prominence:
                    keep_segment = False
                    drop_reason = "low_abs_foot_prominence"
            if keep_segment and segment_len < session_min_events:
                drop_reason = "bridged_short"
            segment_start_ts = int(segment[0].get("timestamp_ms", -1))
            segment_end_ts = int(segment[-1].get("timestamp_ms", -1))
            segment_debug_rows.append(
                (
                    segment_idx,
                    segment_len,
                    segment_start_ts,
                    segment_end_ts,
                    float(segment_cv),
                    float(segment_abs_fp_median),
                    bool(keep_segment),
                    drop_reason or "kept",
                )
            )
            if not keep_segment:
                stats["pruned_by_session_filter"] += segment_len
                continue
            kept_events.extend(segment)
            kept_segments += 1
        selected_events = kept_events
        stats["kept_segments"] = int(kept_segments)
        stats["dropped_segments"] = max(0, len(segments) - kept_segments)
        stats["bridged_short_segments"] = int(bridged_short_segments)
        stats["segment_debug_rows"] = segment_debug_rows

    if strict_guard_mode and bool(STRICT_SESSION_GAP_FILL_ENABLED) and selected_events:
        gap_fill_segments = _split_events_by_gap(selected_events, gap_ms=session_split_gap_ms)
        filled_events = []
        for segment in gap_fill_segments:
            if len(segment) < max(1, int(STRICT_SESSION_GAP_FILL_MIN_EVENTS)):
                filled_events.extend(segment)
                continue
            segment_cv = _compute_event_cadence_cv(segment)
            if segment_cv > float(STRICT_SESSION_GAP_FILL_MAX_CADENCE_CV):
                filled_events.extend(segment)
                continue
            if (
                bool(STRICT_SESSION_HEAD_FILL_ENABLED)
                and len(segment) >= max(2, int(STRICT_SESSION_HEAD_FILL_MIN_EVENTS))
                and segment_cv <= float(STRICT_SESSION_HEAD_FILL_MAX_CADENCE_CV)
            ):
                segment, head_inserted_count = _fill_session_head(
                    segment_events=segment,
                    insert_count=int(STRICT_SESSION_HEAD_FILL_INSERT_COUNT),
                )
                stats["inserted_by_head_fill"] += int(head_inserted_count)
            filled_segment, inserted_count = _fill_session_gaps(
                segment_events=segment,
                min_ratio=float(STRICT_SESSION_GAP_FILL_MIN_RATIO),
                max_ratio=float(STRICT_SESSION_GAP_FILL_MAX_RATIO),
                max_insert_per_gap=int(STRICT_SESSION_GAP_FILL_MAX_INSERT_PER_GAP),
                max_total_insert=int(STRICT_SESSION_GAP_FILL_MAX_TOTAL_INSERT),
            )
            stats["inserted_by_gap_fill"] += int(inserted_count)
            filled_events.extend(filled_segment)
        if filled_events and (
            stats["inserted_by_gap_fill"] > 0 or stats["inserted_by_head_fill"] > 0
        ):
            selected_events = sorted(
                filled_events,
                key=lambda event: (
                    int(event.get("frame", -1)),
                    int(event.get("timestamp_ms", -1)),
                ),
            )

    if strict_guard_mode and selected_events:
        strength_values = [
            float(event.get("strength_ratio", np.nan))
            for event in selected_events
            if np.isfinite(float(event.get("strength_ratio", np.nan)))
        ]
        if strength_values:
            session_median_strength = float(np.median(strength_values))
            if session_median_strength < float(STRICT_SESSION_MIN_MEDIAN_STRENGTH_RATIO):
                stats["pruned_by_session_strength"] = int(len(selected_events))
                selected_events = []

    if (
        strict_guard_mode
        and int(SESSION_MIN_EVENTS) > 1
        and len(selected_events) < int(SESSION_MIN_EVENTS)
    ):
        stats["pruned_by_run_min_events"] = int(len(selected_events))
        selected_events = []

    for idx, event in enumerate(selected_events, start=1):
        event["count"] = idx
    return selected_events, stats


def run_pipeline(
    target_stem=None,
    target_video_path=None,
    camera_index=0,
    use_realtime=False,
    require_label=False,
    write_summary=True,
    realtime_fps_log_interval_s=1.0,
    realtime_demo_log=False,
    realtime_demo_log_dir=None,
    realtime_demo_tag="",
    realtime_demo_save_raw=False,
):
    realtime_fps_log_interval_s = max(0.0, float(realtime_fps_log_interval_s))
    realtime_demo_log = bool(realtime_demo_log)
    realtime_demo_save_raw = bool(realtime_demo_save_raw)
    use_realtime = bool(use_realtime)
    require_label = bool(require_label)
    realtime_demo_tag = _sanitize_path_fragment(realtime_demo_tag)
    if realtime_demo_log_dir:
        realtime_demo_log_dir = os.path.abspath(str(realtime_demo_log_dir))
    else:
        realtime_demo_log_dir = ""
    summary_csv_name = get_summary_csv_name(use_realtime=use_realtime)

    jobs = build_runtime_jobs(
        target_stem=target_stem,
        target_video_path=target_video_path,
        camera_index=camera_index,
        use_realtime=use_realtime,
        require_label=require_label,
    )
    if not jobs:
        if use_realtime:
            print(f"[WARN] No realtime job was created (camera index={camera_index})")
        elif require_label:
            print(f"[WARN] No labeled video jobs found in {INPUT_VIDEO_DIR} and {INPUT_LABEL_DIR}")
        else:
            print(f"[WARN] No video jobs found in {INPUT_VIDEO_DIR}")

    overall_summary_rows = []
    for job in jobs:
        stem = job["stem"]
        file_path = job["video_path"]
        capture_source = job.get("capture_source", file_path)
        is_realtime = bool(job.get("is_realtime", False))
        all_in_one_enabled = bool(ALL_IN_ONE_ENGINE_ENABLED)
        use_live_fixed_lag_overlay = bool(LIVE_OVERLAY_FIXED_LAG_COUNT)
        # Realtime-only architecture: always finalize with fixed-lag confirmed events.
        use_fixed_lag_final = True
        fixed_lag_ms = int(round(max(0.0, float(FIXED_LAG_SECONDS)) * 1000.0))
        fixed_lag_finalize_wait_ms = int(
            round(max(0.0, float(FIXED_LAG_FINALIZE_WAIT_SECONDS)) * 1000.0)
        )
        strict_guard_mode = bool(STRICT_GUARDS)
        runtime_min_strength_ratio = float(MIN_STRENGTH_RATIO)
        runtime_enter_min_events = int(ACTIVE_ENTER_MIN_EVENTS)
        runtime_startup_lockout_seconds = float(STARTUP_LOCKOUT_SECONDS)
        runtime_require_dual_rope = bool(strict_guard_mode and STRICT_REQUIRE_DUAL_ROPE)
        runtime_lower_body_vis_min = float(STRICT_LOWER_BODY_VIS_MIN)
        runtime_foot_lift_min_prominence = float(STRICT_FOOT_LIFT_MIN_PROMINENCE)
        runtime_both_feet_min_prominence = float(STRICT_BOTH_FEET_MIN_PROMINENCE)
        runtime_feet_symmetry_min_ratio = float(STRICT_FEET_SYMMETRY_MIN_RATIO)
        runtime_foot_sync_min_corr = float(STRICT_FOOT_SYNC_MIN_CORR)
        runtime_inplace_max_center_drift = float(STRICT_INPLACE_MAX_CENTER_DRIFT)
        runtime_require_advanced_motion = bool(STRICT_REQUIRE_ADVANCED_MOTION)
        runtime_motion_advanced_min_checks = int(STRICT_MOTION_ADVANCED_MIN_CHECKS)
        runtime_strict_active_override_enabled = bool(
            float(STRICT_OVERRIDE_ACTIVE_MIN_ROPE_RATIO) > 0.0
            or float(STRICT_OVERRIDE_ACTIVE_MIN_DUAL_RATIO) > 0.0
        )
        runtime_strict_entry_override_enabled = bool(
            float(STRICT_OVERRIDE_ENTRY_MIN_ROPE_RATIO) > 0.0
            or float(STRICT_OVERRIDE_ENTRY_MIN_DUAL_RATIO) > 0.0
        )
        if strict_guard_mode:
            runtime_min_strength_ratio = max(
                runtime_min_strength_ratio,
                float(STRICT_MIN_STRENGTH_RATIO),
            )
            runtime_enter_min_events = max(
                runtime_enter_min_events,
                int(STRICT_ENTER_MIN_EVENTS),
            )
            runtime_startup_lockout_seconds = max(
                runtime_startup_lockout_seconds,
                float(STRICT_STARTUP_LOCKOUT_SECONDS),
            )
        label_path = job.get("label_path")
    
        if label_path:
            label_events = load_label_events(label_path)
            if label_events:
                print(f"[INFO] Loaded {len(label_events)} label events from {label_path}")
            else:
                print(f"[WARN] No usable label events loaded from {label_path}")
        else:
            label_events = []
            print(f"[INFO] Running without labels: {stem}")
    
        if is_realtime:
            print(f"\n[PROCESS] realtime camera index={capture_source}")
        else:
            print(f"\n[PROCESS] {file_path}")
        if strict_guard_mode:
            print(
                "[INFO] strict_guards=true "
                f"lower_body_vis_min={runtime_lower_body_vis_min:.2f} "
                f"require_dual_rope={runtime_require_dual_rope} "
                f"min_strength_ratio={runtime_min_strength_ratio:.2f} "
                f"enter_min_events={runtime_enter_min_events} "
                f"startup_lockout_s={runtime_startup_lockout_seconds:.2f} "
                f"foot_lift_min={runtime_foot_lift_min_prominence:.4f} "
                f"both_feet_min={runtime_both_feet_min_prominence:.4f} "
                f"symmetry_min={runtime_feet_symmetry_min_ratio:.2f} "
                f"sync_min={runtime_foot_sync_min_corr:.2f} "
                f"inplace_max={runtime_inplace_max_center_drift:.3f} "
                f"require_adv_motion={runtime_require_advanced_motion} "
                f"advanced_min_checks={runtime_motion_advanced_min_checks} "
                f"active_motion_window={STRICT_ACTIVE_FOOT_MOTION_WINDOW} "
                f"active_motion_min_ratio={STRICT_ACTIVE_FOOT_MOTION_MIN_RATIO:.2f} "
                f"all_in_one_enabled={all_in_one_enabled} "
                f"live_overlay_fixed_lag={use_live_fixed_lag_overlay} "
                f"live_overlay_stream_post={bool(LIVE_OVERLAY_STREAM_POSTPROCESS)} "
                f"fixed_lag_s={FIXED_LAG_SECONDS:.2f} "
                f"fixed_lag_final={use_fixed_lag_final} "
                f"session_min_events={SESSION_MIN_EVENTS}"
            )
    
        job_output_dir = OUTPUT_DIR
        demo_session_dir = ""
        frame_log_csv_path = ""
        if realtime_demo_log:
            out_stamp = time.strftime("%Y%m%d_%H%M%S")
            demo_root_dir = realtime_demo_log_dir or os.path.join(OUTPUT_DIR, "realtime_demo_logs")
            tag_prefix = f"{realtime_demo_tag}_" if realtime_demo_tag else ""
            session_name = f"{tag_prefix}{stem}_{out_stamp}"
            demo_session_dir = os.path.join(demo_root_dir, session_name)
            job_output_dir = demo_session_dir
            frame_log_csv_path = os.path.join(demo_session_dir, "frame_log.csv")
            print(f"[LOG] demo session: {demo_session_dir}")

        os.makedirs(job_output_dir, exist_ok=True)
        if is_realtime:
            if demo_session_dir:
                out_name = "tracked.mp4"
            else:
                out_stamp = time.strftime("%Y%m%d_%H%M%S")
                out_name = f"{stem}_{out_stamp}_tracked.mp4"
        else:
            out_name = os.path.splitext(os.path.basename(file_path))[0] + "_tracked.mp4"
        out_path = os.path.join(job_output_dir, out_name)
        left_foot_x = []
        right_foot_x = []
        left_foot_y = []
        right_foot_y = []
    
        cap = cv2.VideoCapture(capture_source)
    
        if not cap.isOpened():
            print(f"[ERROR] Failed to open video stream: {capture_source}")
            continue
    
        knnfgbg = cv2.createBackgroundSubtractorKNN(
            history=KNN_HISTORY,
            dist2Threshold=KNN_DIST2_THRESHOLD,
            detectShadows=False,
        )
        rope_mask_kernel = None
        if ROPE_MASK_KERNEL_SIZE > 1:
            rope_mask_kernel = np.ones((ROPE_MASK_KERNEL_SIZE, ROPE_MASK_KERNEL_SIZE), dtype=np.uint8)
    
        num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
        if not out.isOpened():
            print(f"[ERROR] Failed to open VideoWriter: {out_path}")
            cap.release()
            continue
        raw_out = None
        if is_realtime and demo_session_dir and realtime_demo_save_raw:
            raw_out_path = os.path.join(demo_session_dir, "raw.mp4")
            raw_out = cv2.VideoWriter(raw_out_path, fourcc, fps, (width, height))
            if not raw_out.isOpened():
                print(f"[WARN] Failed to open raw VideoWriter: {raw_out_path}")
                raw_out.release()
                raw_out = None
            else:
                print(f"[LOG] raw video: {raw_out_path}")
        if frame_log_csv_path:
            print(f"[LOG] frame log: {frame_log_csv_path}")
    
        match_tolerance_ms, frame_based_tolerance_ms = get_match_tolerance_ms(fps)
        print(f"[INFO] fps={fps:.2f} size={width}x{height} frames={num_frames}")
        print(
            f"[INFO] match_tolerance_ms={match_tolerance_ms} "
            f"(base={LABEL_MATCH_TOLERANCE_MS} frame_based={frame_based_tolerance_ms} "
            f"from {LABEL_MATCH_TOLERANCE_FRAMES:.1f} frames)"
        )
        print(
            f"[INFO] display_time_advance_ms={DISPLAY_TIME_ADVANCE_MS} "
            f"overlay_refresh_enabled={ENABLE_OVERLAY_REFRESH}"
        )
        if use_live_fixed_lag_overlay or use_fixed_lag_final:
            print(
                f"[INFO] fixed_lag_ms={fixed_lag_ms} "
                f"finalize_wait_ms={fixed_lag_finalize_wait_ms} "
                f"live_fixed_lag_overlay={use_live_fixed_lag_overlay} "
                f"fixed_lag_final={use_fixed_lag_final}"
            )
        if (
            is_realtime
            and HEADLESS
            and REALTIME_MAX_SECONDS <= 0
            and REALTIME_MAX_FRAMES <= 0
        ):
            print(
                "[WARN] Realtime mode is running headless without stop condition. "
                "Set JR_REALTIME_MAX_SECONDS or JR_REALTIME_MAX_FRAMES."
            )
        processed_frames = 0
        last_frame_shape = None
        start_time = time.time()
        last_fps_log_time = start_time
        last_fps_log_frames = 0
        jump_counter = 0
        last_processed_minima_idx = -1
        hip_center_y = []
        foot_center_y = []
        foot_center_x = []
        label_progress_idx = 0
        detected_jump_events = []
        fixed_lag_confirmed_events = []
        fixed_lag_confirmed_cursor = 0
        live_overlay_events = []
        live_overlay_source_count = -1
        live_all_in_one_events = []
        live_all_in_one_source_count = -1
        landmark_frame_numbers = []
        landmark_timestamps_ms = []
        min_jump_gap_frames = max(4, int(round(fps * MIN_JUMP_GAP_SECONDS)))
        landing_offset_frames = int(round((LANDING_OFFSET_MS / 1000.0) * fps))
        last_jump_frame = -10**9
        enter_window_frames = max(1, int(round(fps * ACTIVE_ENTER_WINDOW_SECONDS)))
        active_enter_max_gap_frames = max(1, int(round(fps * ACTIVE_ENTER_MAX_GAP_SECONDS)))
        exit_idle_frames = max(enter_window_frames, int(round(fps * ACTIVE_EXIT_IDLE_SECONDS)))
        startup_lockout_frames = max(0, int(round(fps * runtime_startup_lockout_seconds)))
        is_active = False
        pending_candidates = []
        last_valid_candidate_frame = -10**9
        rope_flag_series = []
        rope_dual_flag_series = []
        rope_active_window_frames = max(1, int(round(fps * ROPE_ACTIVE_WINDOW_SECONDS)))
        rope_exit_idle_frames = max(1, int(round(fps * ROPE_EXIT_IDLE_SECONDS)))
        last_rope_active_frame = -10**9
        local_motion_flags = []
        last_frame_timestamp_ms = -1
        frame_log_rows = []
    
    
    
        ## Setup mediapipe instance
        try:
            with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret or frame is None:
                        print("[INFO] End of stream or read failure.")
                        break
    
                    processed_frames += 1
                    if is_realtime and REALTIME_MAX_FRAMES > 0 and processed_frames > REALTIME_MAX_FRAMES:
                        print(f"[INFO] Realtime max frame limit reached: {REALTIME_MAX_FRAMES}")
                        break
                    if is_realtime and REALTIME_MAX_SECONDS > 0:
                        elapsed_live = time.time() - start_time
                        if elapsed_live > REALTIME_MAX_SECONDS:
                            print(f"[INFO] Realtime max duration reached: {REALTIME_MAX_SECONDS:.1f}s")
                            break
                    if is_realtime and realtime_fps_log_interval_s > 0.0:
                        now = time.time()
                        fps_log_elapsed = now - last_fps_log_time
                        if fps_log_elapsed >= realtime_fps_log_interval_s:
                            interval_frames = processed_frames - last_fps_log_frames
                            current_fps = interval_frames / max(fps_log_elapsed, 1e-6)
                            avg_fps = processed_frames / max(now - start_time, 1e-6)
                            print(
                                f"[FPS] current={current_fps:.2f} avg={avg_fps:.2f} "
                                f"processed={processed_frames}"
                            )
                            last_fps_log_time = now
                            last_fps_log_frames = processed_frames
                    current_frame_idx = processed_frames - 1
                    raw_timestamp_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
                    current_frame_ts_ms = resolve_frame_timestamp_ms(
                        raw_timestamp_ms,
                        current_frame_idx,
                        fps,
                        last_frame_timestamp_ms,
                    )
                    last_frame_timestamp_ms = current_frame_ts_ms
    
                    # Recolor image to RGB
                    try:
                        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        image.flags.writeable = False
    
                        # Make detection
                        results = pose.process(image)
    
                        # Recolor back to BGR
                        image.flags.writeable = True
                        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                    except Exception as e:
                        print(f"[WARN] Frame processing error: {e}")
                        image = frame
                        results = None
    
                    last_frame_shape = frame.shape
                    pose_detected = False
                    lower_body_reliable = False
                    rope_detected_this_frame = False
                    rope_dual_detected_this_frame = False
                    effective_rope_detected_this_frame = False
    
                    if processed_frames % 300 == 0:
                        elapsed = time.time() - start_time
                        print(f"[INFO] processed={processed_frames} elapsed={elapsed:.1f}s")
    
                    # Extract landmarks
                    try:
                        if results is None or results.pose_landmarks is None:
                            raise ValueError("no_pose")
                        landmarks = results.pose_landmarks.landmark
                        if strict_guard_mode:
                            lower_body_reliable = has_reliable_lower_body(
                                landmarks,
                                runtime_lower_body_vis_min,
                            )
                            if not lower_body_reliable:
                                raise ValueError("lower_body_unreliable")
                        else:
                            lower_body_reliable = True
    
                        hip_center_y.append((landmarks[23].y + landmarks[24].y) * 0.5)
                        left_foot_x.append(landmarks[31].x)
                        right_foot_x.append(landmarks[32].x)
                        left_foot_y.append(landmarks[31].y)
                        right_foot_y.append(landmarks[32].y)
                        foot_center_y.append((landmarks[31].y + landmarks[32].y) * 0.5)
                        foot_center_x.append((landmarks[31].x + landmarks[32].x) * 0.5)
                        landmark_frame_numbers.append(current_frame_idx)
                        landmark_timestamps_ms.append(current_frame_ts_ms)
                        pose_detected = True
    
                        right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                                       landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
                        left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                                      landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
    
                    except Exception:
                        pose_detected = False
    
                    if is_active and (current_frame_idx - last_valid_candidate_frame) > exit_idle_frames:
                        is_active = False
                        pending_candidates = []
                    if is_active and (current_frame_idx - last_rope_active_frame) > rope_exit_idle_frames:
                        is_active = False
                        pending_candidates = []
                    jump_counter_before = jump_counter
    
                    ### online minima counting for jumps (state-gated)
                    readings = hip_center_y
                    candidate_idx = len(readings) - 1 - LOCAL_MINIMA_LAG_FRAMES
                    if candidate_idx >= 1 and candidate_idx > last_processed_minima_idx and (candidate_idx + 1) < len(readings):
                        adaptive_threshold = max(
                            get_adaptive_jump_threshold(readings) * ADAPTIVE_THRESHOLD_GAIN,
                            ADAPTIVE_THRESHOLD_MIN * 0.75,
                        )
                        left_start = max(0, candidate_idx - LOCAL_PROMINENCE_WINDOW)
                        right_end = min(len(readings), candidate_idx + 1 + LOCAL_PROMINENCE_WINDOW)
                        left_neighbors = readings[left_start:candidate_idx]
                        right_neighbors = readings[candidate_idx + 1:right_end]
                        if left_neighbors and right_neighbors:
                            is_local_min = (
                                readings[candidate_idx] < readings[candidate_idx - 1]
                                and readings[candidate_idx] <= readings[candidate_idx + 1]
                            )
                            prominence_left = max(left_neighbors) - readings[candidate_idx]
                            prominence_right = max(right_neighbors) - readings[candidate_idx]
                            prominence = min(prominence_left, prominence_right)
                            strength_ratio = prominence / max(adaptive_threshold, 1e-9)
                            motion_signature = evaluate_foot_motion_signature(
                                candidate_idx=candidate_idx,
                                left_foot_y=left_foot_y,
                                right_foot_y=right_foot_y,
                                foot_center_y=foot_center_y,
                                foot_center_x=foot_center_x,
                                prominence_window=LOCAL_PROMINENCE_WINDOW,
                                foot_lift_min_prominence=runtime_foot_lift_min_prominence,
                                both_feet_min_prominence=STRICT_BOTH_FEET_MIN_PROMINENCE,
                                feet_symmetry_min_ratio=STRICT_FEET_SYMMETRY_MIN_RATIO,
                                foot_sync_window=STRICT_FOOT_SYNC_WINDOW,
                                foot_sync_min_corr=STRICT_FOOT_SYNC_MIN_CORR,
                                require_inplace=STRICT_REQUIRE_INPLACE,
                                inplace_window=STRICT_INPLACE_WINDOW,
                                inplace_max_center_drift=STRICT_INPLACE_MAX_CENTER_DRIFT,
                            )
                            foot_prominence = motion_signature["foot_prominence"]
                            foot_motion_strict_ok = bool(motion_signature["strict_motion_ok"])
                            both_feet_lift_ok = bool(motion_signature.get("both_feet_lift_ok"))
                            feet_symmetry_ok = bool(motion_signature.get("feet_symmetry_ok"))
                            foot_sync_ok = bool(motion_signature.get("foot_sync_ok"))
                            loose_foot_motion_ok = bool(
                                foot_prominence is not None
                                and float(foot_prominence) >= float(STRICT_ACTIVE_LOOSE_FOOT_PROMINENCE)
                            )
                            relaxed_motion_gate_ok = bool(
                                loose_foot_motion_ok
                                and (both_feet_lift_ok or foot_sync_ok)
                            )
                            entry_motion_gate_ok = bool(
                                foot_motion_strict_ok
                                and both_feet_lift_ok
                                and (feet_symmetry_ok or foot_sync_ok)
                            )
                            strict_bilateral_motion_ok = bool(
                                foot_motion_strict_ok
                                and both_feet_lift_ok
                                and (feet_symmetry_ok or foot_sync_ok)
                            )
                            recent_motion_ratio = 0.0
                            recent_motion_true_count = 0
                            if is_local_min:
                                local_motion_flags.append(bool(relaxed_motion_gate_ok))
                                motion_window = max(1, int(STRICT_ACTIVE_FOOT_MOTION_WINDOW))
                                if len(local_motion_flags) > (motion_window * 4):
                                    local_motion_flags = local_motion_flags[-(motion_window * 4):]
                                recent_slice = local_motion_flags[-motion_window:]
                                recent_motion_true_count = int(sum(1 for flag in recent_slice if flag))
                                recent_motion_ratio = (
                                    float(recent_motion_true_count) / float(len(recent_slice))
                                    if recent_slice
                                    else 0.0
                                )
                            candidate_frame = landmark_frame_numbers[candidate_idx]
                            entry_bootstrap_after_idle = bool(
                                last_valid_candidate_frame < 0
                                or (candidate_frame - last_valid_candidate_frame) >= exit_idle_frames
                            )
                            startup_locked = candidate_frame < startup_lockout_frames
                            if startup_locked:
                                last_processed_minima_idx = candidate_idx
                            is_far_enough = (candidate_frame - last_jump_frame) >= min_jump_gap_frames
                            rope_ratio = get_true_ratio(rope_flag_series, candidate_idx, rope_active_window_frames)
                            rope_dual_ratio = get_true_ratio(rope_dual_flag_series, candidate_idx, rope_active_window_frames)
                            candidate_rope_flag = (
                                candidate_idx < len(rope_flag_series)
                                and bool(rope_flag_series[candidate_idx])
                            )
                            candidate_dual_rope_flag = (
                                candidate_idx < len(rope_dual_flag_series)
                                and bool(rope_dual_flag_series[candidate_idx])
                            )
                            foot_motion_ok = foot_motion_strict_ok
                            anti_walk_detected = False
                            if strict_guard_mode:
                                active_history_ok = (
                                    recent_motion_true_count >= max(0, int(STRICT_ACTIVE_FOOT_MOTION_MIN_TRUE))
                                    and recent_motion_ratio >= float(STRICT_ACTIVE_FOOT_MOTION_MIN_RATIO)
                                )
                                if STRICT_ACTIVE_ANTI_WALK_ENABLED:
                                    foot_sync_corr = motion_signature.get("foot_sync_corr")
                                    center_x_drift = motion_signature.get("center_x_drift")
                                    walk_sync = (
                                        foot_sync_corr is not None
                                        and float(foot_sync_corr) <= float(STRICT_ACTIVE_WALK_MAX_SYNC_CORR)
                                    )
                                    walk_drift = (
                                        center_x_drift is not None
                                        and float(center_x_drift) >= float(STRICT_ACTIVE_WALK_MIN_CENTER_DRIFT)
                                        and (
                                            foot_sync_corr is None
                                            or float(foot_sync_corr) < 0.15
                                        )
                                    )
                                    walk_unilateral = (
                                        (not both_feet_lift_ok)
                                        and center_x_drift is not None
                                        and float(center_x_drift) >= float(STRICT_ACTIVE_WALK_MIN_CENTER_DRIFT) * 0.80
                                        and (
                                            foot_sync_corr is None
                                            or float(foot_sync_corr) < 0.25
                                        )
                                    )
                                    anti_walk_detected = bool(walk_sync or walk_drift or walk_unilateral)
                                if is_active:
                                    strong_active_override = (
                                        runtime_strict_active_override_enabled
                                        and rope_ratio >= float(STRICT_OVERRIDE_ACTIVE_MIN_ROPE_RATIO)
                                        and rope_dual_ratio >= float(STRICT_OVERRIDE_ACTIVE_MIN_DUAL_RATIO)
                                    )
                                    foot_motion_ok = bool(
                                        strict_bilateral_motion_ok
                                        or relaxed_motion_gate_ok
                                        or active_history_ok
                                    )
                                    if (not foot_motion_ok) and strong_active_override:
                                        foot_motion_ok = bool(
                                            strength_ratio >= float(STRICT_OVERRIDE_MIN_STRENGTH_RATIO)
                                        )
                                else:
                                    strong_entry_override = (
                                        runtime_strict_entry_override_enabled
                                        and rope_ratio >= float(STRICT_OVERRIDE_ENTRY_MIN_ROPE_RATIO)
                                        and rope_dual_ratio >= float(STRICT_OVERRIDE_ENTRY_MIN_DUAL_RATIO)
                                        and strength_ratio >= float(STRICT_OVERRIDE_MIN_STRENGTH_RATIO)
                                    )
                                    bootstrap_relaxed_entry = bool(
                                        STRICT_ENTRY_NO_FLAG_BOOTSTRAP_ENABLED
                                        and entry_bootstrap_after_idle
                                        and relaxed_motion_gate_ok
                                        and strength_ratio
                                        >= float(STRICT_ENTRY_NO_FLAG_BOOTSTRAP_MIN_STRENGTH_RATIO)
                                    )
                                    foot_motion_ok = bool(
                                        entry_motion_gate_ok
                                        or strong_entry_override
                                        or bootstrap_relaxed_entry
                                    )
                                if anti_walk_detected:
                                    foot_motion_ok = False
                            if not strict_guard_mode:
                                foot_motion_ok = True
                            if is_active:
                                active_dual_ok = (
                                    ROPE_ACTIVE_DUAL_MIN_RATIO <= 0.0
                                    or rope_dual_ratio >= ROPE_ACTIVE_DUAL_MIN_RATIO
                                )
                                rope_active = rope_ratio >= ROPE_ACTIVE_MIN_RATIO and active_dual_ok
                                if (
                                    strict_guard_mode
                                    and (not candidate_rope_flag)
                                    and runtime_strict_active_override_enabled
                                ):
                                    rope_active = bool(
                                        rope_active
                                        and rope_ratio >= float(STRICT_OVERRIDE_ACTIVE_MIN_ROPE_RATIO)
                                        and rope_dual_ratio >= float(STRICT_OVERRIDE_ACTIVE_MIN_DUAL_RATIO)
                                    )
                            else:
                                entry_dual_ok = (
                                    ROPE_ENTRY_DUAL_MIN_RATIO <= 0.0
                                    or rope_dual_ratio >= ROPE_ENTRY_DUAL_MIN_RATIO
                                )
                                rope_active = (
                                    rope_ratio >= ROPE_ENTRY_MIN_RATIO
                                    and entry_dual_ok
                                )
                                if strict_guard_mode:
                                    entry_strict_rope_ok = bool(
                                        candidate_rope_flag
                                        and (
                                            candidate_dual_rope_flag
                                            or rope_dual_ratio >= float(STRICT_OVERRIDE_ENTRY_MIN_DUAL_RATIO)
                                        )
                                    )
                                    entry_no_flag_bootstrap_ok = bool(
                                        STRICT_ENTRY_NO_FLAG_BOOTSTRAP_ENABLED
                                        and entry_bootstrap_after_idle
                                        and strength_ratio >= float(STRICT_ENTRY_NO_FLAG_BOOTSTRAP_MIN_STRENGTH_RATIO)
                                        and rope_ratio >= float(STRICT_ENTRY_NO_FLAG_BOOTSTRAP_MIN_ROPE_RATIO)
                                        and rope_dual_ratio >= float(STRICT_ENTRY_NO_FLAG_BOOTSTRAP_MIN_DUAL_RATIO)
                                    )
                                    rope_active = bool(
                                        rope_active
                                        and (entry_strict_rope_ok or entry_no_flag_bootstrap_ok)
                                    )
                            if (
                                (not startup_locked)
                                and is_local_min
                                and foot_motion_ok
                                and strength_ratio >= runtime_min_strength_ratio
                                and is_far_enough
                                and rope_active
                            ):
                                last_jump_frame = candidate_frame
                                last_valid_candidate_frame = candidate_frame
                                candidate_info = {
                                    "candidate_idx": candidate_idx,
                                    "candidate_frame": candidate_frame,
                                    "strength_ratio": strength_ratio,
                                    "foot_prominence": (
                                        float(foot_prominence) if foot_prominence is not None else np.nan
                                    ),
                                    "left_foot_prominence": float(
                                        motion_signature.get("left_foot_prominence", np.nan)
                                    ),
                                    "right_foot_prominence": float(
                                        motion_signature.get("right_foot_prominence", np.nan)
                                    ),
                                    "both_feet_lift_ok": bool(both_feet_lift_ok),
                                    "feet_symmetry_ratio": float(
                                        motion_signature.get("feet_symmetry_ratio", np.nan)
                                    ),
                                    "feet_symmetry_ok": bool(feet_symmetry_ok),
                                    "foot_sync_corr": float(
                                        motion_signature.get("foot_sync_corr", np.nan)
                                    ),
                                    "foot_sync_ok": bool(foot_sync_ok),
                                    "inplace_ok": bool(motion_signature.get("inplace_ok")),
                                    "rope_ratio": float(rope_ratio),
                                    "rope_dual_ratio": float(rope_dual_ratio),
                                }
    
                                if is_active:
                                    jump_counter += 1
                                    detected_jump_events.append(
                                        build_detected_jump_event(
                                            candidate_idx=candidate_idx,
                                            landing_offset_frames=landing_offset_frames,
                                            event_time_bias_ms=EVENT_TIME_BIAS_MS,
                                            frame_numbers=landmark_frame_numbers,
                                            frame_timestamps_ms=landmark_timestamps_ms,
                                            left_foot_x=left_foot_x,
                                            left_foot_y=left_foot_y,
                                            right_foot_x=right_foot_x,
                                            right_foot_y=right_foot_y,
                                            fps=fps,
                                            width=width,
                                            height=height,
                                            count=jump_counter,
                                            strength_ratio=strength_ratio,
                                            rope_ratio=float(rope_ratio),
                                            rope_dual_ratio=float(rope_dual_ratio),
                                            foot_prominence=float(foot_prominence)
                                            if foot_prominence is not None
                                            else np.nan,
                                            left_foot_prominence=float(
                                                motion_signature.get("left_foot_prominence", np.nan)
                                            ),
                                            right_foot_prominence=float(
                                                motion_signature.get("right_foot_prominence", np.nan)
                                            ),
                                            both_feet_lift_ok=bool(both_feet_lift_ok),
                                            feet_symmetry_ratio=float(
                                                motion_signature.get("feet_symmetry_ratio", np.nan)
                                            ),
                                            feet_symmetry_ok=bool(feet_symmetry_ok),
                                            foot_sync_corr=float(
                                                motion_signature.get("foot_sync_corr", np.nan)
                                            ),
                                            foot_sync_ok=bool(foot_sync_ok),
                                            inplace_ok=bool(motion_signature.get("inplace_ok")),
                                        )
                                    )
                                else:
                                    pending_candidates.append(candidate_info)
                                    min_frame = candidate_frame - enter_window_frames
                                    pending_candidates = [
                                        item for item in pending_candidates
                                        if item["candidate_frame"] >= min_frame
                                    ]
                                    if has_stable_entry_cadence(
                                        pending_candidates,
                                        min_events=runtime_enter_min_events,
                                        max_gap_frames=active_enter_max_gap_frames,
                                        max_cv=ACTIVE_ENTER_CADENCE_MAX_CV,
                                    ):
                                        is_active = True
                                        tail_n = choose_entry_backfill_tail_count(
                                            pending_candidates,
                                            active_enter_min_events=runtime_enter_min_events,
                                        )
                                        if tail_n > 0:
                                            for confirmed in pending_candidates[-tail_n:]:
                                                jump_counter += 1
                                                detected_jump_events.append(
                                                    build_detected_jump_event(
                                                        candidate_idx=confirmed["candidate_idx"],
                                                        landing_offset_frames=landing_offset_frames,
                                                        event_time_bias_ms=EVENT_TIME_BIAS_MS,
                                                        frame_numbers=landmark_frame_numbers,
                                                        frame_timestamps_ms=landmark_timestamps_ms,
                                                        left_foot_x=left_foot_x,
                                                        left_foot_y=left_foot_y,
                                                        right_foot_x=right_foot_x,
                                                        right_foot_y=right_foot_y,
                                                        fps=fps,
                                                        width=width,
                                                        height=height,
                                                        count=jump_counter,
                                                        strength_ratio=confirmed["strength_ratio"],
                                                        rope_ratio=float(confirmed.get("rope_ratio", rope_ratio)),
                                                        rope_dual_ratio=float(confirmed.get("rope_dual_ratio", rope_dual_ratio)),
                                                        foot_prominence=float(
                                                            confirmed.get("foot_prominence", np.nan)
                                                        ),
                                                        left_foot_prominence=float(
                                                            confirmed.get("left_foot_prominence", np.nan)
                                                        ),
                                                        right_foot_prominence=float(
                                                            confirmed.get("right_foot_prominence", np.nan)
                                                        ),
                                                        both_feet_lift_ok=bool(
                                                            confirmed.get("both_feet_lift_ok", False)
                                                        ),
                                                        feet_symmetry_ratio=float(
                                                            confirmed.get("feet_symmetry_ratio", np.nan)
                                                        ),
                                                        feet_symmetry_ok=bool(
                                                            confirmed.get("feet_symmetry_ok", False)
                                                        ),
                                                        foot_sync_corr=float(
                                                            confirmed.get("foot_sync_corr", np.nan)
                                                        ),
                                                        foot_sync_ok=bool(
                                                            confirmed.get("foot_sync_ok", False)
                                                        ),
                                                        inplace_ok=bool(
                                                            confirmed.get("inplace_ok", False)
                                                        ),
                                                    )
                                                )
                                        pending_candidates = []
                        last_processed_minima_idx = candidate_idx
    
                    current_ts_ms = current_frame_ts_ms + DISPLAY_TIME_ADVANCE_MS
                    raw_jump_counter = int(jump_counter)
                    fixed_lag_confirmed_count = int(len(fixed_lag_confirmed_events))
                    fixed_lag_pending_count = int(
                        max(0, len(detected_jump_events) - fixed_lag_confirmed_count)
                    )
                    if use_live_fixed_lag_overlay or use_fixed_lag_final:
                        fixed_lag_confirmed_cursor = advance_fixed_lag_confirmed_events(
                            detected_jump_events=detected_jump_events,
                            confirmed_events=fixed_lag_confirmed_events,
                            confirmed_cursor=fixed_lag_confirmed_cursor,
                            current_display_ts_ms=current_ts_ms,
                            fixed_lag_ms=fixed_lag_ms,
                            fps=fps,
                        )
                        fixed_lag_confirmed_count = int(len(fixed_lag_confirmed_events))
                        fixed_lag_pending_count = int(
                            max(0, len(detected_jump_events) - fixed_lag_confirmed_count)
                        )

                    overlay_jump_counter = raw_jump_counter
                    overlay_counter_mode = "raw"
                    if all_in_one_enabled:
                        if live_all_in_one_source_count != raw_jump_counter:
                            live_all_in_one_events, _ = _select_all_in_one_events(
                                detected_jump_events,
                                strict_guard_mode=strict_guard_mode,
                            )
                            live_all_in_one_source_count = int(raw_jump_counter)
                        overlay_jump_counter = int(len(live_all_in_one_events))
                        overlay_counter_mode = "all_in_one"
                    elif use_live_fixed_lag_overlay:
                        if bool(LIVE_OVERLAY_STREAM_POSTPROCESS):
                            if live_overlay_source_count != fixed_lag_confirmed_count:
                                live_overlay_events, _ = _postprocess_confirmed_events(
                                    fixed_lag_confirmed_events,
                                    strict_guard_mode=strict_guard_mode,
                                )
                                live_overlay_source_count = int(fixed_lag_confirmed_count)
                            overlay_jump_counter = int(len(live_overlay_events))
                            overlay_counter_mode = "stream_post"
                        else:
                            overlay_jump_counter = fixed_lag_confirmed_count
                            overlay_counter_mode = "fixed_lag"
                    ankle_counter = int(overlay_jump_counter)

                    while (
                        label_progress_idx < len(label_events)
                        and label_events[label_progress_idx]["timestamp_ms"] <= current_ts_ms
                    ):
                        label_progress_idx += 1
                    count_delta = ankle_counter - label_progress_idx
                    raw_count_delta = raw_jump_counter - label_progress_idx

                    if results is not None and results.pose_landmarks is not None:
                        mp_drawing.draw_landmarks(
                            image,
                            results.pose_landmarks,
                            mp_pose.POSE_CONNECTIONS,
                            mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                            mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2),
                        )

                    # Show only the current count (no additional labels).
                    panel_width = 110
                    panel_height = 62
                    panel_left = max(0, image.shape[1] - panel_width)
                    panel_top = 0
                    panel_right = image.shape[1]
                    panel_bottom = min(image.shape[0], panel_height)
                    cv2.rectangle(
                        image,
                        (panel_left, panel_top),
                        (panel_right, panel_bottom),
                        (0, 0, 0),
                        -1,
                    )
                    count_text = str(int(ankle_counter))
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 1.35
                    thickness = 3
                    (text_width, text_height), _ = cv2.getTextSize(
                        count_text,
                        font,
                        font_scale,
                        thickness,
                    )
                    text_x = max(panel_left + 8, panel_right - 10 - text_width)
                    text_y = max(panel_top + text_height + 8, panel_bottom - 14)
                    cv2.putText(
                        image,
                        count_text,
                        (text_x, text_y),
                        font,
                        font_scale,
                        (0, 0, 255),
                        thickness,
                        cv2.LINE_AA,
                    )
    
                    if pose_detected:
                        # left hand coordinates for region of interests around wrist
                        e, f = left_wrist
                        efarray = np.multiply((e, f), frame.shape[::-1][1:])
                        e1, f1 = int(efarray[0]), int(efarray[1])
                        f1 += 20
                        e2, f2 = e1 + 70, f1 + 70
    
                        # right hand coordinates for region of interests around wrist
                        m, n = right_wrist
                        mnarray = np.multiply((m, n), frame.shape[::-1][1:])
                        m1, n1 = int(mnarray[0]), int(mnarray[1])
                        n1 += 20
                        m2, n2 = m1 + 70, n1 + 70
    
                        # Apply rope-detection logic near both wrist ROIs.
                        maskKNN = knnfgbg.apply(frame)
                        if rope_mask_kernel is not None:
                            if ROPE_MASK_OPEN_ITERS > 0:
                                maskKNN = cv2.morphologyEx(
                                    maskKNN,
                                    cv2.MORPH_OPEN,
                                    rope_mask_kernel,
                                    iterations=ROPE_MASK_OPEN_ITERS,
                                )
                            if ROPE_MASK_CLOSE_ITERS > 0:
                                maskKNN = cv2.morphologyEx(
                                    maskKNN,
                                    cv2.MORPH_CLOSE,
                                    rope_mask_kernel,
                                    iterations=ROPE_MASK_CLOSE_ITERS,
                                )
                        contours, _ = cv2.findContours(maskKNN, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        left_roi_hit = False
                        right_roi_hit = False
                        for contour in contours:
                            if cv2.contourArea(contour) < ROPE_CONTOUR_MIN_AREA:
                                continue
                            x, y, w, h = cv2.boundingRect(contour)
                            long_side = max(int(w), int(h))
                            short_side = max(1, min(int(w), int(h)))
                            if long_side < int(ROPE_CONTOUR_MIN_SPAN):
                                continue
                            aspect_ratio = float(short_side) / float(long_side)
                            if aspect_ratio > float(ROPE_CONTOUR_MAX_ASPECT_RATIO):
                                continue
                            if cv2.arcLength(contour, True) < float(ROPE_CONTOUR_MIN_PERIMETER):
                                continue
                            for point in contour:
                                i, j = point[0]
                                in_left_roi = e1 <= i <= e2 and f1 <= j <= f2
                                in_right_roi = m1 <= i <= m2 and n1 <= j <= n2
                                if in_left_roi:
                                    left_roi_hit = True
                                if in_right_roi:
                                    right_roi_hit = True
                                if left_roi_hit and right_roi_hit:
                                    break
                            if left_roi_hit and right_roi_hit:
                                break
                        rope_detected_this_frame = left_roi_hit or right_roi_hit
                        rope_dual_detected_this_frame = left_roi_hit and right_roi_hit
                        effective_rope_detected_this_frame = rope_detected_this_frame
                        if runtime_require_dual_rope:
                            effective_rope_detected_this_frame = rope_dual_detected_this_frame
    
                    if pose_detected:
                        rope_flag_series.append(effective_rope_detected_this_frame)
                        rope_dual_flag_series.append(rope_dual_detected_this_frame)
                        if effective_rope_detected_this_frame:
                            last_rope_active_frame = landmark_frame_numbers[-1]

                    if frame_log_csv_path:
                        frame_log_rows.append(
                            {
                                "frame": int(current_frame_idx),
                                "timestamp_ms": int(current_frame_ts_ms),
                                "pose_detected": int(bool(pose_detected)),
                                "lower_body_reliable": int(bool(lower_body_reliable)),
                                "rope_detected": int(bool(rope_detected_this_frame)),
                                "rope_dual_detected": int(bool(rope_dual_detected_this_frame)),
                                "effective_rope_detected": int(bool(effective_rope_detected_this_frame)),
                                "is_active": int(bool(is_active)),
                                "pending_candidates": int(len(pending_candidates)),
                                "jump_counter": int(raw_jump_counter),
                                "jump_increment": int(raw_jump_counter - jump_counter_before),
                                "fixed_lag_confirmed": int(fixed_lag_confirmed_count),
                                "fixed_lag_pending": int(fixed_lag_pending_count),
                                "overlay_counter": int(ankle_counter),
                                "overlay_counter_mode": overlay_counter_mode,
                                "raw_count_delta": int(raw_count_delta),
                                "label_progress": int(label_progress_idx),
                                "count_delta": int(count_delta),
                            }
                        )

                    if raw_out is not None:
                        raw_out.write(frame)

                    if image is not None:
                        out.write(image)
    
                    if not HEADLESS:
                        cv2.imshow('Mediapipe Feed', image)
                        if cv2.waitKey(10) & 0xFF == ord('q'):
                            break
    
        except Exception as e:
            print(f"[ERROR] Processing failed: {e}")
        finally:
            cap.release()
            out.release()
            if raw_out is not None:
                raw_out.release()
            if not HEADLESS:
                cv2.destroyAllWindows()
            if frame_log_csv_path:
                pd.DataFrame(frame_log_rows).to_csv(frame_log_csv_path, index=False)
                print(f"[LOG] frame log saved: {frame_log_csv_path} rows={len(frame_log_rows)}")
            elapsed = time.time() - start_time
            print(f"[DONE] {out_path} frames={processed_frames} elapsed={elapsed:.1f}s")
    
        if processed_frames == 0 or last_frame_shape is None:
            print("[WARN] No frames processed; skipping post-processing for this file.")
            continue
    
        postprocess_start = time.time()
        print(f"[POST] Starting post-processing: {stem}")
        if all_in_one_enabled:
            selected_events, all_in_one_stats = _select_all_in_one_events(
                detected_jump_events,
                strict_guard_mode=strict_guard_mode,
            )
            pruned_weak_non_bilateral = int(all_in_one_stats["pruned_weak_non_bilateral"])
            inserted_by_entry_backfill = int(all_in_one_stats["inserted_by_entry_backfill"])
            inserted_by_head_fill = int(all_in_one_stats["inserted_by_head_fill"])
            inserted_by_gap_fill = int(all_in_one_stats["inserted_by_gap_fill"])
            confirmed_segments = int(all_in_one_stats["confirmed_segments"])
            dropped_segments = int(all_in_one_stats["dropped_segments"])
            segment_debug_rows = list(all_in_one_stats["segment_debug_rows"])
            pending_after_finalize = 0
            print(
                f"[POST] all_in_one_final_count raw={len(detected_jump_events)} "
                f"final={len(selected_events)} "
                f"confirmed_segments={confirmed_segments} "
                f"dropped_segments={dropped_segments}"
            )
            if pruned_weak_non_bilateral > 0:
                print(
                    f"[POST] pruned_weak_non_bilateral events={pruned_weak_non_bilateral} "
                    f"max_strength={float(STRICT_WEAK_NON_BILATERAL_MAX_STRENGTH):.2f}"
                )
            if inserted_by_entry_backfill > 0:
                print(
                    f"[POST] all_in_one_entry_backfill inserted={inserted_by_entry_backfill} "
                    f"min_events={int(ALL_IN_ONE_ENTRY_BACKFILL_MIN_EVENTS)} "
                    f"max_events={int(ALL_IN_ONE_ENTRY_BACKFILL_MAX_EVENTS)}"
                )
            if inserted_by_head_fill > 0:
                print(
                    f"[POST] all_in_one_head_fill inserted={inserted_by_head_fill} "
                    f"max_insert=2"
                )
            if inserted_by_gap_fill > 0:
                print(
                    f"[POST] all_in_one_gap_fill inserted={inserted_by_gap_fill} "
                    f"min_ratio={float(ALL_IN_ONE_GAP_FILL_MIN_RATIO):.2f} "
                    f"max_ratio={float(STRICT_SESSION_GAP_FILL_MAX_RATIO):.2f}"
                )
            if dropped_segments > 0 or confirmed_segments > 0:
                for (
                    seg_idx,
                    seg_len,
                    seg_start_ts,
                    seg_end_ts,
                    seg_cv,
                    seg_abs_fp_median,
                    seg_kept,
                    seg_reason,
                ) in segment_debug_rows:
                    print(
                        f"[POST] all_in_one_segment idx={seg_idx} "
                        f"len={seg_len} start_ts={seg_start_ts} end_ts={seg_end_ts} "
                        f"cv={seg_cv:.3f} abs_fp_median={seg_abs_fp_median:.6f} "
                        f"kept={seg_kept} reason={seg_reason}"
                    )
        else:
            final_display_ts_ms = int(
                last_frame_timestamp_ms + DISPLAY_TIME_ADVANCE_MS + fixed_lag_finalize_wait_ms
            )
            fixed_lag_confirmed_cursor = advance_fixed_lag_confirmed_events(
                detected_jump_events=detected_jump_events,
                confirmed_events=fixed_lag_confirmed_events,
                confirmed_cursor=fixed_lag_confirmed_cursor,
                current_display_ts_ms=final_display_ts_ms,
                fixed_lag_ms=fixed_lag_ms,
                fps=fps,
            )
            selected_events, postprocess_stats = _postprocess_confirmed_events(
                fixed_lag_confirmed_events,
                strict_guard_mode=strict_guard_mode,
            )
            pruned_weak_non_bilateral = int(postprocess_stats["pruned_weak_non_bilateral"])
            pruned_by_session_filter = int(postprocess_stats["pruned_by_session_filter"])
            pruned_by_session_strength = int(postprocess_stats["pruned_by_session_strength"])
            pruned_by_run_min_events = int(postprocess_stats["pruned_by_run_min_events"])
            inserted_by_gap_fill = int(postprocess_stats["inserted_by_gap_fill"])
            inserted_by_head_fill = int(postprocess_stats["inserted_by_head_fill"])
            session_split_gap_ms = int(postprocess_stats["session_split_gap_ms"])
            session_min_events = int(postprocess_stats["session_min_events"])
            session_max_cadence_cv = float(postprocess_stats["session_max_cadence_cv"])
            kept_segments = int(postprocess_stats["kept_segments"])
            dropped_segments = int(postprocess_stats["dropped_segments"])
            bridged_short_segments = int(postprocess_stats["bridged_short_segments"])
            segment_debug_rows = list(postprocess_stats["segment_debug_rows"])
            if strict_guard_mode and bool(STRICT_SESSION_FILTER_ENABLED) and (
                dropped_segments > 0 or bridged_short_segments > 0 or kept_segments > 0
            ):
                total_segments = kept_segments + dropped_segments
                print(
                    f"[POST] session_filter kept_segments={kept_segments}/{total_segments} "
                    f"dropped_segments={dropped_segments} "
                    f"split_gap_ms={session_split_gap_ms} "
                    f"min_events={session_min_events} "
                    f"max_cadence_cv={session_max_cadence_cv:.2f} "
                    f"bridged_short_segments={bridged_short_segments}"
                )
                if dropped_segments > 0 or bridged_short_segments > 0:
                    for (
                        seg_idx,
                        seg_len,
                        seg_start_ts,
                        seg_end_ts,
                        seg_cv,
                        seg_abs_fp_median,
                        seg_kept,
                        seg_reason,
                    ) in segment_debug_rows:
                        print(
                            f"[POST] session_filter_segment idx={seg_idx} "
                            f"len={seg_len} start_ts={seg_start_ts} end_ts={seg_end_ts} "
                            f"cv={seg_cv:.3f} abs_fp_median={seg_abs_fp_median:.6f} "
                            f"kept={seg_kept} reason={seg_reason}"
                        )
            pending_after_finalize = max(0, len(detected_jump_events) - len(selected_events))
            print(
                f"[POST] fixed_lag_final_count first_pass={len(detected_jump_events)} "
                f"final={len(selected_events)} "
                f"pending_after_finalize={pending_after_finalize} "
                f"fixed_lag_s={FIXED_LAG_SECONDS:.2f} "
                f"finalize_wait_s={FIXED_LAG_FINALIZE_WAIT_SECONDS:.2f}"
            )
            if pruned_weak_non_bilateral > 0:
                print(
                    f"[POST] pruned_weak_non_bilateral events={pruned_weak_non_bilateral} "
                    f"max_strength={float(STRICT_WEAK_NON_BILATERAL_MAX_STRENGTH):.2f}"
                )
            if pruned_by_session_strength > 0:
                print(
                    f"[POST] pruned_by_session_strength events={pruned_by_session_strength} "
                    f"min_median_strength={float(STRICT_SESSION_MIN_MEDIAN_STRENGTH_RATIO):.2f}"
                )
            if pruned_by_session_filter > 0:
                print(
                    f"[POST] pruned_by_session_filter events={pruned_by_session_filter} "
                    f"split_gap_s={float(STRICT_SESSION_SPLIT_GAP_SECONDS):.2f} "
                    f"max_cadence_cv={float(STRICT_SESSION_MAX_CADENCE_CV):.2f}"
                )
            if pruned_by_run_min_events > 0:
                print(
                    f"[POST] pruned_by_run_min_events events={pruned_by_run_min_events} "
                    f"min_events={int(SESSION_MIN_EVENTS)}"
                )
            if inserted_by_gap_fill > 0:
                print(
                    f"[POST] session_gap_fill inserted={inserted_by_gap_fill} "
                    f"min_ratio={float(STRICT_SESSION_GAP_FILL_MIN_RATIO):.2f} "
                    f"max_ratio={float(STRICT_SESSION_GAP_FILL_MAX_RATIO):.2f}"
                )
            if inserted_by_head_fill > 0:
                print(
                    f"[POST] session_head_fill inserted={inserted_by_head_fill} "
                    f"min_events={int(STRICT_SESSION_HEAD_FILL_MIN_EVENTS)} "
                    f"max_cadence_cv={float(STRICT_SESSION_HEAD_FILL_MAX_CADENCE_CV):.2f}"
                )
        if not label_events:
            if ENABLE_OVERLAY_REFRESH:
                overlay_ok, overlay_message = rewrite_tracked_video_count_overlay(
                    out_path,
                    selected_events,
                    [],
                )
                if overlay_ok:
                    print(f"[POST] refreshed_count_overlay {overlay_message}")
                else:
                    print(f"[WARN] Failed to refresh count overlay: {overlay_message}")
            else:
                print("[POST] skipped_count_overlay_refresh (JR_ENABLE_OVERLAY_REFRESH=false)")

            detected_events_csv = os.path.join(job_output_dir, f"{stem}_detected_events.csv")
            _write_detected_events_csv(detected_events_csv, selected_events)
            print(f"[RESULT] detected_count={len(selected_events)}")
            print(f"[RESULT] detected events saved: {detected_events_csv}")
    
            post_elapsed = time.time() - postprocess_start
            print(f"[POST] Completed post-processing: {stem} elapsed={post_elapsed:.1f}s")
    
            overall_summary_rows.append(
                {
                    "video_stem": stem,
                    "video_path": file_path,
                    "label_path": label_path or "",
                    "tolerance_ms": match_tolerance_ms,
                    "frame_based_tolerance_ms": frame_based_tolerance_ms,
                    "raw_detected_count": len(selected_events),
                    "compared_detected_count": len(selected_events),
                    "outside_window_count": 0,
                    "label_count": 0,
                    "matched": 0,
                    "missed_labels": 0,
                    "strict_extra_detected": len(selected_events),
                    "adjusted_extra_detected": len(selected_events),
                    "full_strict_extra_detected": len(selected_events),
                    "full_adjusted_extra_detected": len(selected_events),
                    "label_gap_candidates": 0,
                    "label_window_trim_removed": 0,
                    "label_gap_trim_removed": 0,
                    "strict_precision": np.nan,
                    "strict_recall": np.nan,
                    "strict_f1": np.nan,
                    "strict_accuracy": np.nan,
                    "adjusted_precision": np.nan,
                    "adjusted_recall": np.nan,
                    "adjusted_f1": np.nan,
                    "adjusted_accuracy": np.nan,
                    "full_strict_precision": np.nan,
                    "full_strict_recall": np.nan,
                    "full_strict_f1": np.nan,
                    "full_strict_accuracy": np.nan,
                    "full_adjusted_precision": np.nan,
                    "full_adjusted_recall": np.nan,
                    "full_adjusted_f1": np.nan,
                    "full_adjusted_accuracy": np.nan,
                    "mean_abs_time_error_ms": np.nan,
                    "mean_left_point_error_px": np.nan,
                    "mean_right_point_error_px": np.nan,
                }
            )
            continue

        detected_events_csv = os.path.join(job_output_dir, f"{stem}_detected_events.csv")
        _write_detected_events_csv(detected_events_csv, selected_events)
        print(f"[RESULT] detected events saved: {detected_events_csv}")

        baseline_eval = evaluate_detected_events(
            selected_events,
            label_events,
            tolerance_ms=match_tolerance_ms,
        )
        selected_eval = baseline_eval
    
        if ENABLE_OVERLAY_REFRESH:
            overlay_ok, overlay_message = rewrite_tracked_video_count_overlay(
                out_path,
                selected_events,
                label_events,
            )
            if overlay_ok:
                print(f"[POST] refreshed_count_overlay {overlay_message}")
            else:
                print(f"[WARN] Failed to refresh count overlay: {overlay_message}")
        else:
            print("[POST] skipped_count_overlay_refresh (JR_ENABLE_OVERLAY_REFRESH=false)")
    
        post_elapsed = time.time() - postprocess_start
        print(f"[POST] Completed post-processing: {stem} elapsed={post_elapsed:.1f}s")
    
        compared_detected_events = selected_eval["compared_detected_events"]
        outside_window_events = selected_eval["outside_window_events"]
        confirmed_outside_window_events = selected_eval["confirmed_outside_window_events"]
        matched_events = selected_eval["matched_events"]
        strict_extra_detected = selected_eval["strict_extra_detected"]
        extra_detected = selected_eval["extra_detected"]
        full_strict_extra_detected = selected_eval["full_strict_extra_detected"]
        full_extra_detected = selected_eval["full_extra_detected"]
        gap_candidate_events = selected_eval["gap_candidate_events"]
        boundary_candidate_events = selected_eval["boundary_candidate_events"]
        missed_labels = selected_eval["missed_labels"]
        strict_precision = selected_eval["strict_precision"]
        strict_recall = selected_eval["strict_recall"]
        strict_f1 = selected_eval["strict_f1"]
        strict_accuracy = selected_eval["strict_accuracy"]
        full_strict_precision = selected_eval["full_strict_precision"]
        full_strict_recall = selected_eval["full_strict_recall"]
        full_strict_f1 = selected_eval["full_strict_f1"]
        full_strict_accuracy = selected_eval["full_strict_accuracy"]
        precision = selected_eval["precision"]
        recall = selected_eval["recall"]
        f1 = selected_eval["f1"]
        accuracy = selected_eval["accuracy"]
        full_precision = selected_eval["full_precision"]
        full_recall = selected_eval["full_recall"]
        full_f1 = selected_eval["full_f1"]
        full_accuracy = selected_eval["full_accuracy"]
        mean_abs_time_error_ms = selected_eval["mean_abs_time_error_ms"]
        print("\n[COMPARE] label vs detected")
        print(
            f"[COMPARE] label_count={len(label_events)} raw_detected_count={len(selected_events)} "
            f"compared_detected_count={len(compared_detected_events)} "
            f"outside_window={len(outside_window_events)}"
        )
        print(
            f"[COMPARE] matched={len(matched_events)} "
            f"missed_labels={len(missed_labels)} strict_extra_detected={len(strict_extra_detected)} "
            f"(tolerance={match_tolerance_ms}ms)"
        )
        print(
            f"[COMPARE] strict_precision={strict_precision:.3f} strict_recall={strict_recall:.3f} "
            f"strict_f1={strict_f1:.3f} strict_accuracy={strict_accuracy:.3f}"
        )
        print(
            f"[COMPARE] adjusted_extra_detected={len(extra_detected)} "
            f"label_gap_candidates={len(gap_candidate_events)}"
        )
        print(
            f"[COMPARE] adjusted_precision={precision:.3f} adjusted_recall={recall:.3f} "
            f"adjusted_f1={f1:.3f} adjusted_accuracy={accuracy:.3f}"
        )
        print(
            f"[COMPARE] full_strict_extra_detected={len(full_strict_extra_detected)} "
            f"(outside_window={len(outside_window_events)} "
            f"boundary_candidates={len(boundary_candidate_events)})"
        )
        print(
            f"[COMPARE] full_strict_precision={full_strict_precision:.3f} "
            f"full_strict_recall={full_strict_recall:.3f} "
            f"full_strict_f1={full_strict_f1:.3f} "
            f"full_strict_accuracy={full_strict_accuracy:.3f}"
        )
        print(
            f"[COMPARE] full_adjusted_extra_detected={len(full_extra_detected)} "
            f"boundary_candidates={len(boundary_candidate_events)} "
            f"full_adjusted_precision={full_precision:.3f} "
            f"full_adjusted_recall={full_recall:.3f} "
            f"full_adjusted_f1={full_f1:.3f} "
            f"full_adjusted_accuracy={full_accuracy:.3f}"
        )
        mean_left_point_error_px = np.nan
        mean_right_point_error_px = np.nan
        if matched_events:
            left_point_errors = [item["left_point_error_px"] for item in matched_events]
            right_point_errors = [item["right_point_error_px"] for item in matched_events]
            mean_left_point_error_px = float(np.mean(left_point_errors))
            mean_right_point_error_px = float(np.mean(right_point_errors))
            print(
                "[COMPARE] mean_abs_time_error_ms="
                f"{mean_abs_time_error_ms:.1f} mean_left_point_error_px={mean_left_point_error_px:.1f} "
                f"mean_right_point_error_px={mean_right_point_error_px:.1f}"
            )
        else:
            print("[COMPARE] No matched events. Check counting threshold or label timestamps.")
    
        report_rows = []
        for item in matched_events:
            report_rows.append(
                {
                    "status": "matched",
                    "label_timestamp_ms": item["label"]["timestamp_ms"],
                    "detected_timestamp_ms": item["detected"]["timestamp_ms"],
                    "time_error_ms": item["time_error_ms"],
                    "left_point_error_px": item["left_point_error_px"],
                    "right_point_error_px": item["right_point_error_px"],
                    "label_left_x": item["label"]["left_point"][0],
                    "label_left_y": item["label"]["left_point"][1],
                    "detected_left_x": item["detected"]["left_point"][0],
                    "detected_left_y": item["detected"]["left_point"][1],
                    "label_right_x": item["label"]["right_point"][0],
                    "label_right_y": item["label"]["right_point"][1],
                    "detected_right_x": item["detected"]["right_point"][0],
                    "detected_right_y": item["detected"]["right_point"][1],
                }
            )
        for label_item in missed_labels:
            report_rows.append(
                {
                    "status": "missed_label",
                    "label_timestamp_ms": label_item["timestamp_ms"],
                    "detected_timestamp_ms": np.nan,
                    "time_error_ms": np.nan,
                    "left_point_error_px": np.nan,
                    "right_point_error_px": np.nan,
                    "label_left_x": label_item["left_point"][0],
                    "label_left_y": label_item["left_point"][1],
                    "detected_left_x": np.nan,
                    "detected_left_y": np.nan,
                    "label_right_x": label_item["right_point"][0],
                    "label_right_y": label_item["right_point"][1],
                    "detected_right_x": np.nan,
                    "detected_right_y": np.nan,
                }
            )
        for detected_item in extra_detected:
            report_rows.append(
                {
                    "status": "extra_detected",
                    "label_timestamp_ms": np.nan,
                    "detected_timestamp_ms": detected_item["timestamp_ms"],
                    "time_error_ms": np.nan,
                    "left_point_error_px": np.nan,
                    "right_point_error_px": np.nan,
                    "label_left_x": np.nan,
                    "label_left_y": np.nan,
                    "detected_left_x": detected_item["left_point"][0],
                    "detected_left_y": detected_item["left_point"][1],
                    "label_right_x": np.nan,
                    "label_right_y": np.nan,
                    "detected_right_x": detected_item["right_point"][0],
                    "detected_right_y": detected_item["right_point"][1],
                }
            )
        for detected_item in gap_candidate_events:
            report_rows.append(
                {
                    "status": "label_gap_candidate",
                    "label_timestamp_ms": np.nan,
                    "detected_timestamp_ms": detected_item["timestamp_ms"],
                    "time_error_ms": np.nan,
                    "left_point_error_px": np.nan,
                    "right_point_error_px": np.nan,
                    "label_left_x": np.nan,
                    "label_left_y": np.nan,
                    "detected_left_x": detected_item["left_point"][0],
                    "detected_left_y": detected_item["left_point"][1],
                    "label_right_x": np.nan,
                    "label_right_y": np.nan,
                    "detected_right_x": detected_item["right_point"][0],
                    "detected_right_y": detected_item["right_point"][1],
                }
            )
        for detected_item in boundary_candidate_events:
            report_rows.append(
                {
                    "status": "label_boundary_candidate",
                    "label_timestamp_ms": np.nan,
                    "detected_timestamp_ms": detected_item["timestamp_ms"],
                    "time_error_ms": np.nan,
                    "left_point_error_px": np.nan,
                    "right_point_error_px": np.nan,
                    "label_left_x": np.nan,
                    "label_left_y": np.nan,
                    "detected_left_x": detected_item["left_point"][0],
                    "detected_left_y": detected_item["left_point"][1],
                    "label_right_x": np.nan,
                    "label_right_y": np.nan,
                    "detected_right_x": detected_item["right_point"][0],
                    "detected_right_y": detected_item["right_point"][1],
                }
            )
        for detected_item in confirmed_outside_window_events:
            report_rows.append(
                {
                    "status": "outside_label_window",
                    "label_timestamp_ms": np.nan,
                    "detected_timestamp_ms": detected_item["timestamp_ms"],
                    "time_error_ms": np.nan,
                    "left_point_error_px": np.nan,
                    "right_point_error_px": np.nan,
                    "label_left_x": np.nan,
                    "label_left_y": np.nan,
                    "detected_left_x": detected_item["left_point"][0],
                    "detected_left_y": detected_item["left_point"][1],
                    "label_right_x": np.nan,
                    "label_right_y": np.nan,
                    "detected_right_x": detected_item["right_point"][0],
                    "detected_right_y": detected_item["right_point"][1],
                }
            )
    
        compare_csv = os.path.join(job_output_dir, f"{stem}_label_compare.csv")
        pd.DataFrame(report_rows).to_csv(compare_csv, index=False)
        print(f"[COMPARE] Detailed report saved: {compare_csv}")
        overall_summary_rows.append(
            {
                "video_stem": stem,
                "video_path": file_path,
                "label_path": label_path,
                "tolerance_ms": match_tolerance_ms,
                "frame_based_tolerance_ms": frame_based_tolerance_ms,
                "raw_detected_count": len(selected_events),
                "compared_detected_count": len(compared_detected_events),
                "outside_window_count": len(outside_window_events),
                "label_count": len(label_events),
                "matched": len(matched_events),
                "missed_labels": len(missed_labels),
                "strict_extra_detected": len(strict_extra_detected),
                "adjusted_extra_detected": len(extra_detected),
                "full_strict_extra_detected": len(full_strict_extra_detected),
                "full_adjusted_extra_detected": len(full_extra_detected),
                "label_gap_candidates": len(gap_candidate_events),
                "label_boundary_candidates": len(boundary_candidate_events),
                "label_window_trim_removed": 0,
                "label_gap_trim_removed": 0,
                "strict_precision": strict_precision,
                "strict_recall": strict_recall,
                "strict_f1": strict_f1,
                "strict_accuracy": strict_accuracy,
                "adjusted_precision": precision,
                "adjusted_recall": recall,
                "adjusted_f1": f1,
                "adjusted_accuracy": accuracy,
                "full_strict_precision": full_strict_precision,
                "full_strict_recall": full_strict_recall,
                "full_strict_f1": full_strict_f1,
                "full_strict_accuracy": full_strict_accuracy,
                "full_adjusted_precision": full_precision,
                "full_adjusted_recall": full_recall,
                "full_adjusted_f1": full_f1,
                "full_adjusted_accuracy": full_accuracy,
                "mean_abs_time_error_ms": mean_abs_time_error_ms,
                "mean_left_point_error_px": mean_left_point_error_px,
                "mean_right_point_error_px": mean_right_point_error_px,
            }
        )
    
    if overall_summary_rows and bool(write_summary):
        overall_summary_df = pd.DataFrame(overall_summary_rows)
        overall_summary_csv = os.path.join(OUTPUT_DIR, summary_csv_name)
        overall_summary_df.to_csv(overall_summary_csv, index=False)
        print(f"\n[SUMMARY] Saved: {overall_summary_csv}")
        print(
            "[SUMMARY] strict_f1_mean="
            f"{overall_summary_df['strict_f1'].mean():.3f} "
            f"strict_f1_min={overall_summary_df['strict_f1'].min():.3f} "
            f"adjusted_f1_mean={overall_summary_df['adjusted_f1'].mean():.3f} "
            f"full_strict_f1_mean={overall_summary_df['full_strict_f1'].mean():.3f} "
            f"full_adjusted_f1_mean={overall_summary_df['full_adjusted_f1'].mean():.3f}"
        )
        print("[SUMMARY] Per video:")
        for _, row in overall_summary_df.iterrows():
            print(
                f"[SUMMARY] {row['video_stem']}: "
                f"strict_f1={row['strict_f1']:.3f} adjusted_f1={row['adjusted_f1']:.3f} "
                f"full_strict_f1={row['full_strict_f1']:.3f} "
                f"full_adjusted_f1={row['full_adjusted_f1']:.3f} "
                f"matched={int(row['matched'])}/{int(row['label_count'])} "
                f"strict_extra={int(row['strict_extra_detected'])} "
                f"adjusted_extra={int(row['adjusted_extra_detected'])} "
                f"full_strict_extra={int(row['full_strict_extra_detected'])} "
                f"full_adjusted_extra={int(row['full_adjusted_extra_detected'])} "
                f"missed={int(row['missed_labels'])}"
            )

    return overall_summary_rows


def run_with_args(runtime_args):
    return run_pipeline(
        target_stem=runtime_args.target_stem,
        target_video_path=runtime_args.video_path,
        camera_index=runtime_args.camera_index,
        use_realtime=runtime_args.use_realtime,
        require_label=runtime_args.require_label,
    )


def parse_runtime_args(argv=None):
    parser = argparse.ArgumentParser(description="Jump rope realtime-only pipeline")
    parser.add_argument(
        "--target-stem",
        default=TARGET_VIDEO_STEM,
        help="video stem in input/video (ex: 03)",
    )
    parser.add_argument(
        "--video-path",
        default=TARGET_VIDEO_PATH,
        help="explicit video path",
    )
    parser.add_argument(
        "--camera-index",
        type=int,
        default=REALTIME_CAMERA_INDEX,
        help="camera index for realtime input",
    )
    parser.add_argument(
        "--use-realtime",
        action="store_true",
        help="use camera input instead of file input",
    )
    parser.add_argument(
        "--require-label",
        action="store_true",
        help="process only videos that have matching label files",
    )
    return parser.parse_args(argv)


def main(argv=None):
    runtime_args = parse_runtime_args(argv=argv)
    return run_with_args(runtime_args)


if __name__ == "__main__":
    main()
