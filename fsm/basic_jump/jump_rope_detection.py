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


def choose_entry_backfill_tail_count(pending_candidates):
    default_tail = max(1, int(ACTIVE_ENTER_CONFIRM_TAIL_EVENTS))
    if not ENABLE_ADAPTIVE_ENTRY_BACKFILL:
        return default_tail
    if not pending_candidates:
        return default_tail

    recent_count = max(1, int(ACTIVE_ENTER_MIN_EVENTS))
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
    median_rope = float(np.median(rope_values))
    median_dual = float(np.median(dual_values)) if dual_values else 1.0
    very_low_conf_entry = (
        median_rope <= float(ENTRY_BACKFILL_VERY_LOW_ROPE_RATIO)
        and median_dual <= float(ENTRY_BACKFILL_VERY_LOW_DUAL_RATIO)
    )
    if very_low_conf_entry:
        return max(0, int(ENTRY_BACKFILL_VERY_LOW_CONF_TAIL_EVENTS))
    high_conf_entry = (
        median_rope >= float(ENTRY_BACKFILL_HIGH_CONF_MIN_ROPE_RATIO)
        and median_dual >= float(ENTRY_BACKFILL_HIGH_CONF_MIN_DUAL_RATIO)
    )
    if high_conf_entry:
        return max(1, int(ENTRY_BACKFILL_HIGH_CONF_TAIL_EVENTS))
    return max(1, int(ENTRY_BACKFILL_LOW_CONF_TAIL_EVENTS))


def prune_low_confidence_boundary_events(
    detected_events,
    max_head_drop,
    max_tail_drop,
    min_events,
    low_rope_ratio,
    low_dual_ratio,
    transition_min_rope_ratio,
    transition_min_dual_ratio,
    profile_window,
    relative_factor,
    debug_capture=None,
):
    if len(detected_events) <= max(0, int(min_events)):
        return detected_events

    events = sorted(detected_events, key=lambda event: event["frame"])
    removed_indices = set()

    def get_value(event, key, default_value):
        value = event.get(key)
        if value is None:
            return default_value
        return float(value)

    def is_absolutely_low_confidence(event):
        rope_ratio = get_value(event, "rope_ratio", 1.0)
        dual_ratio = get_value(event, "rope_dual_ratio", 1.0)
        return (
            rope_ratio <= float(low_rope_ratio)
            and dual_ratio <= float(low_dual_ratio)
        )

    def get_reference_profile(kept_indices, center_idx, direction):
        window = max(1, int(profile_window))
        if direction > 0:
            reference_indices = [idx for idx in kept_indices if idx > center_idx][:window]
        else:
            reference_indices = [idx for idx in kept_indices if idx < center_idx][-window:]
        if not reference_indices:
            return None
        rope_values = [get_value(events[idx], "rope_ratio", 1.0) for idx in reference_indices]
        dual_values = [get_value(events[idx], "rope_dual_ratio", 1.0) for idx in reference_indices]
        if not rope_values or not dual_values:
            return None
        return {
            "rope_ratio": float(np.median(rope_values)),
            "rope_dual_ratio": float(np.median(dual_values)),
            "reference_indices": reference_indices,
        }

    def is_relatively_low_confidence(event, reference_profile):
        if reference_profile is None:
            return False
        rope_ratio = get_value(event, "rope_ratio", 1.0)
        dual_ratio = get_value(event, "rope_dual_ratio", 1.0)
        ref_rope = float(reference_profile["rope_ratio"])
        ref_dual = float(reference_profile["rope_dual_ratio"])
        factor = float(relative_factor)
        return (
            rope_ratio <= (ref_rope * factor)
            and dual_ratio <= (ref_dual * factor)
            and rope_ratio < float(transition_min_rope_ratio)
            and dual_ratio < float(transition_min_dual_ratio)
        )

    def has_strong_neighbor(event):
        rope_ratio = get_value(event, "rope_ratio", 1.0)
        dual_ratio = get_value(event, "rope_dual_ratio", 1.0)
        return (
            rope_ratio >= float(transition_min_rope_ratio)
            or dual_ratio >= float(transition_min_dual_ratio)
        )

    removed_details = []
    head_removed = 0
    while head_removed < max(0, int(max_head_drop)):
        kept = [idx for idx in range(len(events)) if idx not in removed_indices]
        if len(kept) <= max(0, int(min_events)):
            break
        head_idx = kept[0]
        if len(kept) < 2:
            break
        next_idx = kept[1]
        head_event = events[head_idx]
        next_event = events[next_idx]
        reference_profile = get_reference_profile(kept, head_idx, direction=1)
        absolute_low = is_absolutely_low_confidence(head_event)
        relative_low = is_relatively_low_confidence(head_event, reference_profile)
        if not (absolute_low or relative_low):
            break
        if not has_strong_neighbor(next_event):
            break
        removed_indices.add(head_idx)
        head_removed += 1
        removed_reason = "absolute_low" if absolute_low else "relative_low"
        removed_details.append(
            {
                "side": "head",
                "frame": int(head_event["frame"]),
                "reason": removed_reason,
                "rope_ratio": get_value(head_event, "rope_ratio", 1.0),
                "rope_dual_ratio": get_value(head_event, "rope_dual_ratio", 1.0),
                "reference_rope_ratio": (
                    float(reference_profile["rope_ratio"]) if reference_profile is not None else None
                ),
                "reference_dual_ratio": (
                    float(reference_profile["rope_dual_ratio"]) if reference_profile is not None else None
                ),
            }
        )

    tail_removed = 0
    while tail_removed < max(0, int(max_tail_drop)):
        kept = [idx for idx in range(len(events)) if idx not in removed_indices]
        if len(kept) <= max(0, int(min_events)):
            break
        tail_idx = kept[-1]
        if len(kept) < 2:
            break
        prev_idx = kept[-2]
        tail_event = events[tail_idx]
        prev_event = events[prev_idx]
        reference_profile = get_reference_profile(kept, tail_idx, direction=-1)
        absolute_low = is_absolutely_low_confidence(tail_event)
        relative_low = is_relatively_low_confidence(tail_event, reference_profile)
        if not (absolute_low or relative_low):
            break
        if not has_strong_neighbor(prev_event):
            break
        removed_indices.add(tail_idx)
        tail_removed += 1
        removed_reason = "absolute_low" if absolute_low else "relative_low"
        removed_details.append(
            {
                "side": "tail",
                "frame": int(tail_event["frame"]),
                "reason": removed_reason,
                "rope_ratio": get_value(tail_event, "rope_ratio", 1.0),
                "rope_dual_ratio": get_value(tail_event, "rope_dual_ratio", 1.0),
                "reference_rope_ratio": (
                    float(reference_profile["rope_ratio"]) if reference_profile is not None else None
                ),
                "reference_dual_ratio": (
                    float(reference_profile["rope_dual_ratio"]) if reference_profile is not None else None
                ),
            }
        )

    if not removed_indices:
        filtered = events
    else:
        filtered = [event for idx, event in enumerate(events) if idx not in removed_indices]
        for idx, event in enumerate(filtered, start=1):
            event["count"] = idx

    if isinstance(debug_capture, dict):
        debug_capture["boundary_conf_prune"] = {
            "enabled": True,
            "input_events": int(len(events)),
            "head_removed": int(head_removed),
            "tail_removed": int(tail_removed),
            "removed_frames": [int(events[idx]["frame"]) for idx in sorted(removed_indices)],
            "output_events": int(len(filtered)),
            "low_rope_ratio": float(low_rope_ratio),
            "low_dual_ratio": float(low_dual_ratio),
            "transition_min_rope_ratio": float(transition_min_rope_ratio),
            "transition_min_dual_ratio": float(transition_min_dual_ratio),
            "profile_window": int(max(1, int(profile_window))),
            "relative_factor": float(relative_factor),
            "removed_details": removed_details,
        }

    return filtered


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
    }


def recover_missing_events_in_large_gaps(
    detected_events,
    recovery_candidates,
    min_jump_gap_frames,
    min_strength_ratio,
    max_insert_per_gap,
    min_gap_ratio,
    max_gap_ratio,
    target_tolerance_ratio,
    duplicate_guard_frames,
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
    debug_gap_recovery=False,
    debug_tag="",
    local_minima_candidates=None,
    debug_capture=None,
):
    if len(detected_events) < 4 or not recovery_candidates:
        return detected_events

    ordered_events = sorted(detected_events, key=lambda event: event["frame"])
    event_frames = np.array([event["frame"] for event in ordered_events], dtype=np.int32)
    intervals = np.diff(event_frames)
    if intervals.size == 0:
        return ordered_events

    median_interval = float(np.median(intervals))
    if median_interval <= 0.0:
        return ordered_events

    min_gap_frames = max(int(round(median_interval * min_gap_ratio)), min_jump_gap_frames * 2)
    max_gap_frames = max(min_gap_frames, int(round(median_interval * max_gap_ratio)))
    if max_insert_per_gap <= 0:
        return ordered_events

    best_candidate_by_frame = {}
    for candidate in recovery_candidates:
        frame = int(candidate["candidate_frame"])
        current_best = best_candidate_by_frame.get(frame)
        if current_best is None or candidate["strength_ratio"] > current_best["strength_ratio"]:
            best_candidate_by_frame[frame] = candidate
    candidates = sorted(best_candidate_by_frame.values(), key=lambda item: item["candidate_frame"])
    if not candidates:
        return ordered_events

    if debug_gap_recovery and isinstance(debug_capture, dict):
        debug_capture["recovery_overview"] = {
            "debug_tag": debug_tag,
            "events": int(len(ordered_events)),
            "recovery_candidates": int(len(candidates)),
            "median_interval_frames": float(median_interval),
            "gap_window_frames": [int(min_gap_frames), int(max_gap_frames)],
            "min_jump_gap_frames": int(min_jump_gap_frames),
        }
        debug_capture.setdefault("gaps", [])

    occupied_frames = set(int(frame) for frame in event_frames.tolist())
    new_events = []
    tolerance_frames = max(1, int(round(median_interval * target_tolerance_ratio)))
    relaxed_strength = float(min_strength_ratio) * float(GAP_RECOVERY_STRENGTH_SCALE)

    for idx in range(len(ordered_events) - 1):
        prev_event = ordered_events[idx]
        next_event = ordered_events[idx + 1]
        gap_frames = int(next_event["frame"] - prev_event["frame"])
        if gap_frames < min_gap_frames or gap_frames > max_gap_frames:
            continue

        gap_ratio = float(gap_frames) / median_interval
        estimated_missing = int(round(gap_ratio)) - 1
        insert_count = min(max_insert_per_gap, estimated_missing)
        if insert_count <= 0:
            continue
        prev_rope_ratio = float(prev_event.get("rope_ratio", 0.0))
        next_rope_ratio = float(next_event.get("rope_ratio", 0.0))
        neighbor_rope_evidence = (
            prev_rope_ratio >= float(ROPE_ACTIVE_MIN_RATIO)
            and next_rope_ratio >= float(ROPE_ACTIVE_MIN_RATIO)
        )
        effective_min_rope_ratio = float(GAP_RECOVERY_MIN_ROPE_RATIO)
        if gap_ratio >= float(GAP_RECOVERY_LARGE_GAP_RATIO) and neighbor_rope_evidence:
            effective_min_rope_ratio = min(
                effective_min_rope_ratio,
                float(GAP_RECOVERY_RELAXED_MIN_ROPE_RATIO),
            )

        gap_record = None
        if debug_gap_recovery:
            prev_ms = (prev_event["frame"] * 1000.0) / fps
            next_ms = (next_event["frame"] * 1000.0) / fps
            local_in_gap = []
            if local_minima_candidates:
                local_in_gap = [
                    item for item in local_minima_candidates
                    if prev_event["frame"] < item["candidate_frame"] < next_event["frame"]
                ]
            recovery_in_gap = [
                item for item in candidates
                if prev_event["frame"] < item["candidate_frame"] < next_event["frame"]
            ]
            gap_record = {
                "prev_frame": int(prev_event["frame"]),
                "next_frame": int(next_event["frame"]),
                "prev_ms": float(prev_ms),
                "next_ms": float(next_ms),
                "gap_frames": int(gap_frames),
                "gap_ratio": float(gap_ratio),
                "target_insert": int(insert_count),
                "local_minima_in_gap": int(len(local_in_gap)),
                "eligible_recovery_in_gap": int(len(recovery_in_gap)),
                "effective_min_rope_ratio": float(effective_min_rope_ratio),
                "neighbor_rope_evidence": bool(neighbor_rope_evidence),
                "targets": [],
            }

        target_step = float(gap_frames) / float(insert_count + 1)
        # In long gaps, strict global tolerance can block otherwise valid minima.
        # Expand target tolerance proportionally to target spacing.
        local_tolerance_frames = max(
            tolerance_frames,
            int(round(target_step * 0.8)) if insert_count >= 2 else int(round(target_step * 0.65)),
        )
        local_tolerance_frames = max(1, int(local_tolerance_frames))
        selected_candidates = []
        locally_used = set(occupied_frames)

        for target_idx in range(1, insert_count + 1):
            target_frame = float(prev_event["frame"]) + (target_idx * target_step)
            chosen = None
            chosen_dist = None
            reject_stats = {
                "low_strength": 0,
                "low_rope": 0,
                "boundary_guard": 0,
                "dup_guard": 0,
                "spacing_guard": 0,
                "target_distance": 0,
            }
            for candidate in candidates:
                candidate_frame = int(candidate["candidate_frame"])
                if candidate["strength_ratio"] < relaxed_strength:
                    reject_stats["low_strength"] += 1
                    continue
                candidate_rope_ratio = float(candidate.get("rope_ratio", 1.0))
                if candidate_rope_ratio < effective_min_rope_ratio:
                    reject_stats["low_rope"] += 1
                    continue
                if candidate_frame <= (prev_event["frame"] + min_jump_gap_frames):
                    reject_stats["boundary_guard"] += 1
                    continue
                if candidate_frame >= (next_event["frame"] - min_jump_gap_frames):
                    reject_stats["boundary_guard"] += 1
                    continue
                if any(abs(candidate_frame - used_frame) <= duplicate_guard_frames for used_frame in locally_used):
                    reject_stats["dup_guard"] += 1
                    continue
                if any(abs(candidate_frame - picked["candidate_frame"]) < min_jump_gap_frames for picked in selected_candidates):
                    reject_stats["spacing_guard"] += 1
                    continue
                distance = abs(candidate_frame - target_frame)
                if distance > local_tolerance_frames:
                    reject_stats["target_distance"] += 1
                    continue
                if chosen is None or distance < chosen_dist:
                    chosen = candidate
                    chosen_dist = distance
            if chosen is None:
                if debug_gap_recovery:
                    nearest_preview = []
                    if local_minima_candidates:
                        nearby = [
                            item for item in local_minima_candidates
                            if prev_event["frame"] < item["candidate_frame"] < next_event["frame"]
                        ]
                        nearby = sorted(
                            nearby,
                            key=lambda item: abs(float(item["candidate_frame"]) - target_frame),
                        )[:4]
                        if nearby:
                            nearest_preview = [
                                {
                                    "candidate_frame": int(item["candidate_frame"]),
                                    "strength_ratio": float(item["strength_ratio"]),
                                    "rope_ratio": float(item["rope_ratio"]),
                                }
                                for item in nearby
                            ]
                    if gap_record is not None:
                        gap_record["targets"].append(
                            {
                                "target_frame": float(target_frame),
                                "selected": False,
                                "reject": {k: int(v) for k, v in reject_stats.items()},
                                "nearest": nearest_preview,
                            }
                        )
                break
            if gap_record is not None:
                gap_record["targets"].append(
                    {
                        "target_frame": float(target_frame),
                        "selected": True,
                        "candidate_frame": int(chosen["candidate_frame"]),
                        "distance_frames": float(chosen_dist),
                        "strength_ratio": float(chosen["strength_ratio"]),
                    }
                )
            selected_candidates.append(chosen)
            locally_used.add(int(chosen["candidate_frame"]))

        if not selected_candidates:
            if gap_record is not None and isinstance(debug_capture, dict):
                gap_record["selected_candidates"] = []
                debug_capture["gaps"].append(gap_record)
            continue

        if gap_record is not None and isinstance(debug_capture, dict):
            gap_record["selected_candidates"] = [int(item["candidate_frame"]) for item in selected_candidates]
            debug_capture["gaps"].append(gap_record)

        for candidate in selected_candidates:
            built = build_detected_jump_event(
                candidate_idx=candidate["candidate_idx"],
                landing_offset_frames=landing_offset_frames,
                event_time_bias_ms=event_time_bias_ms,
                frame_numbers=frame_numbers,
                frame_timestamps_ms=frame_timestamps_ms,
                left_foot_x=left_foot_x,
                left_foot_y=left_foot_y,
                right_foot_x=right_foot_x,
                right_foot_y=right_foot_y,
                fps=fps,
                width=width,
                height=height,
                count=0,
                strength_ratio=candidate["strength_ratio"],
                rope_ratio=float(candidate.get("rope_ratio", 1.0)),
                rope_dual_ratio=float(candidate.get("rope_dual_ratio", 1.0)),
            )
            new_events.append(built)
            occupied_frames.add(int(built["frame"]))

    if not new_events:
        if debug_gap_recovery and isinstance(debug_capture, dict):
            debug_capture["recovered_count"] = 0
            debug_capture["added_frames"] = []
        return ordered_events

    recovered_events = sorted(ordered_events + new_events, key=lambda event: event["frame"])
    if debug_gap_recovery and isinstance(debug_capture, dict):
        added_frames = [int(item["frame"]) for item in new_events]
        debug_capture["recovered_count"] = int(len(new_events))
        debug_capture["added_frames"] = added_frames
    for i, event in enumerate(recovered_events, start=1):
        event["count"] = i
    return recovered_events


def recover_high_cadence_single_miss_gaps(
    detected_events,
    min_jump_gap_frames,
    min_ref_interval_frames,
    max_ref_interval_frames,
    min_gap_ratio,
    max_gap_ratio,
    round_tolerance,
    context_window,
    max_context_cv,
    duplicate_guard_frames,
    min_total_events=0,
    debug_key="high_cadence_gap_interp",
    debug_capture=None,
):
    min_total_events = max(0, int(min_total_events))
    if len(detected_events) < max(6, min_total_events):
        return detected_events

    ordered_events = sorted(detected_events, key=lambda event: event["frame"])
    frames = np.array([int(event["frame"]) for event in ordered_events], dtype=np.int32)
    intervals = np.diff(frames).astype(np.float32)
    if intervals.size < 5:
        return ordered_events

    def safe_metric(event, key, default_value):
        value = event.get(key, default_value)
        try:
            casted = float(value)
        except (TypeError, ValueError):
            return float(default_value)
        if not np.isfinite(casted):
            return float(default_value)
        return casted

    def lerp_point(point_a, point_b, alpha):
        ax = safe_metric({"v": point_a[0]}, "v", np.nan)
        ay = safe_metric({"v": point_a[1]}, "v", np.nan)
        bx = safe_metric({"v": point_b[0]}, "v", np.nan)
        by = safe_metric({"v": point_b[1]}, "v", np.nan)
        if not np.isfinite(ax) or not np.isfinite(ay):
            return (float(bx), float(by))
        if not np.isfinite(bx) or not np.isfinite(by):
            return (float(ax), float(ay))
        return (
            float(ax + ((bx - ax) * alpha)),
            float(ay + ((by - ay) * alpha)),
        )

    context_window = max(1, int(context_window))
    duplicate_guard_frames = max(0, int(duplicate_guard_frames))
    occupied_frames = set(int(frame) for frame in frames.tolist())
    new_events = []
    inserted_gap_frames = []

    for idx, gap_frames in enumerate(intervals.tolist()):
        left_start = max(0, idx - context_window)
        left_context = intervals[left_start:idx]
        right_end = min(len(intervals), idx + 1 + context_window)
        right_context = intervals[idx + 1:right_end]
        context_size = int(left_context.size + right_context.size)
        if context_size < max(2, context_window):
            continue

        context_intervals = np.concatenate((left_context, right_context)).astype(np.float32)
        context_mean = float(np.mean(context_intervals))
        if context_mean <= 0.0:
            continue
        context_cv = float(np.std(context_intervals) / context_mean)
        if context_cv > float(max_context_cv):
            continue

        ref_interval = float(np.median(context_intervals))
        if (
            ref_interval <= 0.0
            or ref_interval < float(min_ref_interval_frames)
            or ref_interval > float(max_ref_interval_frames)
        ):
            continue

        gap_ratio = float(gap_frames) / ref_interval
        if gap_ratio < float(min_gap_ratio) or gap_ratio > float(max_gap_ratio):
            continue

        rounded_ratio = int(round(gap_ratio))
        missing_count = rounded_ratio - 1
        if missing_count != 1:
            continue
        if abs(gap_ratio - float(rounded_ratio)) > float(round_tolerance):
            continue

        prev_event = ordered_events[idx]
        next_event = ordered_events[idx + 1]
        prev_frame = int(prev_event["frame"])
        next_frame = int(next_event["frame"])
        if next_frame <= prev_frame:
            continue

        target_frame = int(round((prev_frame + next_frame) / 2.0))
        if (target_frame - prev_frame) < int(min_jump_gap_frames):
            continue
        if (next_frame - target_frame) < int(min_jump_gap_frames):
            continue
        if any(abs(target_frame - used) <= duplicate_guard_frames for used in occupied_frames):
            continue

        alpha = float(target_frame - prev_frame) / float(next_frame - prev_frame)
        prev_ts = int(prev_event.get("timestamp_ms", prev_frame))
        next_ts = int(next_event.get("timestamp_ms", next_frame))
        target_ts = int(round(prev_ts + ((next_ts - prev_ts) * alpha)))

        prev_left = prev_event.get("left_point", (np.nan, np.nan))
        next_left = next_event.get("left_point", (np.nan, np.nan))
        prev_right = prev_event.get("right_point", (np.nan, np.nan))
        next_right = next_event.get("right_point", (np.nan, np.nan))

        new_events.append(
            {
                "count": 0,
                "frame": target_frame,
                "timestamp_ms": target_ts,
                "left_point": lerp_point(prev_left, next_left, alpha),
                "right_point": lerp_point(prev_right, next_right, alpha),
                "strength_ratio": min(
                    safe_metric(prev_event, "strength_ratio", 1.0),
                    safe_metric(next_event, "strength_ratio", 1.0),
                ),
                "rope_ratio": min(
                    safe_metric(prev_event, "rope_ratio", 1.0),
                    safe_metric(next_event, "rope_ratio", 1.0),
                ),
                "rope_dual_ratio": min(
                    safe_metric(prev_event, "rope_dual_ratio", 1.0),
                    safe_metric(next_event, "rope_dual_ratio", 1.0),
                ),
            }
        )
        occupied_frames.add(target_frame)
        inserted_gap_frames.append((prev_frame, next_frame, target_frame))

    if not new_events:
        if isinstance(debug_capture, dict):
            debug_capture[debug_key] = {
                "enabled": True,
                "recovered_count": 0,
            }
        return ordered_events

    recovered_events = sorted(ordered_events + new_events, key=lambda event: event["frame"])
    for idx, event in enumerate(recovered_events, start=1):
        event["count"] = idx

    if isinstance(debug_capture, dict):
        debug_capture[debug_key] = {
            "enabled": True,
            "recovered_count": int(len(new_events)),
            "inserted_gap_frames": inserted_gap_frames,
        }

    return recovered_events


def prune_small_edge_segments(
    detected_events,
    fps,
    split_gap_ratio,
    split_gap_seconds,
    max_edge_events,
    min_main_events,
    debug_capture=None,
):
    if len(detected_events) < (max_edge_events + min_main_events + 1):
        return detected_events

    ordered_events = sorted(detected_events, key=lambda event: event["frame"])
    frames = np.array([event["frame"] for event in ordered_events], dtype=np.int32)
    intervals = np.diff(frames)
    if intervals.size == 0:
        return ordered_events

    median_interval = float(np.median(intervals))
    if median_interval <= 0.0:
        return ordered_events

    split_gap_frames = max(
        int(round(median_interval * split_gap_ratio)),
        int(round(float(fps) * split_gap_seconds)),
    )
    split_points = [0]
    for idx, gap in enumerate(intervals):
        if int(gap) >= split_gap_frames:
            split_points.append(idx + 1)
    split_points.append(len(ordered_events))
    split_points = sorted(set(split_points))
    if len(split_points) <= 2:
        return ordered_events

    segments = []
    for i in range(len(split_points) - 1):
        start = int(split_points[i])
        end = int(split_points[i + 1])
        if end > start:
            segments.append((start, end))
    if len(segments) <= 1:
        return ordered_events

    removed_segment_indices = set()
    if len(segments) >= 2:
        first_len = segments[0][1] - segments[0][0]
        second_len = segments[1][1] - segments[1][0]
        if first_len <= max_edge_events and second_len >= min_main_events:
            removed_segment_indices.add(0)
    if len(segments) >= 2:
        last_idx = len(segments) - 1
        prev_idx = len(segments) - 2
        last_len = segments[last_idx][1] - segments[last_idx][0]
        prev_len = segments[prev_idx][1] - segments[prev_idx][0]
        if last_len <= max_edge_events and prev_len >= min_main_events:
            removed_segment_indices.add(last_idx)

    if not removed_segment_indices:
        return ordered_events

    keep_indices = set()
    removed_frames = []
    for seg_idx, (start, end) in enumerate(segments):
        seg_range = list(range(start, end))
        if seg_idx in removed_segment_indices:
            removed_frames.extend(int(ordered_events[item]["frame"]) for item in seg_range)
        else:
            keep_indices.update(seg_range)

    filtered_events = [ordered_events[i] for i in range(len(ordered_events)) if i in keep_indices]
    filtered_events = sorted(filtered_events, key=lambda event: event["frame"])
    for i, event in enumerate(filtered_events, start=1):
        event["count"] = i

    if isinstance(debug_capture, dict):
        debug_capture["edge_segment_prune"] = {
            "enabled": True,
            "input_events": int(len(ordered_events)),
            "median_interval_frames": float(median_interval),
            "split_gap_frames": int(split_gap_frames),
            "segments": [
                {
                    "index": int(seg_idx),
                    "start_frame": int(ordered_events[start]["frame"]),
                    "end_frame": int(ordered_events[end - 1]["frame"]),
                    "length": int(end - start),
                    "removed": bool(seg_idx in removed_segment_indices),
                }
                for seg_idx, (start, end) in enumerate(segments)
            ],
            "removed_frames": sorted(removed_frames),
            "output_events": int(len(filtered_events)),
        }

    return filtered_events


def detect_jump_events_offline(
    hip_series,
    left_foot_x,
    left_foot_y,
    right_foot_x,
    right_foot_y,
    frame_numbers,
    frame_timestamps_ms,
    fps,
    width,
    height,
    threshold_gain,
    minima_lag_frames,
    prominence_window,
    min_jump_gap_seconds,
    landing_offset_ms,
    event_time_bias_ms,
    min_strength_ratio,
    rope_flag_series=None,
    rope_dual_flag_series=None,
    adaptive_threshold_series=None,
    rope_ratio_series=None,
    rope_dual_ratio_series=None,
    debug_gap_recovery=False,
    debug_tag="",
    debug_capture=None,
):
    detected_events = []
    last_processed_minima_idx = -1
    last_jump_frame = -10**9
    min_jump_gap_frames = max(4, int(round(fps * min_jump_gap_seconds)))
    landing_offset_frames = int(round((landing_offset_ms / 1000.0) * fps))
    enter_window_frames = max(1, int(round(fps * ACTIVE_ENTER_WINDOW_SECONDS)))
    active_enter_max_gap_frames = max(1, int(round(fps * ACTIVE_ENTER_MAX_GAP_SECONDS)))
    candidate_exit_idle_frames = max(enter_window_frames, int(round(fps * ACTIVE_EXIT_IDLE_SECONDS)))
    startup_lockout_frames = max(0, int(round(fps * STARTUP_LOCKOUT_SECONDS)))
    rope_active_window_frames = max(1, int(round(fps * ROPE_ACTIVE_WINDOW_SECONDS)))
    rope_exit_idle_frames = max(1, int(round(fps * ROPE_EXIT_IDLE_SECONDS)))
    is_active = False
    pending_candidates = []
    last_valid_candidate_frame = -10**9
    if rope_flag_series is None:
        rope_flag_series = []
    if rope_dual_flag_series is None:
        rope_dual_flag_series = []
    last_rope_active_frame = -10**9
    recovery_candidates = []
    if frame_timestamps_ms is None:
        frame_timestamps_ms = []
    local_minima_candidates = [] if debug_gap_recovery else None

    def keep_recent_pending(current_candidate_frame):
        min_frame = current_candidate_frame - enter_window_frames
        return [
            item for item in pending_candidates
            if item["candidate_frame"] >= min_frame
        ]

    for current_idx in range(len(hip_series)):
        current_frame = frame_numbers[current_idx] if current_idx < len(frame_numbers) else current_idx
        if rope_flag_series and current_idx < len(rope_flag_series) and rope_flag_series[current_idx]:
            last_rope_active_frame = current_frame

        if is_active and (current_frame - last_valid_candidate_frame) > candidate_exit_idle_frames:
            is_active = False
            pending_candidates = []
        if is_active and rope_flag_series and (current_frame - last_rope_active_frame) > rope_exit_idle_frames:
            is_active = False
            pending_candidates = []

        candidate_idx = current_idx - minima_lag_frames
        if (
            candidate_idx < 1
            or candidate_idx <= last_processed_minima_idx
            or (candidate_idx + 1) >= len(hip_series)
        ):
            continue

        adaptive_base = None
        if adaptive_threshold_series and current_idx < len(adaptive_threshold_series):
            adaptive_base = adaptive_threshold_series[current_idx]
        else:
            adaptive_base = get_adaptive_jump_threshold_at_index(hip_series, current_idx)
        adaptive_threshold = max(adaptive_base * threshold_gain, ADAPTIVE_THRESHOLD_MIN * 0.75)
        left_start = max(0, candidate_idx - prominence_window)
        right_end = min(len(hip_series), candidate_idx + 1 + prominence_window)
        left_neighbors = hip_series[left_start:candidate_idx]
        right_neighbors = hip_series[candidate_idx + 1:right_end]

        if left_neighbors and right_neighbors:
            is_local_min = (
                hip_series[candidate_idx] < hip_series[candidate_idx - 1]
                and hip_series[candidate_idx] <= hip_series[candidate_idx + 1]
            )
            prominence_left = max(left_neighbors) - hip_series[candidate_idx]
            prominence_right = max(right_neighbors) - hip_series[candidate_idx]
            prominence = min(prominence_left, prominence_right)
            strength_ratio = prominence / max(adaptive_threshold, 1e-9)
            candidate_frame = frame_numbers[candidate_idx]
            if candidate_frame < startup_lockout_frames:
                last_processed_minima_idx = candidate_idx
                continue
            is_far_enough = (candidate_frame - last_jump_frame) >= min_jump_gap_frames
            rope_ratio = 1.0
            rope_dual_ratio = 1.0
            if rope_ratio_series and candidate_idx < len(rope_ratio_series):
                rope_ratio = float(rope_ratio_series[candidate_idx])
            elif rope_flag_series:
                rope_ratio = get_true_ratio(rope_flag_series, candidate_idx, rope_active_window_frames)
            if rope_dual_ratio_series and candidate_idx < len(rope_dual_ratio_series):
                rope_dual_ratio = float(rope_dual_ratio_series[candidate_idx])
            elif rope_dual_flag_series:
                rope_dual_ratio = get_true_ratio(rope_dual_flag_series, candidate_idx, rope_active_window_frames)
            if debug_gap_recovery and is_local_min:
                local_minima_candidates.append(
                    {
                        "candidate_idx": candidate_idx,
                        "candidate_frame": candidate_frame,
                        "strength_ratio": float(strength_ratio),
                        "rope_ratio": float(rope_ratio),
                        "rope_dual_ratio": float(rope_dual_ratio),
                    }
                )
            if is_active:
                rope_active = rope_ratio >= ROPE_ACTIVE_MIN_RATIO
            else:
                entry_dual_ok = (
                    ROPE_ENTRY_DUAL_MIN_RATIO <= 0.0
                    or rope_dual_ratio >= ROPE_ENTRY_DUAL_MIN_RATIO
                )
                rope_active = (
                    rope_ratio >= ROPE_ENTRY_MIN_RATIO
                    and entry_dual_ok
                )

            if is_local_min and strength_ratio >= (float(min_strength_ratio) * float(GAP_RECOVERY_STRENGTH_SCALE)):
                recovery_candidates.append(
                    {
                        "candidate_idx": candidate_idx,
                        "candidate_frame": candidate_frame,
                        "strength_ratio": float(strength_ratio),
                        "rope_ratio": float(rope_ratio),
                        "rope_dual_ratio": float(rope_dual_ratio),
                    }
                )

            if is_local_min and strength_ratio >= min_strength_ratio and is_far_enough and rope_active:
                last_jump_frame = candidate_frame
                last_valid_candidate_frame = candidate_frame
                candidate_info = {
                    "candidate_idx": candidate_idx,
                    "candidate_frame": candidate_frame,
                    "strength_ratio": strength_ratio,
                    "rope_ratio": float(rope_ratio),
                    "rope_dual_ratio": float(rope_dual_ratio),
                }

                if is_active:
                    detected_events.append(
                        build_detected_jump_event(
                            candidate_idx=candidate_idx,
                            landing_offset_frames=landing_offset_frames,
                            event_time_bias_ms=event_time_bias_ms,
                            frame_numbers=frame_numbers,
                            frame_timestamps_ms=frame_timestamps_ms,
                            left_foot_x=left_foot_x,
                            left_foot_y=left_foot_y,
                            right_foot_x=right_foot_x,
                            right_foot_y=right_foot_y,
                            fps=fps,
                            width=width,
                            height=height,
                            count=len(detected_events) + 1,
                            strength_ratio=strength_ratio,
                            rope_ratio=float(rope_ratio),
                            rope_dual_ratio=float(rope_dual_ratio),
                        )
                    )
                else:
                    pending_candidates.append(candidate_info)
                    pending_candidates = keep_recent_pending(candidate_frame)
                    if has_stable_entry_cadence(
                        pending_candidates,
                        min_events=ACTIVE_ENTER_MIN_EVENTS,
                        max_gap_frames=active_enter_max_gap_frames,
                        max_cv=ACTIVE_ENTER_CADENCE_MAX_CV,
                    ):
                        is_active = True
                        tail_n = choose_entry_backfill_tail_count(pending_candidates)
                        if tail_n > 0:
                            for confirmed in pending_candidates[-tail_n:]:
                                detected_events.append(
                                    build_detected_jump_event(
                                        candidate_idx=confirmed["candidate_idx"],
                                        landing_offset_frames=landing_offset_frames,
                                        event_time_bias_ms=event_time_bias_ms,
                                        frame_numbers=frame_numbers,
                                        frame_timestamps_ms=frame_timestamps_ms,
                                        left_foot_x=left_foot_x,
                                        left_foot_y=left_foot_y,
                                        right_foot_x=right_foot_x,
                                        right_foot_y=right_foot_y,
                                        fps=fps,
                                        width=width,
                                        height=height,
                                        count=len(detected_events) + 1,
                                        strength_ratio=confirmed["strength_ratio"],
                                        rope_ratio=float(confirmed.get("rope_ratio", rope_ratio)),
                                        rope_dual_ratio=float(confirmed.get("rope_dual_ratio", rope_dual_ratio)),
                                    )
                                )
                        pending_candidates = []

        last_processed_minima_idx = candidate_idx

    if ENABLE_GAP_RECOVERY:
        if debug_gap_recovery and isinstance(debug_capture, dict):
            debug_capture["local_minima_candidates"] = int(len(local_minima_candidates or []))
            debug_capture["recovery_candidates"] = int(len(recovery_candidates))
            debug_capture["recovery_strength_threshold"] = float(float(min_strength_ratio) * float(GAP_RECOVERY_STRENGTH_SCALE))
            debug_capture["recovery_min_rope_ratio"] = float(GAP_RECOVERY_MIN_ROPE_RATIO)
            debug_capture["recovery_large_gap_ratio"] = float(GAP_RECOVERY_LARGE_GAP_RATIO)
            debug_capture["recovery_relaxed_min_rope_ratio"] = float(GAP_RECOVERY_RELAXED_MIN_ROPE_RATIO)
        detected_events = recover_missing_events_in_large_gaps(
            detected_events=detected_events,
            recovery_candidates=recovery_candidates,
            min_jump_gap_frames=min_jump_gap_frames,
            min_strength_ratio=min_strength_ratio,
            max_insert_per_gap=GAP_RECOVERY_MAX_INSERT_PER_GAP,
            min_gap_ratio=GAP_RECOVERY_MIN_GAP_RATIO,
            max_gap_ratio=GAP_RECOVERY_MAX_GAP_RATIO,
            target_tolerance_ratio=GAP_RECOVERY_TARGET_TOLERANCE_RATIO,
            duplicate_guard_frames=GAP_RECOVERY_DUPLICATE_GUARD_FRAMES,
            landing_offset_frames=landing_offset_frames,
            event_time_bias_ms=event_time_bias_ms,
            frame_numbers=frame_numbers,
            frame_timestamps_ms=frame_timestamps_ms,
            left_foot_x=left_foot_x,
            left_foot_y=left_foot_y,
            right_foot_x=right_foot_x,
            right_foot_y=right_foot_y,
            fps=fps,
            width=width,
            height=height,
            debug_gap_recovery=debug_gap_recovery,
            debug_tag=debug_tag,
            local_minima_candidates=local_minima_candidates,
            debug_capture=debug_capture,
        )

    if ENABLE_HIGH_CADENCE_GAP_INTERP:
        detected_events = recover_high_cadence_single_miss_gaps(
            detected_events=detected_events,
            min_jump_gap_frames=min_jump_gap_frames,
            min_ref_interval_frames=0.0,
            max_ref_interval_frames=HIGH_CADENCE_GAP_INTERP_MAX_REF_INTERVAL_FRAMES,
            min_gap_ratio=HIGH_CADENCE_GAP_INTERP_MIN_GAP_RATIO,
            max_gap_ratio=HIGH_CADENCE_GAP_INTERP_MAX_GAP_RATIO,
            round_tolerance=HIGH_CADENCE_GAP_INTERP_ROUND_TOL,
            context_window=HIGH_CADENCE_GAP_INTERP_CONTEXT,
            max_context_cv=HIGH_CADENCE_GAP_INTERP_MAX_CV,
            duplicate_guard_frames=HIGH_CADENCE_GAP_INTERP_DUPLICATE_GUARD_FRAMES,
            min_total_events=0,
            debug_key="high_cadence_gap_interp",
            debug_capture=debug_capture if isinstance(debug_capture, dict) else None,
        )

    if ENABLE_LONG_RUN_GAP_INTERP:
        detected_events = recover_high_cadence_single_miss_gaps(
            detected_events=detected_events,
            min_jump_gap_frames=min_jump_gap_frames,
            min_ref_interval_frames=LONG_RUN_GAP_INTERP_MIN_REF_INTERVAL_FRAMES,
            max_ref_interval_frames=LONG_RUN_GAP_INTERP_MAX_REF_INTERVAL_FRAMES,
            min_gap_ratio=LONG_RUN_GAP_INTERP_MIN_GAP_RATIO,
            max_gap_ratio=LONG_RUN_GAP_INTERP_MAX_GAP_RATIO,
            round_tolerance=LONG_RUN_GAP_INTERP_ROUND_TOL,
            context_window=LONG_RUN_GAP_INTERP_CONTEXT,
            max_context_cv=LONG_RUN_GAP_INTERP_MAX_CV,
            duplicate_guard_frames=LONG_RUN_GAP_INTERP_DUPLICATE_GUARD_FRAMES,
            min_total_events=LONG_RUN_GAP_INTERP_MIN_TOTAL_EVENTS,
            debug_key="long_run_gap_interp",
            debug_capture=debug_capture if isinstance(debug_capture, dict) else None,
        )

    if ENABLE_EDGE_SEGMENT_PRUNE:
        detected_events = prune_small_edge_segments(
            detected_events=detected_events,
            fps=fps,
            split_gap_ratio=EDGE_SEGMENT_SPLIT_GAP_RATIO,
            split_gap_seconds=EDGE_SEGMENT_SPLIT_GAP_SECONDS,
            max_edge_events=EDGE_SEGMENT_MAX_EVENTS,
            min_main_events=EDGE_SEGMENT_MIN_MAIN_EVENTS,
            debug_capture=debug_capture if isinstance(debug_capture, dict) else None,
        )

    if ENABLE_BOUNDARY_CONF_PRUNE:
        detected_events = prune_low_confidence_boundary_events(
            detected_events=detected_events,
            max_head_drop=BOUNDARY_HEAD_MAX_DROP,
            max_tail_drop=BOUNDARY_TAIL_MAX_DROP,
            min_events=BOUNDARY_PRUNE_MIN_EVENTS,
            low_rope_ratio=BOUNDARY_LOW_ROPE_RATIO,
            low_dual_ratio=BOUNDARY_LOW_DUAL_RATIO,
            transition_min_rope_ratio=BOUNDARY_TRANSITION_MIN_ROPE_RATIO,
            transition_min_dual_ratio=BOUNDARY_TRANSITION_MIN_DUAL_RATIO,
            profile_window=BOUNDARY_PROFILE_WINDOW,
            relative_factor=BOUNDARY_RELATIVE_FACTOR,
            debug_capture=debug_capture if isinstance(debug_capture, dict) else None,
        )

    return detected_events

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
