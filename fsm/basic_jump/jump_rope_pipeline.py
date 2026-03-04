import os
import time

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd

from jump_rope_detection import (
    build_detected_jump_event,
    choose_entry_backfill_tail_count,
    detect_jump_events_offline,
    evaluate_foot_motion_signature,
    get_adaptive_jump_threshold,
    get_true_ratio,
    has_stable_entry_cadence,
    prune_low_confidence_boundary_events,
    prune_low_dual_ratio_segments,
    prune_small_edge_segments,
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


def has_reliable_lower_body(landmarks, min_visibility):
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
    # Require hips plus at least one lower-limb keypoint.
    return (left_reliable + right_reliable) >= 1


def get_live_refined_overlay_count(
    detected_jump_events,
    current_frame_idx,
    fps,
    strict_guard_mode,
):
    if not detected_jump_events:
        return 0

    stable_tail_frames = max(0, int(round(float(REALTIME_OVERLAY_STABLE_TAIL_SECONDS) * float(fps))))
    stable_frame_cutoff = int(current_frame_idx) - stable_tail_frames
    stable_events = [
        dict(event)
        for event in detected_jump_events
        if int(event.get("frame", -1)) <= stable_frame_cutoff
    ]
    if not stable_events:
        return 0

    stable_events = sorted(stable_events, key=lambda event: int(event["frame"]))
    for idx, event in enumerate(stable_events, start=1):
        event["count"] = idx

    if ENABLE_EDGE_SEGMENT_PRUNE:
        stable_events = prune_small_edge_segments(
            detected_events=stable_events,
            fps=fps,
            split_gap_ratio=EDGE_SEGMENT_SPLIT_GAP_RATIO,
            split_gap_seconds=EDGE_SEGMENT_SPLIT_GAP_SECONDS,
            max_edge_events=EDGE_SEGMENT_MAX_EVENTS,
            min_main_events=EDGE_SEGMENT_MIN_MAIN_EVENTS,
            debug_capture=None,
        )

    if bool(strict_guard_mode) and ENABLE_SEGMENT_DUAL_PRUNE:
        stable_events = prune_low_dual_ratio_segments(
            detected_events=stable_events,
            fps=fps,
            split_gap_ratio=EDGE_SEGMENT_SPLIT_GAP_RATIO,
            split_gap_seconds=EDGE_SEGMENT_SPLIT_GAP_SECONDS,
            min_dual_median=SEGMENT_DUAL_PRUNE_MIN_MEDIAN,
            max_segment_events_to_prune=SEGMENT_DUAL_PRUNE_MAX_SEGMENT_EVENTS,
            min_segment_events_to_keep=SEGMENT_MIN_EVENTS,
            enable_segment_cadence_prune=ENABLE_SEGMENT_CADENCE_PRUNE,
            segment_cadence_max_cv=SEGMENT_CADENCE_MAX_CV,
            short_segment_cadence_max_events=SHORT_SEGMENT_CADENCE_MAX_EVENTS,
            short_segment_cadence_max_cv=SHORT_SEGMENT_CADENCE_MAX_CV,
            enable_segment_dual_cadence_override=ENABLE_SEGMENT_DUAL_CADENCE_OVERRIDE,
            segment_dual_cadence_override_min_events=SEGMENT_DUAL_CADENCE_OVERRIDE_MIN_EVENTS,
            segment_dual_cadence_override_max_cv=SEGMENT_DUAL_CADENCE_OVERRIDE_MAX_CV,
            debug_capture=None,
        )

    if ENABLE_BOUNDARY_CONF_PRUNE:
        stable_events = prune_low_confidence_boundary_events(
            detected_events=stable_events,
            max_head_drop=BOUNDARY_HEAD_MAX_DROP,
            max_tail_drop=BOUNDARY_TAIL_MAX_DROP,
            min_events=BOUNDARY_PRUNE_MIN_EVENTS,
            low_rope_ratio=BOUNDARY_LOW_ROPE_RATIO,
            low_dual_ratio=BOUNDARY_LOW_DUAL_RATIO,
            transition_min_rope_ratio=BOUNDARY_TRANSITION_MIN_ROPE_RATIO,
            transition_min_dual_ratio=BOUNDARY_TRANSITION_MIN_DUAL_RATIO,
            profile_window=BOUNDARY_PROFILE_WINDOW,
            relative_factor=BOUNDARY_RELATIVE_FACTOR,
            debug_capture=None,
        )

    if bool(strict_guard_mode) and int(SESSION_MIN_EVENTS) > 1:
        if len(stable_events) < int(SESSION_MIN_EVENTS):
            stable_events = []

    return int(len(stable_events))


def _sanitize_path_fragment(text):
    if text is None:
        return ""
    cleaned = "".join(
        ch if (ch.isascii() and (ch.isalnum() or ch in {"-", "_", "."})) else "_"
        for ch in str(text).strip()
    )
    return cleaned.strip("._-")


def run_pipeline(
    mode,
    target_stem=None,
    target_video_path=None,
    camera_index=0,
    realtime_fps_log_interval_s=1.0,
    realtime_demo_log=False,
    realtime_demo_log_dir=None,
    realtime_demo_tag="",
    realtime_demo_save_raw=False,
):
    run_mode = (mode or "labeled").strip().lower()
    realtime_fps_log_interval_s = max(0.0, float(realtime_fps_log_interval_s))
    realtime_demo_log = bool(realtime_demo_log)
    realtime_demo_save_raw = bool(realtime_demo_save_raw)
    realtime_demo_tag = _sanitize_path_fragment(realtime_demo_tag)
    if realtime_demo_log_dir:
        realtime_demo_log_dir = os.path.abspath(str(realtime_demo_log_dir))
    else:
        realtime_demo_log_dir = ""
    summary_csv_name = get_summary_csv_name(run_mode)

    jobs = build_jobs_for_mode(
        mode=run_mode,
        target_stem=target_stem,
        target_video_path=target_video_path,
        camera_index=camera_index,
    )
    if not jobs:
        if run_mode == "labeled":
            print(f"[WARN] No labeled video jobs found in {INPUT_VIDEO_DIR} and {INPUT_LABEL_DIR}")
        elif run_mode == "video":
            print(f"[WARN] No video jobs found in {INPUT_VIDEO_DIR}")
        else:
            print(f"[WARN] No realtime job was created (camera index={camera_index})")

    overall_summary_rows = []
    for job in jobs:
        stem = job["stem"]
        file_path = job["video_path"]
        capture_source = job.get("capture_source", file_path)
        is_realtime = bool(job.get("is_realtime", False))
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
                f"segment_dual_prune={ENABLE_SEGMENT_DUAL_PRUNE} "
                f"segment_dual_min={SEGMENT_DUAL_PRUNE_MIN_MEDIAN:.2f} "
                f"segment_dual_max_events={SEGMENT_DUAL_PRUNE_MAX_SEGMENT_EVENTS} "
                f"segment_min_events={SEGMENT_MIN_EVENTS} "
                f"segment_cadence_prune={ENABLE_SEGMENT_CADENCE_PRUNE} "
                f"segment_cadence_max_cv={SEGMENT_CADENCE_MAX_CV:.2f} "
                f"short_segment_max_events={SHORT_SEGMENT_CADENCE_MAX_EVENTS} "
                f"short_segment_max_cv={SHORT_SEGMENT_CADENCE_MAX_CV:.2f} "
                f"dual_cadence_override={ENABLE_SEGMENT_DUAL_CADENCE_OVERRIDE} "
                f"dual_cadence_min_events={SEGMENT_DUAL_CADENCE_OVERRIDE_MIN_EVENTS} "
                f"dual_cadence_max_cv={SEGMENT_DUAL_CADENCE_OVERRIDE_MAX_CV:.2f} "
                f"live_overlay_refined={LIVE_OVERLAY_REFINED_COUNT} "
                f"overlay_stable_tail_s={REALTIME_OVERLAY_STABLE_TAIL_SECONDS:.2f} "
                f"session_min_events={SESSION_MIN_EVENTS}"
            )
    
        job_output_dir = OUTPUT_DIR
        demo_session_dir = ""
        frame_log_csv_path = ""
        if is_realtime and realtime_demo_log:
            out_stamp = time.strftime("%Y%m%d_%H%M%S")
            demo_root_dir = realtime_demo_log_dir or os.path.join(OUTPUT_DIR, "realtime_demo_logs")
            tag_prefix = f"{realtime_demo_tag}_" if realtime_demo_tag else ""
            session_name = f"{tag_prefix}{stem}_{out_stamp}"
            demo_session_dir = os.path.join(demo_root_dir, session_name)
            job_output_dir = demo_session_dir
            frame_log_csv_path = os.path.join(demo_session_dir, "frame_log.csv")
            print(f"[LOG] realtime demo session: {demo_session_dir}")

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
        if demo_session_dir and realtime_demo_save_raw:
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
                            loose_foot_motion_ok = (
                                (foot_prominence is not None)
                                and (float(foot_prominence) >= float(STRICT_ACTIVE_LOOSE_FOOT_PROMINENCE))
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
                                        rope_ratio >= float(STRICT_OVERRIDE_ACTIVE_MIN_ROPE_RATIO)
                                        and rope_dual_ratio >= float(STRICT_OVERRIDE_ACTIVE_MIN_DUAL_RATIO)
                                    )
                                    foot_motion_ok = bool(
                                        relaxed_motion_gate_ok
                                        or active_history_ok
                                        or strong_active_override
                                    )
                                else:
                                    strong_entry_override = (
                                        rope_ratio >= float(STRICT_OVERRIDE_ENTRY_MIN_ROPE_RATIO)
                                        and rope_dual_ratio >= float(STRICT_OVERRIDE_ENTRY_MIN_DUAL_RATIO)
                                    )
                                    foot_motion_ok = bool(entry_motion_gate_ok or strong_entry_override)
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
                                if strict_guard_mode and not candidate_rope_flag:
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
                                    rope_active = bool(
                                        rope_active
                                        and candidate_rope_flag
                                        and (
                                            candidate_dual_rope_flag
                                            or rope_dual_ratio >= float(STRICT_OVERRIDE_ENTRY_MIN_DUAL_RATIO)
                                        )
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
                                                    )
                                                )
                                        pending_candidates = []
                        last_processed_minima_idx = candidate_idx
    
                    raw_jump_counter = int(jump_counter)
                    overlay_jump_counter = raw_jump_counter
                    overlay_counter_mode = "raw"
                    if LIVE_OVERLAY_REFINED_COUNT:
                        overlay_jump_counter = get_live_refined_overlay_count(
                            detected_jump_events=detected_jump_events,
                            current_frame_idx=current_frame_idx,
                            fps=fps,
                            strict_guard_mode=strict_guard_mode,
                        )
                        overlay_counter_mode = "refined"
                    ankle_counter = int(overlay_jump_counter)
    
                    current_ts_ms = current_frame_ts_ms + DISPLAY_TIME_ADVANCE_MS
                    while (
                        label_progress_idx < len(label_events)
                        and label_events[label_progress_idx]["timestamp_ms"] <= current_ts_ms
                    ):
                        label_progress_idx += 1
                    count_delta = ankle_counter - label_progress_idx
                    raw_count_delta = raw_jump_counter - label_progress_idx
                    delta_color = (0, 180, 0) if count_delta == 0 else (0, 0, 255)
    
                    # Counter overlay
                    jump_title = "Jumps"
                    if LIVE_OVERLAY_REFINED_COUNT:
                        jump_title = "Jumps*"
                    cv2.putText(image, jump_title, (image.shape[1] - 130, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2,
                                cv2.LINE_AA)
                    cv2.putText(image, str(ankle_counter),
                                (image.shape[1] - 40, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
                    cv2.putText(image, "Label", (image.shape[1] - 130, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 120, 0), 2,
                                cv2.LINE_AA)
                    cv2.putText(image, str(label_progress_idx),
                                (image.shape[1] - 40, 62),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 120, 0), 2, cv2.LINE_AA)
                    cv2.putText(image, f"Delta {count_delta:+d}",
                                (image.shape[1] - 220, 94),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, delta_color, 2, cv2.LINE_AA)
                    if LIVE_OVERLAY_REFINED_COUNT:
                        cv2.putText(
                            image,
                            "* refined live",
                            (image.shape[1] - 220, 118),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.55,
                            (180, 180, 180),
                            1,
                            cv2.LINE_AA,
                        )
    
                    # Frame index overlay
                    cv2.putText(image, 'Frame', (0, 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
                    cv2.putText(image, str(int(current_frame_idx)),
                                (10, 52),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
    
                    # Render detections
                    if results is not None and results.pose_landmarks is not None:
                        mp_drawing.draw_landmarks(
                            image,
                            results.pose_landmarks,
                            mp_pose.POSE_CONNECTIONS,
                            mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                            mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2),
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
    
                        cv2.rectangle(image, (e1 - 20, f1 + 10), (e1 + 60, f1 + 70), (0, 255, 0), 2)
                        cv2.rectangle(image, (m1 - 70, n1 + 10), (m1 + 10, n1 + 70), (0, 255, 0), 2)
    
                    rope_label = "Flag: Rope(Dual)" if runtime_require_dual_rope else "Flag: Rope"
                    if effective_rope_detected_this_frame:
                        cv2.putText(image, rope_label, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    else:
                        cv2.putText(image, "Flag: No Rope", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
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
        selected_events = detect_jump_events_offline(
            hip_series=hip_center_y,
            left_foot_x=left_foot_x,
            left_foot_y=left_foot_y,
            right_foot_x=right_foot_x,
            right_foot_y=right_foot_y,
            frame_numbers=landmark_frame_numbers,
            frame_timestamps_ms=landmark_timestamps_ms,
            fps=fps,
            width=width,
            height=height,
            threshold_gain=ADAPTIVE_THRESHOLD_GAIN,
            minima_lag_frames=LOCAL_MINIMA_LAG_FRAMES,
            prominence_window=LOCAL_PROMINENCE_WINDOW,
            min_jump_gap_seconds=MIN_JUMP_GAP_SECONDS,
            landing_offset_ms=LANDING_OFFSET_MS,
            event_time_bias_ms=EVENT_TIME_BIAS_MS,
            min_strength_ratio=runtime_min_strength_ratio,
            strict_guards_enabled=strict_guard_mode,
            rope_flag_series=rope_flag_series,
            rope_dual_flag_series=rope_dual_flag_series,
            startup_lockout_seconds=runtime_startup_lockout_seconds,
            active_enter_min_events=runtime_enter_min_events,
            foot_lift_min_prominence=runtime_foot_lift_min_prominence,
        )
        selected_events = sorted(selected_events, key=lambda event: int(event["frame"]))
        for idx, event in enumerate(selected_events, start=1):
            event["count"] = idx
        if len(selected_events) != len(detected_jump_events):
            print(
                f"[POST] offline_refine_count raw_online={len(detected_jump_events)} "
                f"offline_refined={len(selected_events)}"
            )
    
        if not label_events:
            if ENABLE_OVERLAY_REFRESH:
                overlay_ok, overlay_message = rewrite_tracked_video_count_overlay(
                    out_path,
                    selected_events,
                    [],
                    raw_online_count=len(detected_jump_events),
                )
                if overlay_ok:
                    print(f"[POST] refreshed_count_overlay {overlay_message}")
                else:
                    print(f"[WARN] Failed to refresh count overlay: {overlay_message}")
            else:
                print("[POST] skipped_count_overlay_refresh (JR_ENABLE_OVERLAY_REFRESH=false)")
    
            detected_events_csv = os.path.join(job_output_dir, f"{stem}_detected_events.csv")
            detected_rows = []
            for item in selected_events:
                left_point = item.get("left_point", (np.nan, np.nan))
                right_point = item.get("right_point", (np.nan, np.nan))
                detected_rows.append(
                    {
                        "count": int(item.get("count", 0)),
                        "frame": int(item.get("frame", -1)),
                        "timestamp_ms": int(item.get("timestamp_ms", -1)),
                        "strength_ratio": float(item.get("strength_ratio", np.nan)),
                        "rope_ratio": float(item.get("rope_ratio", np.nan)),
                        "rope_dual_ratio": float(item.get("rope_dual_ratio", np.nan)),
                        "left_x": left_point[0],
                        "left_y": left_point[1],
                        "right_x": right_point[0],
                        "right_y": right_point[1],
                    }
                )
            pd.DataFrame(detected_rows).to_csv(detected_events_csv, index=False)
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
                raw_online_count=len(detected_jump_events),
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
        matched_events = selected_eval["matched_events"]
        strict_extra_detected = selected_eval["strict_extra_detected"]
        extra_detected = selected_eval["extra_detected"]
        full_strict_extra_detected = selected_eval["full_strict_extra_detected"]
        full_extra_detected = selected_eval["full_extra_detected"]
        gap_candidate_events = selected_eval["gap_candidate_events"]
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
            f"(includes_outside_window={len(outside_window_events)})"
        )
        print(
            f"[COMPARE] full_strict_precision={full_strict_precision:.3f} "
            f"full_strict_recall={full_strict_recall:.3f} "
            f"full_strict_f1={full_strict_f1:.3f} "
            f"full_strict_accuracy={full_strict_accuracy:.3f}"
        )
        print(
            f"[COMPARE] full_adjusted_extra_detected={len(full_extra_detected)} "
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
        for detected_item in outside_window_events:
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
    
    if overall_summary_rows:
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
        mode=runtime_args.mode,
        target_stem=runtime_args.target_stem,
        target_video_path=runtime_args.video_path,
        camera_index=runtime_args.camera_index,
    )


def main(argv=None):
    runtime_args = parse_runtime_args(argv=argv)
    return run_with_args(runtime_args)


if __name__ == "__main__":
    main()
