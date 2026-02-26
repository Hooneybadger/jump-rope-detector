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
def run_pipeline(
    mode,
    target_stem=None,
    target_video_path=None,
    camera_index=0,
    realtime_fps_log_interval_s=1.0,
):
    run_mode = (mode or "labeled").strip().lower()
    realtime_fps_log_interval_s = max(0.0, float(realtime_fps_log_interval_s))
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
    
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        if is_realtime:
            out_stamp = time.strftime("%Y%m%d_%H%M%S")
            out_name = f"{stem}_{out_stamp}_tracked.mp4"
        else:
            out_name = os.path.splitext(os.path.basename(file_path))[0] + "_tracked.mp4"
        out_path = os.path.join(OUTPUT_DIR, out_name)
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
        startup_lockout_frames = max(0, int(round(fps * STARTUP_LOCKOUT_SECONDS)))
        is_active = False
        pending_candidates = []
        last_valid_candidate_frame = -10**9
        rope_flag_series = []
        rope_dual_flag_series = []
        rope_active_window_frames = max(1, int(round(fps * ROPE_ACTIVE_WINDOW_SECONDS)))
        rope_exit_idle_frames = max(1, int(round(fps * ROPE_EXIT_IDLE_SECONDS)))
        last_rope_active_frame = -10**9
        last_frame_timestamp_ms = -1
    
    
    
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
                    rope_detected_this_frame = False
                    rope_dual_detected_this_frame = False
    
                    if processed_frames % 300 == 0:
                        elapsed = time.time() - start_time
                        print(f"[INFO] processed={processed_frames} elapsed={elapsed:.1f}s")
    
                    # Extract landmarks
                    try:
                        if results is None or results.pose_landmarks is None:
                            raise ValueError("no_pose")
                        landmarks = results.pose_landmarks.landmark
    
                        hip_center_y.append((landmarks[23].y + landmarks[24].y) * 0.5)
                        left_foot_x.append(landmarks[31].x)
                        right_foot_x.append(landmarks[32].x)
                        left_foot_y.append(landmarks[31].y)
                        right_foot_y.append(landmarks[32].y)
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
                            candidate_frame = landmark_frame_numbers[candidate_idx]
                            startup_locked = candidate_frame < startup_lockout_frames
                            if startup_locked:
                                last_processed_minima_idx = candidate_idx
                            is_far_enough = (candidate_frame - last_jump_frame) >= min_jump_gap_frames
                            rope_ratio = get_true_ratio(rope_flag_series, candidate_idx, rope_active_window_frames)
                            rope_dual_ratio = get_true_ratio(rope_dual_flag_series, candidate_idx, rope_active_window_frames)
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
                            if (
                                (not startup_locked)
                                and is_local_min
                                and strength_ratio >= MIN_STRENGTH_RATIO
                                and is_far_enough
                                and rope_active
                            ):
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
                                        min_events=ACTIVE_ENTER_MIN_EVENTS,
                                        max_gap_frames=active_enter_max_gap_frames,
                                        max_cv=ACTIVE_ENTER_CADENCE_MAX_CV,
                                    ):
                                        is_active = True
                                        tail_n = choose_entry_backfill_tail_count(pending_candidates)
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
    
                    ankle_counter = jump_counter
    
                    current_ts_ms = current_frame_ts_ms + DISPLAY_TIME_ADVANCE_MS
                    while (
                        label_progress_idx < len(label_events)
                        and label_events[label_progress_idx]["timestamp_ms"] <= current_ts_ms
                    ):
                        label_progress_idx += 1
                    count_delta = ankle_counter - label_progress_idx
                    delta_color = (0, 180, 0) if count_delta == 0 else (0, 0, 255)
    
                    # Counter overlay
                    cv2.putText(image, "Jumps", (image.shape[1] - 130, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2,
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
    
                        cv2.rectangle(image, (e1 - 20, f1 + 10), (e1 + 60, f1 + 70), (0, 255, 0), 2)
                        cv2.rectangle(image, (m1 - 70, n1 + 10), (m1 + 10, n1 + 70), (0, 255, 0), 2)
    
                    if rope_detected_this_frame:
                        cv2.putText(image, "Flag: Rope", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    else:
                        cv2.putText(image, "Flag: No Rope", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
                    if pose_detected:
                        rope_flag_series.append(rope_detected_this_frame)
                        rope_dual_flag_series.append(rope_dual_detected_this_frame)
                        if rope_detected_this_frame:
                            last_rope_active_frame = landmark_frame_numbers[-1]
    
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
            if not HEADLESS:
                cv2.destroyAllWindows()
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
            min_strength_ratio=MIN_STRENGTH_RATIO,
            rope_flag_series=rope_flag_series,
            rope_dual_flag_series=rope_dual_flag_series,
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
                )
                if overlay_ok:
                    print(f"[POST] refreshed_count_overlay {overlay_message}")
                else:
                    print(f"[WARN] Failed to refresh count overlay: {overlay_message}")
            else:
                print("[POST] skipped_count_overlay_refresh (JR_ENABLE_OVERLAY_REFRESH=false)")
    
            detected_events_csv = os.path.join(OUTPUT_DIR, f"{stem}_detected_events.csv")
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
        gap_candidate_events = selected_eval["gap_candidate_events"]
        missed_labels = selected_eval["missed_labels"]
        strict_precision = selected_eval["strict_precision"]
        strict_recall = selected_eval["strict_recall"]
        strict_f1 = selected_eval["strict_f1"]
        strict_accuracy = selected_eval["strict_accuracy"]
        precision = selected_eval["precision"]
        recall = selected_eval["recall"]
        f1 = selected_eval["f1"]
        accuracy = selected_eval["accuracy"]
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
    
        compare_csv = os.path.join(OUTPUT_DIR, f"{stem}_label_compare.csv")
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
            f"adjusted_f1_mean={overall_summary_df['adjusted_f1'].mean():.3f}"
        )
        print("[SUMMARY] Per video:")
        for _, row in overall_summary_df.iterrows():
            print(
                f"[SUMMARY] {row['video_stem']}: "
                f"strict_f1={row['strict_f1']:.3f} adjusted_f1={row['adjusted_f1']:.3f} "
                f"matched={int(row['matched'])}/{int(row['label_count'])} "
                f"strict_extra={int(row['strict_extra_detected'])} "
                f"adjusted_extra={int(row['adjusted_extra_detected'])} "
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
