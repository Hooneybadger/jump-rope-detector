import os
import xml.etree.ElementTree as ET

import cv2
import numpy as np

from jump_rope_settings import *
def parse_point(point_text):
    x_str, y_str = point_text.split(";")
    return float(x_str), float(y_str)


def load_label_events(label_path):
    if not os.path.isfile(label_path):
        print(f"[WARN] Label file not found: {label_path}")
        return []

    try:
        root = ET.parse(label_path).getroot()
    except Exception as exc:
        print(f"[WARN] Failed to parse label file: {exc}")
        return []

    avg_ts_per_frame = float(root.findtext("AverageTimeStampsPerFrame") or 0.0)
    user_fps = float(root.findtext("UserFramerate") or 0.0)
    timestamp_scale_ms = 1.0
    if avg_ts_per_frame >= 100.0 and user_fps > 0.0:
        # Some Kinovea exports store timestamps in high-frequency ticks (e.g., 512 ticks/frame).
        # Convert them to milliseconds using the file timebase.
        timestamp_scale_ms = 1000.0 / (avg_ts_per_frame * user_fps)
        print(
            f"[INFO] Label timestamp scale applied for {os.path.basename(label_path)}: "
            f"x{timestamp_scale_ms:.6f} ms/tick"
        )

    events = []
    for keyframe in root.findall(".//Keyframe"):
        timestamp_text = keyframe.findtext("Timestamp")
        if timestamp_text is None:
            continue

        foot_points = []
        for pencil in keyframe.findall("./Drawings/Pencil"):
            point_text = pencil.findtext("./PointList/Point")
            if not point_text:
                continue
            try:
                foot_points.append(parse_point(point_text))
            except ValueError:
                continue

        if len(foot_points) < 2:
            continue

        foot_points = sorted(foot_points, key=lambda pt: pt[0])
        events.append(
            {
                "timestamp_ms": int(round(float(timestamp_text) * timestamp_scale_ms)),
                "left_point": foot_points[0],
                "right_point": foot_points[1],
            }
        )

    events.sort(key=lambda event: event["timestamp_ms"])
    return events


def get_match_tolerance_ms(fps):
    frame_based_ms = LABEL_MATCH_TOLERANCE_MS
    if fps and fps > 0:
        frame_based_ms = int(round((LABEL_MATCH_TOLERANCE_FRAMES * 1000.0) / fps))

    tolerance_ms = max(LABEL_MATCH_TOLERANCE_MS, frame_based_ms)
    if LABEL_MATCH_TOLERANCE_MAX_MS > 0:
        tolerance_ms = min(tolerance_ms, LABEL_MATCH_TOLERANCE_MAX_MS)

    return int(tolerance_ms), int(frame_based_ms)


def compare_events(detected_events, label_events, tolerance_ms):
    matched = []
    extra_detected = []
    missed_labels = []
    i, j = 0, 0

    while i < len(detected_events) and j < len(label_events):
        detected = detected_events[i]
        labeled = label_events[j]
        time_error = detected["timestamp_ms"] - labeled["timestamp_ms"]

        if abs(time_error) <= int(tolerance_ms):
            left_error = float(np.linalg.norm(np.array(detected["left_point"]) - np.array(labeled["left_point"])))
            right_error = float(np.linalg.norm(np.array(detected["right_point"]) - np.array(labeled["right_point"])))
            matched.append(
                {
                    "detected": detected,
                    "label": labeled,
                    "time_error_ms": time_error,
                    "left_point_error_px": left_error,
                    "right_point_error_px": right_error,
                }
            )
            i += 1
            j += 1
        elif detected["timestamp_ms"] < labeled["timestamp_ms"] - tolerance_ms:
            extra_detected.append(detected)
            i += 1
        else:
            missed_labels.append(labeled)
            j += 1

    if i < len(detected_events):
        extra_detected.extend(detected_events[i:])
    if j < len(label_events):
        missed_labels.extend(label_events[j:])

    return matched, extra_detected, missed_labels


def compute_metrics(matched_count, extra_count, missed_count):
    precision_den = matched_count + extra_count
    recall_den = matched_count + missed_count
    precision = (matched_count / precision_den) if precision_den else 0.0
    recall = (matched_count / recall_den) if recall_den else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    accuracy_den = matched_count + extra_count + missed_count
    accuracy = (matched_count / accuracy_den) if accuracy_den else 0.0
    return precision, recall, f1, accuracy


def filter_events_to_label_window(detected_events, label_events, padding_ms=LABEL_WINDOW_PADDING_MS):
    if not label_events:
        return list(detected_events), []
    start_ms = label_events[0]["timestamp_ms"] - padding_ms
    end_ms = label_events[-1]["timestamp_ms"] + padding_ms
    in_window = []
    out_window = []
    for event in detected_events:
        if start_ms <= event["timestamp_ms"] <= end_ms:
            in_window.append(event)
        else:
            out_window.append(event)
    return in_window, out_window


def rewrite_tracked_video_count_overlay(tracked_video_path, selected_events, label_events):
    if not os.path.isfile(tracked_video_path):
        return False, "tracked video not found"

    cap = cv2.VideoCapture(tracked_video_path)
    if not cap.isOpened():
        return False, "failed to open tracked video"

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    source_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    temp_output_path = tracked_video_path + ".tmp.mp4"
    writer = cv2.VideoWriter(temp_output_path, fourcc, fps, (width, height))
    if not writer.isOpened():
        cap.release()
        return False, "failed to open overlay writer"

    ordered_detected = sorted(selected_events, key=lambda item: int(item["timestamp_ms"]))
    ordered_labels = sorted(label_events, key=lambda item: int(item["timestamp_ms"]))
    detected_idx = 0
    label_idx = 0
    frame_idx = -1
    last_timestamp_ms = -1
    panel_left = max(0, width - 270)
    panel_top = 0
    panel_right = width
    panel_bottom = 110

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        frame_idx += 1
        raw_ts_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
        current_ts_ms = resolve_frame_timestamp_ms(raw_ts_ms, frame_idx, fps, last_timestamp_ms)
        last_timestamp_ms = current_ts_ms
        display_ts_ms = current_ts_ms + DISPLAY_TIME_ADVANCE_MS

        while detected_idx < len(ordered_detected) and int(ordered_detected[detected_idx]["timestamp_ms"]) <= display_ts_ms:
            detected_idx += 1
        while label_idx < len(ordered_labels) and int(ordered_labels[label_idx]["timestamp_ms"]) <= display_ts_ms:
            label_idx += 1

        count_delta = detected_idx - label_idx
        delta_color = (0, 180, 0) if count_delta == 0 else (0, 0, 255)

        # Hide the first-pass counters and redraw finalized counters from post-processed events.
        cv2.rectangle(frame, (panel_left, panel_top), (panel_right, panel_bottom), (0, 0, 0), -1)
        cv2.putText(
            frame,
            "Jumps",
            (width - 130, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            str(detected_idx),
            (width - 40, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            "Label",
            (width - 130, 62),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 120, 0),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            str(label_idx),
            (width - 40, 62),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 120, 0),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            f"Delta {count_delta:+d}",
            (width - 220, 94),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            delta_color,
            2,
            cv2.LINE_AA,
        )

        writer.write(frame)

    cap.release()
    writer.release()

    if not os.path.isfile(temp_output_path):
        return False, "overlay temp video missing"

    refreshed_frame_count = frame_idx + 1
    if source_frame_count > 0 and (refreshed_frame_count + 2) < source_frame_count:
        try:
            os.remove(temp_output_path)
        except OSError:
            pass
        return False, (
            "overlay frame mismatch "
            f"source={source_frame_count} refreshed={refreshed_frame_count}"
        )

    os.replace(temp_output_path, tracked_video_path)
    final_delta = detected_idx - label_idx
    return True, (
        f"frames={refreshed_frame_count} jumps={detected_idx} labels={label_idx} "
        f"delta={final_delta:+d}"
    )


def split_gap_candidate_events(extra_detected, label_events):
    if len(label_events) < 3 or not extra_detected:
        return extra_detected, []

    label_timestamps = [item["timestamp_ms"] for item in label_events]
    intervals = np.diff(label_timestamps)
    if intervals.size == 0:
        return extra_detected, []

    median_gap = float(np.median(intervals))
    large_gaps = []
    for i in range(len(label_timestamps) - 1):
        start_ts = label_timestamps[i]
        end_ts = label_timestamps[i + 1]
        gap = end_ts - start_ts
        if gap >= (1.7 * median_gap):
            large_gaps.append((start_ts, end_ts))

    if not large_gaps:
        return extra_detected, []

    confirmed_extra = []
    gap_candidates = []
    for event in extra_detected:
        ts = event["timestamp_ms"]
        is_gap_candidate = False
        for start_ts, end_ts in large_gaps:
            if start_ts < ts < end_ts:
                mid_ts = 0.5 * (start_ts + end_ts)
                if abs(ts - mid_ts) <= ((end_ts - start_ts) * 0.30):
                    is_gap_candidate = True
                    break
        if is_gap_candidate:
            gap_candidates.append(event)
        else:
            confirmed_extra.append(event)

    return confirmed_extra, gap_candidates

def evaluate_detected_events(detected_events, label_events, tolerance_ms):
    compared_detected_events, outside_window_events = filter_events_to_label_window(
        detected_events,
        label_events,
    )
    matched_events, strict_extra_detected_raw, missed_labels = compare_events(
        compared_detected_events,
        label_events,
        tolerance_ms=tolerance_ms,
    )
    extra_detected, gap_candidate_events = split_gap_candidate_events(strict_extra_detected_raw, label_events)
    if STRICT_IGNORE_GAP_CANDIDATES:
        strict_extra_detected = extra_detected
    else:
        strict_extra_detected = strict_extra_detected_raw
    strict_precision, strict_recall, strict_f1, strict_accuracy = compute_metrics(
        len(matched_events),
        len(strict_extra_detected),
        len(missed_labels),
    )
    precision, recall, f1, accuracy = compute_metrics(
        len(matched_events),
        len(extra_detected),
        len(missed_labels),
    )
    mean_abs_time_error_ms = float(
        np.mean([abs(item["time_error_ms"]) for item in matched_events])
    ) if matched_events else float("inf")

    return {
        "compared_detected_events": compared_detected_events,
        "outside_window_events": outside_window_events,
        "matched_events": matched_events,
        "strict_extra_detected": strict_extra_detected,
        "extra_detected": extra_detected,
        "gap_candidate_events": gap_candidate_events,
        "missed_labels": missed_labels,
        "strict_precision": strict_precision,
        "strict_recall": strict_recall,
        "strict_f1": strict_f1,
        "strict_accuracy": strict_accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
        "mean_abs_time_error_ms": mean_abs_time_error_ms,
        "tolerance_ms": int(tolerance_ms),
    }
