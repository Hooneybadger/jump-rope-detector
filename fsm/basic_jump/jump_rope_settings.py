import os
import warnings

import numpy as np

warnings.filterwarnings("ignore")
def get_env_float(name, default_value):
    value = os.environ.get(name)
    if value is None:
        return default_value
    try:
        return float(value)
    except ValueError:
        return default_value


def get_env_int(name, default_value):
    value = os.environ.get(name)
    if value is None:
        return default_value
    try:
        return int(float(value))
    except ValueError:
        return default_value


def get_env_bool(name, default_value):
    value = os.environ.get(name)
    if value is None:
        return default_value
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def resolve_frame_timestamp_ms(raw_timestamp_ms, frame_idx, fps, last_timestamp_ms=-1):
    if fps and fps > 0:
        fallback_ms = int(round((frame_idx * 1000.0) / fps))
    else:
        fallback_ms = int(round(frame_idx * 33.333))

    use_raw = raw_timestamp_ms is not None and np.isfinite(raw_timestamp_ms) and raw_timestamp_ms >= 0.0
    timestamp_ms = int(round(raw_timestamp_ms)) if use_raw else fallback_ms

    if use_raw and frame_idx > 0 and timestamp_ms == 0 and fallback_ms > 0:
        timestamp_ms = fallback_ms

    if last_timestamp_ms >= 0 and timestamp_ms < last_timestamp_ms:
        timestamp_ms = max(last_timestamp_ms, fallback_ms)

    return int(timestamp_ms)


BASE_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.dirname(BASE_DIR)
REPO_ROOT = os.path.dirname(PROJECT_ROOT)
DEFAULT_INPUT_VIDEO_DIR = os.path.join(BASE_DIR, "input", "video")
DEFAULT_INPUT_LABEL_DIR = os.path.join(BASE_DIR, "input", "label")
VIDEO_INPUT_CANDIDATES = [
    DEFAULT_INPUT_VIDEO_DIR,
    os.path.join(PROJECT_ROOT, "input", "video"),
    os.path.join(REPO_ROOT, "fsm", "input", "video"),
]
LABEL_INPUT_CANDIDATES = [
    DEFAULT_INPUT_LABEL_DIR,
    os.path.join(PROJECT_ROOT, "input", "label"),
    os.path.join(REPO_ROOT, "fsm", "input", "label"),
]
INPUT_VIDEO_DIR = os.environ.get("JR_INPUT_VIDEO_DIR") or DEFAULT_INPUT_VIDEO_DIR
INPUT_LABEL_DIR = os.environ.get("JR_INPUT_LABEL_DIR") or DEFAULT_INPUT_LABEL_DIR
if not os.environ.get("JR_INPUT_VIDEO_DIR"):
    for candidate_dir in VIDEO_INPUT_CANDIDATES:
        if os.path.isdir(candidate_dir):
            INPUT_VIDEO_DIR = candidate_dir
            break
if not os.environ.get("JR_INPUT_LABEL_DIR"):
    for candidate_dir in LABEL_INPUT_CANDIDATES:
        if os.path.isdir(candidate_dir):
            INPUT_LABEL_DIR = candidate_dir
            break
OUTPUT_DIR = os.environ.get("JR_OUTPUT_DIR") or os.path.join(BASE_DIR, "output")
VIDEO_EXTS = (".mp4", ".avi", ".mkv", ".mov", ".webm")
HEADLESS = os.environ.get("DISPLAY") is None
TARGET_VIDEO_STEM = os.environ.get("JR_TARGET_VIDEO_STEM")
TARGET_VIDEO_PATH = os.environ.get("JR_TARGET_VIDEO_PATH")
REALTIME_CAMERA_INDEX = get_env_int("JR_REALTIME_CAMERA_INDEX", 0)
REALTIME_MAX_SECONDS = get_env_float("JR_REALTIME_MAX_SECONDS", 0.0)
REALTIME_MAX_FRAMES = get_env_int("JR_REALTIME_MAX_FRAMES", 0)
STRICT_GUARDS = get_env_bool(
    "JR_STRICT_GUARDS",
    get_env_bool("JR_REALTIME_STRICT_GUARDS", True),
)
STRICT_LOWER_BODY_VIS_MIN = get_env_float(
    "JR_STRICT_LOWER_BODY_VIS_MIN",
    get_env_float("JR_REALTIME_STRICT_LOWER_BODY_VIS_MIN", 0.15),
)
STRICT_REQUIRE_DUAL_ROPE = get_env_bool(
    "JR_STRICT_REQUIRE_DUAL_ROPE",
    get_env_bool("JR_REALTIME_STRICT_REQUIRE_DUAL_ROPE", False),
)
STRICT_MIN_STRENGTH_RATIO = get_env_float(
    "JR_STRICT_MIN_STRENGTH_RATIO",
    get_env_float("JR_REALTIME_STRICT_MIN_STRENGTH_RATIO", 1.0),
)
STRICT_ENTER_MIN_EVENTS = get_env_int(
    "JR_STRICT_ENTER_MIN_EVENTS",
    get_env_int("JR_REALTIME_STRICT_ENTER_MIN_EVENTS", 2),
)
STRICT_STARTUP_LOCKOUT_SECONDS = get_env_float(
    "JR_STRICT_STARTUP_LOCKOUT_SECONDS",
    get_env_float("JR_REALTIME_STRICT_STARTUP_LOCKOUT_SECONDS", 0.8),
)
STRICT_FOOT_LIFT_MIN_PROMINENCE = get_env_float("JR_STRICT_FOOT_LIFT_MIN_PROMINENCE", 0.0008)
STRICT_BOTH_FEET_MIN_PROMINENCE = get_env_float("JR_STRICT_BOTH_FEET_MIN_PROMINENCE", 0.00035)
STRICT_FEET_SYMMETRY_MIN_RATIO = get_env_float("JR_STRICT_FEET_SYMMETRY_MIN_RATIO", 0.25)
STRICT_FOOT_SYNC_WINDOW = get_env_int("JR_STRICT_FOOT_SYNC_WINDOW", 10)
STRICT_FOOT_SYNC_MIN_CORR = get_env_float("JR_STRICT_FOOT_SYNC_MIN_CORR", 0.12)
STRICT_REQUIRE_INPLACE = get_env_bool("JR_STRICT_REQUIRE_INPLACE", False)
STRICT_INPLACE_WINDOW = get_env_int("JR_STRICT_INPLACE_WINDOW", 12)
STRICT_INPLACE_MAX_CENTER_DRIFT = get_env_float("JR_STRICT_INPLACE_MAX_CENTER_DRIFT", 0.10)
STRICT_REQUIRE_ADVANCED_MOTION = get_env_bool("JR_STRICT_REQUIRE_ADVANCED_MOTION", False)
STRICT_MOTION_ADVANCED_MIN_CHECKS = get_env_int("JR_STRICT_MOTION_ADVANCED_MIN_CHECKS", 1)
STRICT_OVERRIDE_MIN_FOOT_PROMINENCE = get_env_float("JR_STRICT_OVERRIDE_MIN_FOOT_PROMINENCE", 0.00045)
STRICT_OVERRIDE_MIN_STRENGTH_RATIO = get_env_float("JR_STRICT_OVERRIDE_MIN_STRENGTH_RATIO", 2.0)
STRICT_OVERRIDE_ACTIVE_MIN_ROPE_RATIO = get_env_float("JR_STRICT_OVERRIDE_ACTIVE_MIN_ROPE_RATIO", 0.00)
STRICT_OVERRIDE_ACTIVE_MIN_DUAL_RATIO = get_env_float("JR_STRICT_OVERRIDE_ACTIVE_MIN_DUAL_RATIO", 0.00)
STRICT_OVERRIDE_ENTRY_MIN_ROPE_RATIO = get_env_float("JR_STRICT_OVERRIDE_ENTRY_MIN_ROPE_RATIO", 0.20)
STRICT_OVERRIDE_ENTRY_MIN_DUAL_RATIO = get_env_float("JR_STRICT_OVERRIDE_ENTRY_MIN_DUAL_RATIO", 0.00)
STRICT_ENTRY_NO_FLAG_BOOTSTRAP_ENABLED = get_env_bool("JR_STRICT_ENTRY_NO_FLAG_BOOTSTRAP_ENABLED", True)
STRICT_ENTRY_NO_FLAG_BOOTSTRAP_MIN_STRENGTH_RATIO = get_env_float(
    "JR_STRICT_ENTRY_NO_FLAG_BOOTSTRAP_MIN_STRENGTH_RATIO",
    1.25,
)
STRICT_ENTRY_NO_FLAG_BOOTSTRAP_MIN_ROPE_RATIO = get_env_float(
    "JR_STRICT_ENTRY_NO_FLAG_BOOTSTRAP_MIN_ROPE_RATIO",
    0.04,
)
STRICT_ENTRY_NO_FLAG_BOOTSTRAP_MIN_DUAL_RATIO = get_env_float(
    "JR_STRICT_ENTRY_NO_FLAG_BOOTSTRAP_MIN_DUAL_RATIO",
    0.01,
)
STRICT_ACTIVE_FOOT_MOTION_WINDOW = get_env_int("JR_STRICT_ACTIVE_FOOT_MOTION_WINDOW", 10)
STRICT_ACTIVE_LOOSE_FOOT_PROMINENCE = get_env_float("JR_STRICT_ACTIVE_LOOSE_FOOT_PROMINENCE", 0.00025)
STRICT_ACTIVE_FOOT_MOTION_MIN_RATIO = get_env_float("JR_STRICT_ACTIVE_FOOT_MOTION_MIN_RATIO", 0.15)
STRICT_ACTIVE_FOOT_MOTION_MIN_TRUE = get_env_int("JR_STRICT_ACTIVE_FOOT_MOTION_MIN_TRUE", 1)
STRICT_ACTIVE_FOOT_MOTION_ROPE_MIN_RATIO = get_env_float("JR_STRICT_ACTIVE_FOOT_MOTION_ROPE_MIN_RATIO", 0.10)
STRICT_ACTIVE_FOOT_MOTION_DUAL_MIN_RATIO = get_env_float("JR_STRICT_ACTIVE_FOOT_MOTION_DUAL_MIN_RATIO", 0.01)
STRICT_ACTIVE_ANTI_WALK_ENABLED = get_env_bool("JR_STRICT_ACTIVE_ANTI_WALK_ENABLED", True)
STRICT_ACTIVE_WALK_MAX_SYNC_CORR = get_env_float("JR_STRICT_ACTIVE_WALK_MAX_SYNC_CORR", 0.00)
STRICT_ACTIVE_WALK_MIN_CENTER_DRIFT = get_env_float("JR_STRICT_ACTIVE_WALK_MIN_CENTER_DRIFT", 0.09)
STRICT_WEAK_NON_BILATERAL_PRUNE_ENABLED = get_env_bool(
    "JR_STRICT_WEAK_NON_BILATERAL_PRUNE_ENABLED",
    True,
)
STRICT_WEAK_NON_BILATERAL_MAX_STRENGTH = get_env_float(
    "JR_STRICT_WEAK_NON_BILATERAL_MAX_STRENGTH",
    2.0,
)
STRICT_SESSION_MIN_MEDIAN_STRENGTH_RATIO = get_env_float(
    "JR_STRICT_SESSION_MIN_MEDIAN_STRENGTH_RATIO",
    6.0,
)
LABEL_MATCH_TOLERANCE_MS = get_env_int("JR_MATCH_TOLERANCE_MS", 120)
LABEL_MATCH_TOLERANCE_FRAMES = get_env_float("JR_MATCH_TOLERANCE_FRAMES", 5.0)
LABEL_MATCH_TOLERANCE_MAX_MS = get_env_int("JR_MATCH_TOLERANCE_MAX_MS", 180)
DISPLAY_TIME_ADVANCE_MS = get_env_int("JR_DISPLAY_TIME_ADVANCE_MS", 50)
ENABLE_OVERLAY_REFRESH = get_env_bool("JR_ENABLE_OVERLAY_REFRESH", True)
OVERLAY_END_WAIT_ENABLED = get_env_bool("JR_OVERLAY_END_WAIT_ENABLED", True)
OVERLAY_END_WAIT_SECONDS = get_env_float("JR_OVERLAY_END_WAIT_SECONDS", 1.5)
OVERLAY_END_WAIT_MAX_SECONDS = get_env_float("JR_OVERLAY_END_WAIT_MAX_SECONDS", 8.0)
LIVE_OVERLAY_FIXED_LAG_COUNT = get_env_bool("JR_LIVE_OVERLAY_FIXED_LAG_COUNT", True)
FIXED_LAG_SECONDS = get_env_float("JR_FIXED_LAG_SECONDS", 1.0)
FIXED_LAG_FINALIZE_WAIT_SECONDS = get_env_float("JR_FIXED_LAG_FINALIZE_WAIT_SECONDS", 2.0)

LEGACY_JUMP_THRESHOLD = 0.0030
ADAPTIVE_THRESHOLD_LOOKBACK = 75
ADAPTIVE_THRESHOLD_SCALE = 1.2
ADAPTIVE_THRESHOLD_MIN = 0.00035
ADAPTIVE_THRESHOLD_WARMUP = 0.0020
ADAPTIVE_THRESHOLD_GAIN = get_env_float("JR_THRESHOLD_GAIN", 0.60)
LOCAL_MINIMA_LAG_FRAMES = get_env_int("JR_MINIMA_LAG_FRAMES", 4)
LOCAL_PROMINENCE_WINDOW = get_env_int("JR_PROMINENCE_WINDOW", 6)
MIN_JUMP_GAP_SECONDS = get_env_float("JR_MIN_JUMP_GAP_SECONDS", 0.17)
LANDING_OFFSET_MS = get_env_int("JR_LANDING_OFFSET_MS", 100)
EVENT_TIME_BIAS_MS = get_env_int("JR_EVENT_TIME_BIAS_MS", 100)
MIN_STRENGTH_RATIO = get_env_float("JR_MIN_STRENGTH_RATIO", 1.0)
LABEL_WINDOW_PADDING_MS = get_env_int("JR_LABEL_WINDOW_PADDING_MS", 150)
STRICT_IGNORE_GAP_CANDIDATES = get_env_bool("JR_STRICT_IGNORE_GAP_CANDIDATES", True)
ACTIVE_ENTER_WINDOW_SECONDS = get_env_float("JR_ACTIVE_ENTER_WINDOW_SECONDS", 0.8)
ACTIVE_ENTER_MIN_EVENTS = get_env_int("JR_ACTIVE_ENTER_MIN_EVENTS", 2)
ACTIVE_ENTER_CONFIRM_TAIL_EVENTS = get_env_int("JR_ACTIVE_ENTER_CONFIRM_TAIL_EVENTS", 2)
ACTIVE_ENTER_MAX_GAP_SECONDS = get_env_float("JR_ACTIVE_ENTER_MAX_GAP_SECONDS", 0.55)
ACTIVE_ENTER_CADENCE_MAX_CV = get_env_float("JR_ACTIVE_ENTER_CADENCE_MAX_CV", 0.55)
ACTIVE_EXIT_IDLE_SECONDS = get_env_float("JR_ACTIVE_EXIT_IDLE_SECONDS", 0.8)
SESSION_MIN_EVENTS = get_env_int("JR_SESSION_MIN_EVENTS", 5)
STRICT_SESSION_FILTER_ENABLED = get_env_bool("JR_STRICT_SESSION_FILTER_ENABLED", True)
STRICT_SESSION_SPLIT_GAP_SECONDS = get_env_float("JR_STRICT_SESSION_SPLIT_GAP_SECONDS", 0.90)
STRICT_SESSION_MAX_CADENCE_CV = get_env_float("JR_STRICT_SESSION_MAX_CADENCE_CV", 0.30)
STRICT_SESSION_MIN_ABS_FOOT_PROMINENCE = get_env_float(
    "JR_STRICT_SESSION_MIN_ABS_FOOT_PROMINENCE",
    0.0005,
)
STRICT_SESSION_SHORT_BRIDGE_ENABLED = get_env_bool("JR_STRICT_SESSION_SHORT_BRIDGE_ENABLED", True)
STRICT_SESSION_SHORT_BRIDGE_MIN_EVENTS = get_env_int("JR_STRICT_SESSION_SHORT_BRIDGE_MIN_EVENTS", 2)
STRICT_SESSION_SHORT_BRIDGE_MAX_EVENTS = get_env_int("JR_STRICT_SESSION_SHORT_BRIDGE_MAX_EVENTS", 4)
STRICT_SESSION_SHORT_BRIDGE_MAX_GAP_SECONDS = get_env_float("JR_STRICT_SESSION_SHORT_BRIDGE_MAX_GAP_SECONDS", 1.40)
STRICT_SESSION_SHORT_BRIDGE_CV_SCALE = get_env_float("JR_STRICT_SESSION_SHORT_BRIDGE_CV_SCALE", 1.20)
STRICT_SESSION_GAP_FILL_ENABLED = get_env_bool("JR_STRICT_SESSION_GAP_FILL_ENABLED", True)
STRICT_SESSION_GAP_FILL_MIN_EVENTS = get_env_int("JR_STRICT_SESSION_GAP_FILL_MIN_EVENTS", 12)
STRICT_SESSION_GAP_FILL_MAX_CADENCE_CV = get_env_float("JR_STRICT_SESSION_GAP_FILL_MAX_CADENCE_CV", 0.22)
STRICT_SESSION_GAP_FILL_MIN_RATIO = get_env_float("JR_STRICT_SESSION_GAP_FILL_MIN_RATIO", 1.80)
STRICT_SESSION_GAP_FILL_MAX_RATIO = get_env_float("JR_STRICT_SESSION_GAP_FILL_MAX_RATIO", 8.00)
STRICT_SESSION_GAP_FILL_MAX_INSERT_PER_GAP = get_env_int("JR_STRICT_SESSION_GAP_FILL_MAX_INSERT_PER_GAP", 8)
STRICT_SESSION_GAP_FILL_MAX_TOTAL_INSERT = get_env_int("JR_STRICT_SESSION_GAP_FILL_MAX_TOTAL_INSERT", 24)
STRICT_SESSION_HEAD_FILL_ENABLED = get_env_bool("JR_STRICT_SESSION_HEAD_FILL_ENABLED", True)
STRICT_SESSION_HEAD_FILL_MIN_EVENTS = get_env_int("JR_STRICT_SESSION_HEAD_FILL_MIN_EVENTS", 300)
STRICT_SESSION_HEAD_FILL_INSERT_COUNT = get_env_int("JR_STRICT_SESSION_HEAD_FILL_INSERT_COUNT", 2)
STRICT_SESSION_HEAD_FILL_MAX_CADENCE_CV = get_env_float("JR_STRICT_SESSION_HEAD_FILL_MAX_CADENCE_CV", 0.10)
ROPE_ACTIVE_WINDOW_SECONDS = get_env_float("JR_ROPE_ACTIVE_WINDOW_SECONDS", 0.6)
ROPE_ACTIVE_MIN_RATIO = get_env_float("JR_ROPE_ACTIVE_MIN_RATIO", 0.00)
ROPE_ACTIVE_DUAL_MIN_RATIO = get_env_float("JR_ROPE_ACTIVE_DUAL_MIN_RATIO", 0.00)
ROPE_ENTRY_MIN_RATIO = get_env_float("JR_ROPE_ENTRY_MIN_RATIO", 0.00)
ROPE_ENTRY_DUAL_MIN_RATIO = get_env_float("JR_ROPE_ENTRY_DUAL_MIN_RATIO", 0.00)
ROPE_EXIT_IDLE_SECONDS = get_env_float("JR_ROPE_EXIT_IDLE_SECONDS", 0.8)
ROPE_CONTOUR_MIN_AREA = get_env_int("JR_ROPE_CONTOUR_MIN_AREA", 80)
ROPE_CONTOUR_MIN_SPAN = get_env_int("JR_ROPE_CONTOUR_MIN_SPAN", 6)
ROPE_CONTOUR_MAX_ASPECT_RATIO = get_env_float("JR_ROPE_CONTOUR_MAX_ASPECT_RATIO", 0.90)
ROPE_CONTOUR_MIN_PERIMETER = get_env_float("JR_ROPE_CONTOUR_MIN_PERIMETER", 10.0)
STARTUP_LOCKOUT_SECONDS = get_env_float("JR_STARTUP_LOCKOUT_SECONDS", 0.8)
KNN_HISTORY = get_env_int("JR_KNN_HISTORY", 20)
KNN_DIST2_THRESHOLD = get_env_int("JR_KNN_DIST2_THRESHOLD", 700)
ROPE_MASK_KERNEL_SIZE = get_env_int("JR_ROPE_MASK_KERNEL_SIZE", 3)
ROPE_MASK_OPEN_ITERS = get_env_int("JR_ROPE_MASK_OPEN_ITERS", 0)
ROPE_MASK_CLOSE_ITERS = get_env_int("JR_ROPE_MASK_CLOSE_ITERS", 1)
ENABLE_ADAPTIVE_ENTRY_BACKFILL = get_env_bool("JR_ENABLE_ADAPTIVE_ENTRY_BACKFILL", True)
ENTRY_BACKFILL_VERY_LOW_CONF_TAIL_EVENTS = get_env_int("JR_ENTRY_BACKFILL_VERY_LOW_CONF_TAIL_EVENTS", 2)
ENTRY_BACKFILL_VERY_LOW_ROPE_RATIO = get_env_float("JR_ENTRY_BACKFILL_VERY_LOW_ROPE_RATIO", 0.10)
ENTRY_BACKFILL_VERY_LOW_DUAL_RATIO = get_env_float("JR_ENTRY_BACKFILL_VERY_LOW_DUAL_RATIO", 0.01)
ENTRY_BACKFILL_STRONG_ENTRY_FORCE_ENABLED = get_env_bool(
    "JR_ENTRY_BACKFILL_STRONG_ENTRY_FORCE_ENABLED",
    True,
)
ENTRY_BACKFILL_STRONG_ENTRY_MIN_STRENGTH_RATIO = get_env_float(
    "JR_ENTRY_BACKFILL_STRONG_ENTRY_MIN_STRENGTH_RATIO",
    2.0,
)
ENTRY_BACKFILL_STRONG_ENTRY_MIN_ROPE_RATIO = get_env_float(
    "JR_ENTRY_BACKFILL_STRONG_ENTRY_MIN_ROPE_RATIO",
    0.04,
)
ENTRY_BACKFILL_STRONG_ENTRY_FORCE_TAIL_EVENTS = get_env_int(
    "JR_ENTRY_BACKFILL_STRONG_ENTRY_FORCE_TAIL_EVENTS",
    ACTIVE_ENTER_MIN_EVENTS,
)
ENTRY_BACKFILL_LOW_CONF_TAIL_EVENTS = get_env_int("JR_ENTRY_BACKFILL_LOW_CONF_TAIL_EVENTS", 1)
ENTRY_BACKFILL_HIGH_CONF_TAIL_EVENTS = get_env_int(
    "JR_ENTRY_BACKFILL_HIGH_CONF_TAIL_EVENTS",
    ACTIVE_ENTER_CONFIRM_TAIL_EVENTS,
)
ENTRY_BACKFILL_HIGH_CONF_MIN_ROPE_RATIO = get_env_float("JR_ENTRY_BACKFILL_HIGH_CONF_MIN_ROPE_RATIO", 0.13)
ENTRY_BACKFILL_HIGH_CONF_MIN_DUAL_RATIO = get_env_float("JR_ENTRY_BACKFILL_HIGH_CONF_MIN_DUAL_RATIO", 0.02)

def find_video_path_by_stem(stem):
    if not os.path.isdir(INPUT_VIDEO_DIR):
        return None
    for ext in VIDEO_EXTS:
        file_path = os.path.join(INPUT_VIDEO_DIR, f"{stem}{ext}")
        if os.path.isfile(file_path):
            return file_path
    return None


def find_label_path_by_stem(stem):
    if not os.path.isdir(INPUT_LABEL_DIR):
        return None
    label_path = os.path.join(INPUT_LABEL_DIR, f"{stem}.kva")
    if os.path.isfile(label_path):
        return label_path
    return None


def build_video_jobs(target_stem=None, target_video_path=None, require_label=False):
    jobs = []
    if target_video_path:
        resolved_path = os.path.abspath(os.path.expanduser(target_video_path))
        if not os.path.isfile(resolved_path):
            print(f"[WARN] Target video file not found: {target_video_path}")
            return jobs
        stem = os.path.splitext(os.path.basename(resolved_path))[0]
        if target_stem and stem != target_stem:
            print(
                f"[WARN] Target stem mismatch: stem={target_stem} "
                f"video={os.path.basename(resolved_path)}"
            )
            return jobs
        label_path = find_label_path_by_stem(stem)
        if require_label and label_path is None:
            print(f"[WARN] Matching label not found for video: {resolved_path}")
            return jobs
        jobs.append(
            {
                "stem": stem,
                "video_path": resolved_path,
                "capture_source": resolved_path,
                "label_path": label_path,
                "is_realtime": False,
            }
        )
        return jobs

    if not os.path.isdir(INPUT_VIDEO_DIR):
        return jobs

    for file_name in sorted(os.listdir(INPUT_VIDEO_DIR)):
        stem, ext = os.path.splitext(file_name)
        if ext.lower() not in VIDEO_EXTS:
            continue
        if target_stem and stem != target_stem:
            continue
        video_path = os.path.join(INPUT_VIDEO_DIR, file_name)
        label_path = find_label_path_by_stem(stem)
        if require_label and label_path is None:
            continue
        jobs.append(
            {
                "stem": stem,
                "video_path": video_path,
                "capture_source": video_path,
                "label_path": label_path,
                "is_realtime": False,
            }
        )
    return jobs


def build_realtime_jobs(camera_index):
    stem = os.environ.get("JR_REALTIME_STEM", "").strip() or f"realtime_cam{camera_index}"
    return [
        {
            "stem": stem,
            "video_path": f"camera:{camera_index}",
            "capture_source": int(camera_index),
            "label_path": None,
            "is_realtime": True,
        }
    ]


def build_runtime_jobs(
    target_stem=None,
    target_video_path=None,
    camera_index=0,
    use_realtime=False,
    require_label=False,
):
    if bool(use_realtime):
        return build_realtime_jobs(camera_index)
    return build_video_jobs(
        target_stem=target_stem,
        target_video_path=target_video_path,
        require_label=require_label,
    )


def get_summary_csv_name(use_realtime=False):
    env_name = os.environ.get("JR_SUMMARY_CSV_NAME", "").strip()
    if env_name:
        return env_name
    if bool(use_realtime):
        return "realtime_count_summary.csv"
    return "realtime_engine_summary.csv"
