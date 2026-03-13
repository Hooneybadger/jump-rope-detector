import argparse

from jump_rope_pipeline import run_pipeline
from jump_rope_settings import OUTPUT_DIR, REALTIME_CAMERA_INDEX


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Jump rope tracker (realtime mode)")
    parser.add_argument(
        "--camera-index",
        type=int,
        default=REALTIME_CAMERA_INDEX,
        help="camera index for realtime mode",
    )
    parser.add_argument(
        "--fps-log-interval",
        type=float,
        default=1.0,
        help="realtime FPS log interval seconds (<=0 to disable)",
    )
    parser.add_argument(
        "--demo-log",
        action="store_true",
        help="save a demo-log session folder (tracked video + frame log csv)",
    )
    parser.add_argument(
        "--demo-log-dir",
        default=f"{OUTPUT_DIR}/realtime_demo_logs",
        help="base directory for demo-log sessions",
    )
    parser.add_argument(
        "--demo-tag",
        default="",
        help="optional tag added to demo-log session folder names",
    )
    parser.add_argument(
        "--demo-save-raw",
        action="store_true",
        help="also save raw camera video as raw.mp4 when --demo-log is enabled",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    return run_pipeline(
        camera_index=args.camera_index,
        use_realtime=True,
        realtime_fps_log_interval_s=args.fps_log_interval,
        realtime_demo_log=args.demo_log,
        realtime_demo_log_dir=args.demo_log_dir,
        realtime_demo_tag=args.demo_tag,
        realtime_demo_save_raw=args.demo_save_raw,
    )


if __name__ == "__main__":
    main()
