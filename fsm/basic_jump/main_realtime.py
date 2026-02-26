import argparse

from jump_rope_pipeline import run_pipeline
from jump_rope_settings import REALTIME_CAMERA_INDEX


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
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    return run_pipeline(
        mode="realtime",
        camera_index=args.camera_index,
        realtime_fps_log_interval_s=args.fps_log_interval,
    )


if __name__ == "__main__":
    main()
