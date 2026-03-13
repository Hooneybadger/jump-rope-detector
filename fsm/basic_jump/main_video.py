import argparse

from jump_rope_pipeline import run_pipeline
from jump_rope_settings import OUTPUT_DIR, TARGET_VIDEO_PATH, TARGET_VIDEO_STEM


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Jump rope tracker (video mode)")
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
        "--demo-log",
        action="store_true",
        help="save a probe session folder (tracked video + frame log csv)",
    )
    parser.add_argument(
        "--demo-log-dir",
        default=f"{OUTPUT_DIR}/ux_probe",
        help="base directory for probe session folders",
    )
    parser.add_argument(
        "--demo-tag",
        default="",
        help="optional tag added to probe session folder names",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    return run_pipeline(
        target_stem=args.target_stem,
        target_video_path=args.video_path,
        realtime_demo_log=args.demo_log,
        realtime_demo_log_dir=args.demo_log_dir,
        realtime_demo_tag=args.demo_tag,
    )


if __name__ == "__main__":
    main()
