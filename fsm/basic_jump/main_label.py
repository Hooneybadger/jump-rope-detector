import argparse

from jump_rope_pipeline import run_pipeline
from jump_rope_settings import TARGET_VIDEO_STEM


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Jump rope tracker (labeled mode)")
    parser.add_argument(
        "--target-stem",
        default=TARGET_VIDEO_STEM,
        help="video stem in input/video (ex: 03)",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    return run_pipeline(
        mode="labeled",
        target_stem=args.target_stem,
    )


if __name__ == "__main__":
    main()
