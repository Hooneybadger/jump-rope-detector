"""Compatibility module for the refactored jump rope tracker."""

from jump_rope_detection import *  # noqa: F401,F403
from jump_rope_eval import *  # noqa: F401,F403
from jump_rope_pipeline import main, run_pipeline, run_with_args
from jump_rope_settings import *  # noqa: F401,F403


if __name__ == "__main__":
    main()
