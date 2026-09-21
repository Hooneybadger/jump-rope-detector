"""Build compact, transparent encouragement GIFs from the source illustrations."""

from __future__ import annotations

import math
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
STATIC = ROOT / "static"
MODES = ("basic", "alternating", "double")
CANVAS_SIZE = 320
FRAME_COUNT = 12


def _trim_and_fit(source: Image.Image) -> Image.Image:
    source = source.convert("RGBA")
    bounds = source.getbbox()
    if bounds:
        source = source.crop(bounds)
    source.thumbnail((280, 280), Image.Resampling.LANCZOS)
    return source


def _gif_frame(frame: Image.Image) -> Image.Image:
    """Reserve palette index 255 for transparency on every GIF frame."""
    alpha = frame.getchannel("A")
    paletted = frame.convert("RGB").quantize(colors=255, method=Image.Quantize.MEDIANCUT)
    transparent_pixels = alpha.point(lambda value: 255 if value < 16 else 0)
    paletted.paste(255, mask=transparent_pixels)
    return paletted


def build(mode: str) -> None:
    source_path = STATIC / f"cheer-{mode}-poster.png"
    source = _trim_and_fit(Image.open(source_path))
    frames: list[Image.Image] = []
    for index in range(FRAME_COUNT):
        phase = (index / FRAME_COUNT) * math.tau
        bounce = round(-8 * abs(math.sin(phase)))
        scale = 1 + (0.018 * math.sin(phase))
        size = (max(1, round(source.width * scale)), max(1, round(source.height * scale)))
        athlete = source.resize(size, Image.Resampling.LANCZOS)
        frame = Image.new("RGBA", (CANVAS_SIZE, CANVAS_SIZE), (0, 0, 0, 0))
        x = (CANVAS_SIZE - athlete.width) // 2
        y = (CANVAS_SIZE - athlete.height) // 2 + bounce
        frame.alpha_composite(athlete, (x, y))
        frames.append(frame)

    frames[0].save(source_path, optimize=True)
    gif_frames = [_gif_frame(frame) for frame in frames]
    gif_frames[0].save(
        STATIC / f"cheer-{mode}.gif",
        save_all=True,
        append_images=gif_frames[1:],
        duration=85,
        loop=0,
        disposal=2,
        transparency=255,
        optimize=False,
    )


if __name__ == "__main__":
    for mode_name in MODES:
        build(mode_name)
