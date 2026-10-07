import re
import zlib
from datetime import UTC, datetime

import pytest

import app.pdf as pdf_module


def build(mode="alternating", count=112, duration=60, target=60, status="완료"):
    return pdf_module.workout_pdf(
        workout_id=7,
        user_name="점퍼",
        mode_name={"basic": "모아뛰기", "alternating": "번갈아뛰기", "double": "이중뛰기"}[mode],
        count=count,
        duration=duration,
        status_name=status,
        started_at=datetime(2026, 9, 21, tzinfo=UTC),
        mode=mode,
        target_duration=target,
    )


def streams(content: bytes) -> list[bytes]:
    return [zlib.decompress(match) for match in re.findall(rb"/FlateDecode [^>]*>>\nstream\n(.*?)\nendstream", content, re.S)]


@pytest.mark.parametrize("mode", ["basic", "alternating", "double"])
def test_report_renders_for_every_mode(mode):
    content = build(mode=mode)
    assert content.startswith(b"%PDF-1.7")
    assert content.rstrip().endswith(b"%%EOF")
    assert b"/FontFile2" in content
    assert b"/Subtype /Image" in content and b"/SMask" in content


def test_fonts_carry_real_glyph_widths_instead_of_a_fixed_em():
    content = build()
    widths = re.findall(rb"/W \[([^\]]*\])+\s*\]", content)
    assert widths, "every embedded font needs a /W array"
    # Hangul and Latin glyphs must not all share the 1000-unit default advance.
    advances = {int(value) for value in re.findall(rb"\[(\d+)\]", b" ".join(re.findall(rb"/W \[(.*?)\] >>", content, re.S)))}
    assert len(advances) > 3
    assert min(advances) < 600


def test_report_content_includes_calorie_chart_and_cheer_tier():
    content = build(mode="basic", count=112, duration=60)
    page = next(data for data in streams(content) if b" Tj ET" in data)
    assert page.count(b" Tj ET") > 40
    assert page.count(b" re f") + page.count(b" c h f") > 10
    assert b"/Im1 Do" in page and b"/Im2 Do" in page


def test_empty_session_report_still_renders():
    content = build(count=0, duration=0, status="중단")
    assert content.startswith(b"%PDF-1.7")
