from __future__ import annotations

import struct
import zlib
from dataclasses import dataclass, field
from datetime import datetime
from functools import lru_cache
from pathlib import Path

from . import calories

ROOT = Path(__file__).resolve().parents[1]
ASSETS = Path(__file__).resolve().parent / "report_assets"
FONT_PATH = ROOT / "static" / "fonts" / "GothicA1-Regular.ttf"
FONT_FILES = {
    "regular": ("GothicA1-Regular", FONT_PATH),
    "bold": ("GothicA1-Bold", ASSETS / "fonts" / "GothicA1-Bold-KSX1001.ttf"),
    "display": ("BlackHanSans-Regular", ASSETS / "fonts" / "BlackHanSans-KSX1001.ttf"),
    "digits": ("BarlowCondensed-Bold", ASSETS / "fonts" / "BarlowCondensed-Bold-Latin.ttf"),
}
PAGE_W, PAGE_H = 595, 842
WEEKDAYS = "월화수목금토일"

INK = "#10141a"
MUTED = "#4f5965"
FAINT = "#6b7480"
LINE = "#d5dae0"
PANEL = "#f4f5f7"
ACCENT = "#d9622a"
ACCENT_TEXT = "#b04a17"
TINT = "#fdf1ea"
BAR = "#c9ced6"
DARK = "#0b0e12"
DARK_MUTED = "#a3acb7"
WHITE = "#ffffff"


def _font_cmap(font: bytes) -> dict[int, int]:
    """Read a Unicode cmap from the bundled TrueType font."""
    num_tables = struct.unpack_from(">H", font, 4)[0]
    tables: dict[bytes, tuple[int, int]] = {}
    for index in range(num_tables):
        tag, _, offset, length = struct.unpack_from(">4sIII", font, 12 + index * 16)
        tables[tag] = (offset, length)
    cmap_offset, _ = tables[b"cmap"]
    count = struct.unpack_from(">H", font, cmap_offset + 2)[0]
    candidates: list[tuple[int, int, int]] = []
    for index in range(count):
        platform, encoding, relative = struct.unpack_from(">HHI", font, cmap_offset + 4 + index * 8)
        subtable = cmap_offset + relative
        format_number = struct.unpack_from(">H", font, subtable)[0]
        priority = 3 if (platform, encoding) == (3, 10) else 2 if platform == 0 else 1
        if format_number in {4, 12}:
            candidates.append((priority, format_number, subtable))
    if not candidates:
        raise ValueError("The PDF font has no supported Unicode cmap.")
    _, format_number, offset = max(candidates)
    mapping: dict[int, int] = {}
    if format_number == 12:
        groups = struct.unpack_from(">I", font, offset + 12)[0]
        for index in range(groups):
            start, end, glyph = struct.unpack_from(">III", font, offset + 16 + index * 12)
            for codepoint in range(start, end + 1):
                mapping[codepoint] = glyph + codepoint - start
        return mapping

    segment_count = struct.unpack_from(">H", font, offset + 6)[0] // 2
    end_codes = offset + 14
    start_codes = end_codes + segment_count * 2 + 2
    deltas = start_codes + segment_count * 2
    range_offsets = deltas + segment_count * 2
    for index in range(segment_count):
        end = struct.unpack_from(">H", font, end_codes + index * 2)[0]
        start = struct.unpack_from(">H", font, start_codes + index * 2)[0]
        delta = struct.unpack_from(">h", font, deltas + index * 2)[0]
        range_offset = struct.unpack_from(">H", font, range_offsets + index * 2)[0]
        for codepoint in range(start, end + 1):
            if codepoint == 0xFFFF:
                continue
            if range_offset == 0:
                glyph = (codepoint + delta) & 0xFFFF
            else:
                glyph_position = range_offsets + index * 2 + range_offset + (codepoint - start) * 2
                glyph = struct.unpack_from(">H", font, glyph_position)[0]
                if glyph:
                    glyph = (glyph + delta) & 0xFFFF
            if glyph:
                mapping[codepoint] = glyph
    return mapping


def _font_tables(font: bytes) -> dict[bytes, tuple[int, int]]:
    num_tables = struct.unpack_from(">H", font, 4)[0]
    tables = {}
    for index in range(num_tables):
        tag, _, offset, length = struct.unpack_from(">4sIII", font, 12 + index * 16)
        tables[tag] = (offset, length)
    return tables


def _read_png(path: Path) -> tuple[int, int, bytes, bytes]:
    """Decode an 8-bit RGB/RGBA, non-interlaced PNG into RGB and alpha planes."""
    data = path.read_bytes()
    if data[:8] != bytes.fromhex("89504e470d0a1a0a"):
        raise ValueError(f"not a PNG file: {path.name}")
    offset, idat = 8, bytearray()
    width = height = color_type = 0
    while offset < len(data):
        length, kind = struct.unpack_from(">I4s", data, offset)
        chunk = data[offset + 8:offset + 8 + length]
        if kind == b"IHDR":
            width, height, depth, color_type, _, _, interlace = struct.unpack(">IIBBBBB", chunk)
            if depth != 8 or color_type not in {2, 6} or interlace:
                raise ValueError(f"unsupported PNG layout: {path.name}")
        elif kind == b"IDAT":
            idat.extend(chunk)
        offset += 12 + length
    channels = 4 if color_type == 6 else 3
    stride = width * channels
    raw = zlib.decompress(bytes(idat))
    pixels = bytearray(stride * height)
    previous = bytearray(stride)
    for row in range(height):
        start = row * (stride + 1)
        kind, line = raw[start], bytearray(raw[start + 1:start + 1 + stride])
        for index in range(stride):
            left = line[index - channels] if index >= channels else 0
            up = previous[index]
            corner = previous[index - channels] if index >= channels else 0
            if kind == 1:
                line[index] = (line[index] + left) & 0xFF
            elif kind == 2:
                line[index] = (line[index] + up) & 0xFF
            elif kind == 3:
                line[index] = (line[index] + ((left + up) >> 1)) & 0xFF
            elif kind == 4:
                estimate = left + up - corner
                distances = (abs(estimate - left), abs(estimate - up), abs(estimate - corner))
                predictor = left if distances[0] <= distances[1] and distances[0] <= distances[2] else up if distances[1] <= distances[2] else corner
                line[index] = (line[index] + predictor) & 0xFF
        pixels[row * stride:(row + 1) * stride] = line
        previous = line
    if channels == 3:
        return width, height, bytes(pixels), bytes([255]) * (width * height)
    rgb = bytearray(width * height * 3)
    rgb[0::3], rgb[1::3], rgb[2::3] = pixels[0::4], pixels[1::4], pixels[2::4]
    return width, height, bytes(rgb), bytes(pixels[3::4])


@dataclass
class _FontFace:
    """A TrueType font with real advance widths, so text is spaced as designed."""

    name: str
    data: bytes
    cmap: dict[int, int]
    advances: list[int]
    units_per_em: int
    ascent: int
    descent: int
    bbox: tuple[int, int, int, int]
    used: dict[int, int] = field(default_factory=dict)

    @classmethod
    def load(cls, name: str, path: Path) -> _FontFace:
        data = path.read_bytes()
        tables = _font_tables(data)
        head = tables[b"head"][0]
        units_per_em = struct.unpack_from(">H", data, head + 18)[0]
        bbox = struct.unpack_from(">hhhh", data, head + 36)
        hhea = tables[b"hhea"][0]
        ascent, descent = struct.unpack_from(">hh", data, hhea + 4)
        metric_count = struct.unpack_from(">H", data, hhea + 34)[0]
        glyph_count = struct.unpack_from(">H", data, tables[b"maxp"][0] + 4)[0]
        hmtx = tables[b"hmtx"][0]
        advances = [struct.unpack_from(">H", data, hmtx + index * 4)[0] for index in range(metric_count)]
        advances += [advances[-1]] * (glyph_count - metric_count)
        return cls(name, data, _font_cmap(data), advances, units_per_em, ascent, descent, bbox)

    def covers(self, text: str) -> bool:
        return all(ord(character) in self.cmap for character in text)

    def width(self, text: str, size: float) -> float:
        total = sum(self.advances[self.cmap.get(ord(character), 0)] for character in text)
        return total * size / self.units_per_em

    def encode(self, text: str) -> str:
        glyphs = []
        for character in text:
            glyph = self.cmap.get(ord(character), self.cmap.get(ord("?"), 0))
            self.used[glyph] = ord(character)
            glyphs.append(f"{glyph:04X}")
        return "".join(glyphs)

    def scaled(self, value: float) -> int:
        return round(value * 1000 / self.units_per_em)


@lru_cache(maxsize=1)
def _font_sources() -> dict[str, tuple[str, Path]]:
    return {key: value for key, value in FONT_FILES.items() if value[1].exists()}


def _rgb(color: str) -> str:
    value = color.lstrip("#")
    red, green, blue = (int(value[index:index + 2], 16) / 255 for index in (0, 2, 4))
    return f"{red:.3f} {green:.3f} {blue:.3f}"


class _Report:
    """Minimal PDF canvas: text with real metrics, shapes, and PNG images."""

    def __init__(self) -> None:
        self.fonts = {key: _FontFace.load(name, path) for key, (name, path) in _font_sources().items()}
        self.ops: list[str] = []
        self.images: dict[str, Path] = {}

    @staticmethod
    def y(top: float) -> float:
        return PAGE_H - top

    def face(self, key: str, text: str) -> tuple[str, _FontFace]:
        face = self.fonts.get(key)
        if face is not None and face.covers(text):
            return key, face
        return "regular", self.fonts["regular"]

    def width(self, text: str, size: float, font: str = "regular") -> float:
        return self.face(font, text)[1].width(text, size)

    def text(self, value: str, x: float, top: float, size: float, font: str = "regular",
             color: str = INK, align: str = "left") -> float:
        key, face = self.face(font, value)
        width = face.width(value, size)
        if align == "right":
            x -= width
        elif align == "center":
            x -= width / 2
        self.ops.append(f"{_rgb(color)} rg BT /{key} {size:g} Tf {x:.2f} {self.y(top):.2f} Td <{face.encode(value)}> Tj ET")
        return width

    def wrap(self, value: str, size: float, max_width: float, font: str = "regular") -> list[str]:
        lines: list[str] = []
        current = ""
        for word in value.split(" "):
            candidate = f"{current} {word}".strip()
            if self.width(candidate, size, font) <= max_width or not current:
                current = candidate
            else:
                lines.append(current)
                current = word
        if current:
            lines.append(current)
        return lines

    def paragraph(self, value: str, x: float, top: float, size: float, max_width: float,
                  font: str = "regular", color: str = INK, leading: float = 1.6) -> float:
        for index, line in enumerate(self.wrap(value, size, max_width, font)):
            self.text(line, x, top + index * size * leading, size, font, color)
        return len(self.wrap(value, size, max_width, font)) * size * leading

    def rect(self, x: float, top: float, width: float, height: float, fill: str, radius: float = 0,
             stroke: str | None = None) -> None:
        bottom = self.y(top + height)
        paint = f"{_rgb(fill)} rg"
        if stroke:
            paint += f" {_rgb(stroke)} RG 0.8 w"
        if radius <= 0:
            path = f"{x:.2f} {bottom:.2f} {width:.2f} {height:.2f} re"
        else:
            r = min(radius, width / 2, height / 2)
            k = r * 0.5523
            left, right, low, high = x, x + width, bottom, bottom + height
            path = (
                f"{left + r:.2f} {low:.2f} m {right - r:.2f} {low:.2f} l "
                f"{right - r + k:.2f} {low:.2f} {right:.2f} {low + r - k:.2f} {right:.2f} {low + r:.2f} c "
                f"{right:.2f} {high - r:.2f} l {right:.2f} {high - r + k:.2f} {right - r + k:.2f} {high:.2f} {right - r:.2f} {high:.2f} c "
                f"{left + r:.2f} {high:.2f} l {left + r - k:.2f} {high:.2f} {left:.2f} {high - r + k:.2f} {left:.2f} {high - r:.2f} c "
                f"{left:.2f} {low + r:.2f} l {left:.2f} {low + r - k:.2f} {left + r - k:.2f} {low:.2f} {left + r:.2f} {low:.2f} c h"
            )
        self.ops.append(f"{paint} {path} {'B' if stroke else 'f'}")

    def line(self, x1: float, top1: float, x2: float, top2: float, color: str = LINE, width: float = 0.8) -> None:
        self.ops.append(f"{_rgb(color)} RG {width} w {x1:.2f} {self.y(top1):.2f} m {x2:.2f} {self.y(top2):.2f} l S")

    def circle(self, cx: float, ctop: float, r: float, fill: str) -> None:
        self.rect(cx - r, ctop - r, r * 2, r * 2, fill, radius=r)

    def image(self, path: Path, x: float, top: float, width: float, height: float) -> None:
        name = f"Im{len(self.images) + 1}"
        self.images[name] = path
        self.ops.append(f"q {width:.2f} 0 0 {height:.2f} {x:.2f} {self.y(top + height):.2f} cm /{name} Do Q")

    def render(self) -> bytes:
        objects: list[bytes | None] = [None, None, None]  # catalog, pages, page

        def add(obj: bytes) -> int:
            objects.append(obj)
            return len(objects)

        def stream(data: bytes, extra: str = "") -> bytes:
            packed = zlib.compress(data, 9)
            return f"<< /Length {len(packed)} /Filter /FlateDecode {extra}>>\nstream\n".encode() + packed + b"\nendstream"

        content_id = add(stream("\n".join(self.ops).encode("ascii")))
        font_refs = []
        for key, face in self.fonts.items():
            if not face.used:
                continue
            widths = " ".join(f"{glyph} [{face.scaled(face.advances[glyph])}]" for glyph in sorted(face.used))
            file_id = add(stream(face.data, f"/Length1 {len(face.data)} "))
            bbox = " ".join(str(face.scaled(value)) for value in face.bbox)
            descriptor_id = add(
                f"<< /Type /FontDescriptor /FontName /{face.name} /Flags 32 /FontBBox [{bbox}] /ItalicAngle 0 "
                f"/Ascent {face.scaled(face.ascent)} /Descent {face.scaled(face.descent)} /CapHeight 700 /StemV 80 "
                f"/FontFile2 {file_id} 0 R >>".encode()
            )
            cid_id = add(
                f"<< /Type /Font /Subtype /CIDFontType2 /BaseFont /{face.name} "
                "/CIDSystemInfo << /Registry (Adobe) /Ordering (Identity) /Supplement 0 >> "
                f"/FontDescriptor {descriptor_id} 0 R /CIDToGIDMap /Identity /DW 1000 /W [{widths}] >>".encode()
            )
            mappings = "\n".join(
                f"<{glyph:04X}> <{chr(codepoint).encode('utf-16-be').hex().upper()}>"
                for glyph, codepoint in sorted(face.used.items())
            )
            to_unicode = (
                "/CIDInit /ProcSet findresource begin\n12 dict begin\nbegincmap\n"
                "/CIDSystemInfo << /Registry (Adobe) /Ordering (UCS) /Supplement 0 >> def\n"
                f"/CMapName /{face.name}-UTF16 def\n/CMapType 2 def\n"
                "1 begincodespacerange\n<0000> <FFFF>\nendcodespacerange\n"
                f"{len(face.used)} beginbfchar\n{mappings}\nendbfchar\n"
                "endcmap\nCMapName currentdict /CMap defineresource pop\nend\nend"
            ).encode("ascii")
            unicode_id = add(stream(to_unicode))
            font_id = add(
                f"<< /Type /Font /Subtype /Type0 /BaseFont /{face.name} /Encoding /Identity-H "
                f"/DescendantFonts [{cid_id} 0 R] /ToUnicode {unicode_id} 0 R >>".encode()
            )
            font_refs.append(f"/{key} {font_id} 0 R")

        image_refs = []
        for name, path in self.images.items():
            width, height, rgb, alpha = _read_png(path)
            base = f"/Type /XObject /Subtype /Image /Width {width} /Height {height} /BitsPerComponent 8 "
            mask_id = add(stream(alpha, base + "/ColorSpace /DeviceGray "))
            image_id = add(stream(rgb, base + f"/ColorSpace /DeviceRGB /SMask {mask_id} 0 R "))
            image_refs.append(f"/{name} {image_id} 0 R")

        objects[0] = b"<< /Type /Catalog /Pages 2 0 R >>"
        objects[1] = b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>"
        objects[2] = (
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {PAGE_W} {PAGE_H}] "
            f"/Resources << /Font << {' '.join(font_refs)} >> /XObject << {' '.join(image_refs)} >> >> "
            f"/Contents {content_id} 0 R >>"
        ).encode()

        document = bytearray(b"%PDF-1.7\n%\xe2\xe3\xcf\xd3\n")
        offsets = []
        for number, obj in enumerate(objects, 1):
            offsets.append(len(document))
            document.extend(f"{number} 0 obj\n".encode())
            document.extend(obj)
            document.extend(b"\nendobj\n")
        xref = len(document)
        document.extend(f"xref\n0 {len(objects) + 1}\n".encode())
        document.extend(b"0000000000 65535 f \n")
        for offset in offsets:
            document.extend(f"{offset:010d} 00000 n \n".encode())
        document.extend(f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode())
        return bytes(document)


def _clock(seconds: int) -> str:
    seconds = max(0, int(seconds))
    return f"{seconds // 60:02d}:{seconds % 60:02d}"


def _number(value: float) -> str:
    return f"{value:,.1f}" if value < 100 else f"{value:,.0f}"


def _duration_label(minutes: float) -> str:
    if abs(minutes - round(minutes)) < 1e-6:
        return f"{round(minutes)}분"
    return _clock(round(minutes * 60))


def workout_pdf(*, workout_id: int, user_name: str, mode_name: str, count: int,
                duration: int, status_name: str, started_at: datetime,
                mode: str = "basic", target_duration: int | None = None) -> bytes:
    report = _Report()
    estimate = calories.estimate(mode, count, duration)
    tier = calories.cheer_tier(mode, count)
    local = started_at.astimezone()
    started = f"{local:%Y.%m.%d} ({WEEKDAYS[local.weekday()]}) {local:%H:%M}"
    left, right = 40, PAGE_W - 40
    width = right - left

    # Header band
    report.rect(0, 0, PAGE_W, 168, DARK)
    report.image(ASSETS / "mark-on-dark.png", left, 30, 30, 30)
    report.text("뜀결", left + 38, 51, 13, "bold", WHITE)
    report.text(f"기록 #{workout_id}", right, 51, 9, "regular", DARK_MUTED, "right")
    report.text("줄넘기 측정 리포트", left, 110, 30, "display", WHITE)
    report.text(f"{started}  ·  {user_name}", left, 140, 10, "regular", DARK_MUTED)
    chip_w = report.width(mode_name, 11, "bold") + 28
    report.rect(right - chip_w, 88, chip_w, 26, ACCENT, radius=13)
    report.text(mode_name, right - chip_w / 2, 105, 11, "bold", INK, "center")

    # Result card
    top = 192
    report.rect(left, top, width, 132, WHITE, radius=12, stroke=LINE)
    report.text("점프 횟수", left + 22, top + 28, 9.5, "bold", MUTED)
    count_text = f"{count:,}"
    count_w = report.text(count_text, left + 20, top + 104, 68, "digits", ACCENT)
    report.text("회", left + 26 + count_w, top + 102, 16, "bold", MUTED)
    stats_x = left + 300
    pace = estimate.pace_per_min
    target_text = f" / {_clock(target_duration)}" if target_duration else ""
    rows = [
        ("측정 시간", f"{_clock(duration)}{target_text}"),
        ("분당 평균", f"{round(pace)}회" if count else "0회"),
        ("측정 상태", status_name),
    ]
    for index, (label, value) in enumerate(rows):
        row_top = top + 20 + index * 34
        if index:
            report.line(stats_x, row_top - 6, right - 20, row_top - 6)
        report.text(label, stats_x, row_top + 14, 9.5, "regular", MUTED)
        report.text(value, right - 20, row_top + 15, 13, "bold", INK, "right")

    # Cheer card
    top = 340
    report.rect(left, top, width, 110, TINT, radius=12)
    report.image(ASSETS / f"tier-{tier.level}.png", left + 20, top + 19, 72, 72)
    report.text(tier.title, left + 110, top + 38, 18, "display", INK)
    report.paragraph(tier.message, left + 110, top + 60, 10, 232, "regular", MUTED)
    thresholds = calories.TIER_THRESHOLDS.get(mode, calories.TIER_THRESHOLDS["basic"])
    ladder_left, ladder_right = right - 158, right - 22
    step = (ladder_right - ladder_left) / (len(thresholds) - 1)
    report.line(ladder_left, top + 46, ladder_right, top + 46, "#efc3a8", 2)
    for index, threshold in enumerate(thresholds):
        cx = ladder_left + index * step
        reached = index <= tier.level
        report.circle(cx, top + 46, 7 if index == tier.level else 5, ACCENT if reached else "#efc3a8")
        report.text(f"{threshold}", cx, top + 70, 9, "digits", ACCENT_TEXT if reached else FAINT, "center")
    report.text("단계 기준 (회)", ladder_left - 4, top + 24, 8.5, "regular", FAINT)
    if tier.next_threshold is not None:
        report.text(f"다음 단계까지 {tier.next_threshold - count}회", ladder_right, top + 92, 9.5, "bold", ACCENT_TEXT, "right")
    else:
        report.text("최고 단계 달성", ladder_right, top + 92, 9.5, "bold", ACCENT_TEXT, "right")

    # Calories
    top = 474
    report.text("예상 칼로리 소모량", left, top + 18, 17, "display", INK)
    report.text(f"체중 {calories.REFERENCE_WEIGHT_KG} kg 기준", right, top + 16, 9, "regular", FAINT, "right")
    card_top = top + 32
    report.rect(left, card_top, 214, 214, PANEL, radius=12)
    if estimate.basis is None:
        zero_w = report.text("0", left + 18, card_top + 70, 46, "digits", FAINT)
        report.text("kcal", left + 24 + zero_w, card_top + 68, 13, "bold", FAINT)
        report.paragraph("세어진 점프가 없어 칼로리를 계산하지 않았습니다.", left + 18, card_top + 98, 9.5, 178, "regular", MUTED)
    else:
        kcal_text = _number(estimate.kcal)
        kcal_w = report.text(kcal_text, left + 18, card_top + 70, 46, "digits", INK)
        report.text("kcal", left + 24 + kcal_w, card_top + 68, 13, "bold", MUTED)
        report.text(f"MET {estimate.basis.met:g}", left + 18, card_top + 96, 11, "bold", ACCENT_TEXT)
        report.text(f"Compendium {estimate.basis.code}", left + 18 + report.width(f"MET {estimate.basis.met:g}", 11, "bold") + 8,
                    card_top + 96, 9, "regular", FAINT)
        report.paragraph(estimate.basis.label, left + 18, card_top + 116, 9.5, 180, "regular", MUTED)
        report.text("체중별 환산 (kcal)", left + 18, card_top + 150, 8.5, "bold", MUTED)
        cell_w = 178 / len(calories.WEIGHT_TABLE_KG)
        for index, weight in enumerate(calories.WEIGHT_TABLE_KG):
            cx = left + 18 + cell_w * index + cell_w / 2
            highlight = weight == calories.REFERENCE_WEIGHT_KG
            if highlight:
                report.rect(cx - cell_w / 2 + 2, card_top + 160, cell_w - 4, 40, WHITE, radius=6)
            report.text(f"{weight}kg", cx, card_top + 175, 8, "regular", FAINT, "center")
            report.text(_number(estimate.kcal_for(estimate.minutes, weight)), cx, card_top + 193, 12, "digits",
                        ACCENT_TEXT if highlight else INK, "center")

    chart_left, chart_right = left + 236, right
    report.text("같은 페이스로 계속 뛰면", chart_left, card_top + 14, 10, "bold", INK)
    report.text("측정 시간별 예상 칼로리와 횟수", chart_left, card_top + 30, 8.5, "regular", FAINT)
    base_top, chart_h = card_top + 176, 118
    report.line(chart_left, base_top, chart_right, base_top, LINE, 1)
    if estimate.basis is not None:
        bars = [("이번 기록", estimate.minutes, estimate.kcal, count, True)]
        planned = [1, 3, 5, 10]
        if target_duration and target_duration / 60 not in planned:
            planned.append(target_duration / 60)
        for minutes in sorted(planned):
            if abs(minutes - estimate.minutes) < 0.05:
                continue
            bars.append((_duration_label(minutes), minutes, estimate.kcal_for(minutes), round(pace * minutes), False))
        bars = sorted(bars, key=lambda bar: bar[1])[:6]
        peak = max(bar[2] for bar in bars) or 1
        slot = (chart_right - chart_left) / len(bars)
        bar_w = min(30, slot * 0.5)
        for index, (label, _, value, jumps, current) in enumerate(bars):
            cx = chart_left + slot * index + slot / 2
            height = max(2, value / peak * chart_h)
            report.rect(cx - bar_w / 2, base_top - height, bar_w, height, ACCENT if current else BAR, radius=3)
            report.text(_number(value), cx, base_top - height - 5, 10, "digits", ACCENT_TEXT if current else INK, "center")
            report.text(label, cx, base_top + 14, 8.5, "bold", INK if current else MUTED, "center")
            report.text(f"약 {jumps:,}회", cx, base_top + 27, 8, "regular", FAINT, "center")
    else:
        report.text("계산할 기록이 없습니다.", chart_left, base_top - 50, 9.5, "regular", FAINT)

    # Method and references
    top = 736
    report.line(left, top, right, top)
    notes = [
        "산출식: kcal = MET × 3.5 × 체중(kg) ÷ 200 × 운동 시간(분). 실제 소모량은 체중, 체력, 쉬는 시간에 따라 달라지는 추정치입니다.",
        "MET: Herrmann SD 외, 2024 Adult Compendium of Physical Activities, J Sport Health Sci 2024;13(1):6-12. "
        "모아뛰기는 분당 횟수로 8.3·11.8·12.3, 이중뛰기는 10.0을 적용합니다.",
        "번갈아뛰기는 같은 줄 회전 속도에서 모아뛰기와 산소섭취량 차이가 없다는 연구(최대혁, 2004, 운동과학 13(1):25-34)에 따라 같은 기준을 적용합니다.",
    ]
    note_top = top + 16
    for note in notes:
        note_top += report.paragraph(note, left, note_top, 7.5, width, "regular", FAINT, 1.5) + 2
    report.text("뜀결  ·  동작을 읽고, 리듬을 기록하다", left, 826, 8, "regular", FAINT)
    report.text("PAGE 1 / 1", right, 826, 8, "digits", FAINT, "right")
    return report.render()
