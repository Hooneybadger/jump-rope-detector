from __future__ import annotations

import struct
from datetime import datetime
from pathlib import Path


FONT_PATH = Path(__file__).resolve().parents[1] / "static" / "fonts" / "GothicA1-Regular.ttf"


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


def workout_pdf(*, workout_id: int, user_name: str, mode_name: str, count: int,
                duration: int, status_name: str, started_at: datetime) -> bytes:
    font = FONT_PATH.read_bytes()
    cmap = _font_cmap(font)
    started = started_at.astimezone().strftime("%Y.%m.%d %H:%M")
    used: dict[int, int] = {}

    def encoded(value: str) -> str:
        glyphs: list[str] = []
        for character in value:
            codepoint = ord(character)
            glyph = cmap.get(codepoint, cmap.get(ord("?"), 0))
            used[glyph] = codepoint
            glyphs.append(f"{glyph:04X}")
        return "".join(glyphs)

    content_parts = [
        "0.094 0.094 0.094 rg", "0 682 595 160 re f",
        "0.902 0.098 0.098 rg", "36 696 72 4 re f",
        "0.953 0.965 0.973 rg", "36 548 523 104 re f",
        "0.851 0.878 0.906 rg", "297 564 1 72 re f",
        "0.851 0.878 0.906 RG", "36 382 523 1 re S",
    ]

    def add_text(value: str, x: int, y: int, size: int, color: str = "0.094 0.094 0.094") -> None:
        content_parts.append(f"{color} rg BT /F1 {size} Tf {x} {y} Td <{encoded(value)}> Tj ET")

    add_text("뜀결", 36, 804, 10, "1 1 1")
    add_text("줄넘기 측정 리포트", 36, 756, 24, "1 1 1")
    add_text(f"{started}  ·  기록 {workout_id}", 36, 720, 9, "0.608 0.608 0.608")

    add_text("점프 횟수", 52, 622, 9, "0.349 0.404 0.478")
    add_text(f"{count:,}회", 52, 580, 24)
    add_text("측정 시간", 322, 622, 9, "0.349 0.404 0.478")
    add_text(f"{duration // 60:02d}:{duration % 60:02d}", 322, 580, 24)

    add_text("측정 정보", 36, 510, 14)
    add_text(f"사용자  {user_name}", 36, 478, 10)
    add_text(f"측정 종목  {mode_name}", 300, 478, 10)
    add_text(f"측정 상태  {status_name}", 36, 452, 10)
    add_text(f"기록 번호  {workout_id}", 300, 452, 10)

    add_text("기록 안내", 36, 346, 14)
    add_text("이 문서는 측정이 완료된 시점의 점프 횟수와 측정 시간을 기록합니다.", 36, 314, 10)
    add_text("상세 기록은 뜀결의 전체 측정 결과에서 다시 확인할 수 있습니다.", 36, 286, 10, "0.349 0.404 0.478")
    add_text("뜀결  ·  동작을 읽고, 리듬을 기록하다", 36, 54, 8, "0.349 0.404 0.478")
    content = "\n".join(content_parts).encode("ascii")
    mappings = []
    for glyph, codepoint in sorted(used.items()):
        unicode_hex = chr(codepoint).encode("utf-16-be").hex().upper()
        mappings.append(f"<{glyph:04X}> <{unicode_hex}>")
    to_unicode = (
        "/CIDInit /ProcSet findresource begin\n12 dict begin\nbegincmap\n"
        "/CIDSystemInfo << /Registry (Adobe) /Ordering (UCS) /Supplement 0 >> def\n"
        "/CMapName /GothicA1-UTF16 def\n/CMapType 2 def\n"
        "1 begincodespacerange\n<0000> <FFFF>\nendcodespacerange\n"
        f"{len(mappings)} beginbfchar\n" + "\n".join(mappings) +
        "\nendbfchar\nendcmap\nCMapName currentdict /CMap defineresource pop\nend\nend"
    ).encode("ascii")
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 595 842] /Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >>",
        b"<< /Length " + str(len(content)).encode() + b" >>\nstream\n" + content + b"\nendstream",
        b"<< /Type /Font /Subtype /Type0 /BaseFont /GothicA1 /Encoding /Identity-H /DescendantFonts [6 0 R] /ToUnicode 9 0 R >>",
        b"<< /Type /Font /Subtype /CIDFontType2 /BaseFont /GothicA1 /CIDSystemInfo << /Registry (Adobe) /Ordering (Identity) /Supplement 0 >> /FontDescriptor 7 0 R /CIDToGIDMap /Identity /DW 1000 >>",
        b"<< /Type /FontDescriptor /FontName /GothicA1 /Flags 32 /FontBBox [-1000 -400 2000 1200] /ItalicAngle 0 /Ascent 1000 /Descent -300 /CapHeight 750 /StemV 80 /FontFile2 8 0 R >>",
        b"<< /Length " + str(len(font)).encode() + b" /Length1 " + str(len(font)).encode() + b" >>\nstream\n" + font + b"\nendstream",
        b"<< /Length " + str(len(to_unicode)).encode() + b" >>\nstream\n" + to_unicode + b"\nendstream",
    ]
    document = bytearray(b"%PDF-1.7\n%\xe2\xe3\xcf\xd3\n")
    offsets = [0]
    for number, obj in enumerate(objects, 1):
        offsets.append(len(document))
        document.extend(f"{number} 0 obj\n".encode())
        document.extend(obj)
        document.extend(b"\nendobj\n")
    xref = len(document)
    document.extend(f"xref\n0 {len(objects) + 1}\n".encode())
    document.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        document.extend(f"{offset:010d} 00000 n \n".encode())
    document.extend(f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode())
    return bytes(document)
