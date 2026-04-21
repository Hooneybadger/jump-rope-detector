"""
Jump Rope Detector — Unified Launcher
Material Design 3 dark theme, OpenCV-based menu + detector runner.
"""
from __future__ import annotations

import argparse
import sys
import time
from enum import Enum
from pathlib import Path
from typing import Optional

import cv2
import mediapipe as mp
import numpy as np

# ── project root on path ────────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


# ═══════════════════════════════════════════════════════════════════════════
#  COLOR PALETTE  (all BGR)
# ═══════════════════════════════════════════════════════════════════════════
_BG          = (18,  18,  18)   # #121212  surface background
_SURFACE     = (31,  27,  34)   # #1C1B22  card surface
_SURFACE_HI  = (44,  40,  50)   # slightly lighter card on hover
_OUTLINE     = (79,  69,  83)   # #49454F  card border
_ON_SURFACE  = (229, 225, 230)  # #E6E1E5  primary text
_ON_MUTED    = (147, 143, 153)  # #938F99  secondary text

# phase colors
_PHASE_COUNT   = (105, 200, 100)  # green
_PHASE_CDWN    = (210, 165,  80)  # amber/gold
_PHASE_SEARCH  = (140, 135, 155)  # neutral grey

# per-jump accent (BGR)
_ACC_BASIC = (200, 110, 120)   # violet-rose
_ACC_ALT   = (210, 165,  75)   # warm amber
_ACC_DBL   = ( 90, 200, 160)   # teal

WINDOW_W, WINDOW_H = 820, 620
WINDOW_NAME = "Jump Rope Detector"


# ═══════════════════════════════════════════════════════════════════════════
#  JUMP TYPE REGISTRY
# ═══════════════════════════════════════════════════════════════════════════
class JumpType(Enum):
    BASIC       = 1
    ALTERNATING = 2
    DOUBLE      = 3


_JUMP_META = {
    JumpType.BASIC: {
        "name_ko":  "모아뛰기",
        "name_en":  "Basic Jump",
        "desc":     "양발 모아 동시에 뛰기  ·  Both feet together",
        "desc_cv":  "Both feet together",        # cv2.putText: ASCII only
        "key":      "1",
        "accent":   _ACC_BASIC,
        "module":   "basic_jump",
    },
    JumpType.ALTERNATING: {
        "name_ko":  "번갈아뛰기",
        "name_en":  "Alternating Jump",
        "desc":     "왼발·오른발 교차  ·  Left-right alternation",
        "desc_cv":  "Left-right alternation",    # cv2.putText: ASCII only
        "key":      "2",
        "accent":   _ACC_ALT,
        "module":   "alternating_jump",
    },
    JumpType.DOUBLE: {
        "name_ko":  "이중뛰기",
        "name_en":  "Double Under",
        "desc":     "줄 두 번 통과  ·  Rope passes twice per jump",
        "desc_cv":  "Rope passes twice per jump", # cv2.putText: ASCII only
        "key":      "3",
        "accent":   _ACC_DBL,
        "module":   "double_jump",
    },
}


# ═══════════════════════════════════════════════════════════════════════════
#  DRAWING PRIMITIVES
# ═══════════════════════════════════════════════════════════════════════════

def _blend_rect(frame, tl, br, color, alpha: float) -> None:
    ov = frame.copy()
    cv2.rectangle(ov, tl, br, color, -1)
    cv2.addWeighted(ov, alpha, frame, 1.0 - alpha, 0.0, dst=frame)


def _rounded_rect(
    frame,
    tl: tuple[int, int],
    br: tuple[int, int],
    color: tuple[int, int, int],
    radius: int = 14,
    alpha: float = 1.0,
) -> None:
    x1, y1 = tl
    x2, y2 = br
    r = min(radius, (x2 - x1) // 2, (y2 - y1) // 2)
    ov = frame.copy()
    cv2.rectangle(ov, (x1 + r, y1), (x2 - r, y2), color, -1)
    cv2.rectangle(ov, (x1, y1 + r), (x2, y2 - r), color, -1)
    for cx, cy in [(x1 + r, y1 + r), (x2 - r, y1 + r),
                   (x1 + r, y2 - r), (x2 - r, y2 - r)]:
        cv2.circle(ov, (cx, cy), r, color, -1)
    cv2.addWeighted(ov, alpha, frame, 1.0 - alpha, 0.0, dst=frame)


def _text(
    frame,
    text: str,
    origin: tuple[int, int],
    scale: float,
    color: tuple[int, int, int],
    thickness: int = 1,
    shadow: bool = True,
) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    if shadow:
        cv2.putText(frame, text, (origin[0] + 1, origin[1] + 1),
                    font, scale, (4, 4, 4), thickness + 2, cv2.LINE_AA)
    cv2.putText(frame, text, origin, font, scale, color, thickness, cv2.LINE_AA)


def _text_size(text: str, scale: float, thickness: int = 1) -> tuple[int, int]:
    (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    return w, h


def _pill(
    frame,
    text: str,
    cx: int, cy: int,
    scale: float,
    text_color: tuple[int, int, int],
    bg_color: tuple[int, int, int],
    thickness: int = 1,
    pad_x: int = 10,
    pad_y: int = 6,
) -> None:
    tw, th = _text_size(text, scale, thickness)
    x1 = cx - tw // 2 - pad_x
    y1 = cy - th // 2 - pad_y
    x2 = cx + tw // 2 + pad_x
    y2 = cy + th // 2 + pad_y
    _rounded_rect(frame, (x1, y1), (x2, y2), bg_color, radius=(y2 - y1) // 2, alpha=0.92)
    _text(frame, text, (cx - tw // 2, cy + th // 2), scale, text_color, thickness=thickness, shadow=False)


def _progress_bar(
    frame,
    tl: tuple[int, int],
    size: tuple[int, int],
    progress: float,
    fill_color: tuple[int, int, int],
    bg_color: tuple[int, int, int] = (40, 38, 46),
    height: int = 8,
) -> None:
    x, y = tl
    w, h = size
    progress = max(0.0, min(1.0, progress))
    r = h // 2
    _rounded_rect(frame, (x, y), (x + w, y + h), bg_color, radius=r, alpha=0.9)
    fill_w = max(0, int(w * progress))
    if fill_w >= r * 2:
        _rounded_rect(frame, (x, y), (x + fill_w, y + h), fill_color, radius=r, alpha=1.0)
    elif fill_w > 0:
        cv2.circle(frame, (x + r, y + r), r, fill_color, -1)


def _ease_out(t: float) -> float:
    t = max(0.0, min(1.0, t))
    return 1.0 - (1.0 - t) ** 2


def _phase_color(phase: str) -> tuple[int, int, int]:
    if phase == "COUNTING":
        return _PHASE_COUNT
    if phase == "COUNTDOWN":
        return _PHASE_CDWN
    return _PHASE_SEARCH


# ═══════════════════════════════════════════════════════════════════════════
#  MENU SCREEN
# ═══════════════════════════════════════════════════════════════════════════
_CARD_MARGIN  = 36
_CARD_H       = 148
_CARD_GAP     = 14
_HEADER_H     = 118


def _card_rect(idx: int) -> tuple[int, int, int, int]:
    cards_h = 3 * _CARD_H + 2 * _CARD_GAP
    top = _HEADER_H + (WINDOW_H - _HEADER_H - cards_h - 48) // 2
    y = top + idx * (_CARD_H + _CARD_GAP)
    return (_CARD_MARGIN, y, WINDOW_W - _CARD_MARGIN, y + _CARD_H)


def _build_menu_frame(hovered: Optional[JumpType]) -> np.ndarray:
    frame = np.full((WINDOW_H, WINDOW_W, 3), _BG, dtype=np.uint8)

    # ── header ──────────────────────────────────────────────────────────
    _blend_rect(frame, (0, 0), (WINDOW_W, _HEADER_H), _SURFACE, 0.95)
    cv2.rectangle(frame, (0, 0), (WINDOW_W, 4), (160, 90, 110), -1)  # top accent line

    title = "Jump Rope Detector"
    tw, _ = _text_size(title, 1.05, 2)
    _text(frame, title, ((WINDOW_W - tw) // 2, 62), 1.05, _ON_SURFACE, thickness=2)

    sub = "Select a jump type to begin"
    sw, _ = _text_size(sub, 0.50, 1)
    _text(frame, sub, ((WINDOW_W - sw) // 2, 93), 0.50, _ON_MUTED, thickness=1, shadow=False)

    # ── cards ────────────────────────────────────────────────────────────
    for i, jt in enumerate(JumpType):
        meta   = _JUMP_META[jt]
        accent = meta["accent"]
        x1, y1, x2, y2 = _card_rect(i)
        is_hov = hovered == jt

        bg = _SURFACE_HI if is_hov else _SURFACE
        _rounded_rect(frame, (x1, y1), (x2, y2), bg, radius=16, alpha=0.96)

        # border
        border_col = accent if is_hov else _OUTLINE
        cv2.rectangle(frame, (x1, y1), (x2, y2), border_col, 2 if is_hov else 1)

        # left accent stripe
        cv2.rectangle(frame, (x1 + 10, y1 + 18), (x1 + 14, y2 - 18), accent, -1)

        # key badge circle
        kx, ky = x1 + 52, (y1 + y2) // 2
        cv2.circle(frame, (kx, ky), 24, accent, -1)
        cv2.circle(frame, (kx, ky), 24, (255, 255, 255) if is_hov else bg, 2)
        kt = meta["key"]
        ktw, kth = _text_size(kt, 0.78, 2)
        _text(frame, kt, (kx - ktw // 2, ky + kth // 2), 0.78, (15, 15, 15), thickness=2, shadow=False)

        # text block  (cv2.putText: ASCII only — no Korean, no middle-dot)
        tx = x1 + 94
        _text(frame, meta["name_en"],  (tx, y1 + 52), 0.88, _ON_SURFACE, thickness=2)
        _text(frame, meta["desc_cv"],  (tx, y1 + 90), 0.48, _ON_MUTED, thickness=1, shadow=False)

        # hover chevron
        if is_hov:
            ax, ay = x2 - 34, (y1 + y2) // 2
            pts = np.array([[ax, ay - 13], [ax + 15, ay], [ax, ay + 13]], dtype=np.int32)
            cv2.fillPoly(frame, [pts], accent)

    # ── footer ───────────────────────────────────────────────────────────
    hint = "Press  1 / 2 / 3  to select   |   Q  to quit"
    hw, _ = _text_size(hint, 0.50, 1)
    _text(frame, hint, ((WINDOW_W - hw) // 2, WINDOW_H - 20), 0.50, _ON_MUTED, thickness=1, shadow=False)

    return frame


def run_menu() -> Optional[JumpType]:
    """Render the menu and return selected JumpType, or None to quit."""
    state: dict = {"selected": None}

    def _on_mouse(event, x, y, flags, param):
        param["mx"], param["my"] = x, y
        if event == cv2.EVENT_LBUTTONDOWN:
            for i, jt in enumerate(JumpType):
                x1, y1, x2, y2 = _card_rect(i)
                if x1 <= x <= x2 and y1 <= y <= y2:
                    param["selected"] = jt

    mouse: dict = {"mx": 0, "my": 0, "selected": None}
    cv2.setMouseCallback(WINDOW_NAME, _on_mouse, param=mouse)

    while True:
        hov = None
        for i, jt in enumerate(JumpType):
            x1, y1, x2, y2 = _card_rect(i)
            if x1 <= mouse["mx"] <= x2 and y1 <= mouse["my"] <= y2:
                hov = jt
                break

        frame = _build_menu_frame(hov)
        cv2.imshow(WINDOW_NAME, frame)

        key = cv2.waitKey(16) & 0xFF
        if key in (ord("q"), 27):
            return None
        if key == ord("1"):
            return JumpType.BASIC
        if key == ord("2"):
            return JumpType.ALTERNATING
        if key == ord("3"):
            return JumpType.DOUBLE

        if mouse["selected"] is not None:
            sel = mouse["selected"]
            mouse["selected"] = None
            return sel


# ═══════════════════════════════════════════════════════════════════════════
#  COUNTING OVERLAY  (drawn on live camera frame)
# ═══════════════════════════════════════════════════════════════════════════

def _draw_counting_overlay(
    frame,
    stream_state,
    running_count: int,
    count_ready: bool,
    countdown_total_sec: float,
    pulse_progress: float,
    jump_type: JumpType,
    elapsed_sec: float,
) -> None:
    h, w = frame.shape[:2]
    meta   = _JUMP_META[jump_type]
    jt_acc = meta["accent"]
    p_col  = _phase_color(stream_state.phase)
    pulse  = _ease_out(pulse_progress)

    # ── top-left status card ─────────────────────────────────────────────
    px, py = 16, 16
    pw = min(310, w - 32)
    ph = 132

    _rounded_rect(frame, (px, py), (px + pw, py + ph), (26, 24, 32), radius=16, alpha=0.82)

    # accent stripe
    cv2.rectangle(frame, (px + 11, py + 16), (px + 15, py + ph - 16), jt_acc, -1)

    # pulse flash border
    if pulse > 0.01:
        fc = tuple(int(p_col[c] * pulse + 120 * (1 - pulse)) for c in range(3))
        cv2.rectangle(frame, (px, py), (px + pw, py + ph), fc, 2)

    # "Count" label
    _text(frame, "Count", (px + 28, py + 30), 0.52, (160, 155, 175), thickness=1, shadow=False)

    # large count number
    cs = 1.55 + 0.22 * pulse
    ct = 2 + int(round(pulse))
    _text(frame, str(running_count), (px + 28, py + 88), cs, _ON_SURFACE, thickness=ct)

    # phase pill (right side)
    _pill(frame, stream_state.phase, px + pw - 58, py + 30, 0.44,
          (14, 14, 14), p_col, thickness=1, pad_x=8, pad_y=5)

    # status text
    status = "Ready" if count_ready else "Align body"
    _text(frame, status, (px + 148, py + 64), 0.50, (195, 190, 210), thickness=1, shadow=False)

    # elapsed timer
    mm = int(elapsed_sec) // 60
    ss = int(elapsed_sec) % 60
    _text(frame, f"{mm:02d}:{ss:02d}", (px + 148, py + 90), 0.54, (160, 155, 170), thickness=1, shadow=False)

    # progress bar
    if stream_state.phase == "SEARCHING":
        pv = stream_state.ready_progress
    elif stream_state.phase == "COUNTDOWN":
        pv = 1.0 - min(1.0, stream_state.countdown_remaining_sec / countdown_total_sec)
    else:
        pv = 1.0
    _progress_bar(frame, (px + 28, py + 110), (pw - 44, 9), pv, p_col)

    # ── countdown centre overlay ──────────────────────────────────────────
    if stream_state.phase == "COUNTDOWN":
        n = max(1, int(stream_state.countdown_remaining_sec) + 1)
        bw2, bh2 = 200, 88
        bx2 = (w - bw2) // 2
        by2 = (h - bh2) // 2
        _rounded_rect(frame, (bx2, by2), (bx2 + bw2, by2 + bh2), (26, 24, 32), radius=18, alpha=0.88)
        msg = f"Starting in  {n}"
        mw, _ = _text_size(msg, 0.80, 2)
        _text(frame, msg, ((w - mw) // 2, by2 + 56), 0.80, _ON_SURFACE, thickness=2)

    # ── jump-type badge  (bottom-left) ───────────────────────────────────
    bt = meta["name_en"]   # cv2.putText: ASCII only
    btw, bth = _text_size(bt, 0.47, 1)
    bx3, by3 = 16, h - 20
    _rounded_rect(frame, (bx3 - 6, by3 - bth - 5), (bx3 + btw + 6, by3 + 5),
                  (26, 24, 32), radius=8, alpha=0.78)
    _text(frame, bt, (bx3, by3), 0.47, jt_acc, thickness=1, shadow=False)

    # ── quit hint  (bottom-right) ─────────────────────────────────────────
    hint = "Q  stop & return"
    htw, hth = _text_size(hint, 0.47, 1)
    hx, hy = w - htw - 20, h - 20
    _rounded_rect(frame, (hx - 8, hy - hth - 5), (hx + htw + 8, hy + 5),
                  (26, 24, 32), radius=8, alpha=0.78)
    _text(frame, hint, (hx, hy), 0.47, _ON_MUTED, thickness=1, shadow=False)


# ═══════════════════════════════════════════════════════════════════════════
#  RESULTS SCREEN
# ═══════════════════════════════════════════════════════════════════════════

def show_results(final_count: int, jump_type: JumpType, duration_sec: float) -> None:
    meta   = _JUMP_META[jump_type]
    accent = meta["accent"]
    deadline = time.monotonic() + 3.5

    while time.monotonic() < deadline:
        frame = np.full((WINDOW_H, WINDOW_W, 3), _BG, dtype=np.uint8)
        cv2.rectangle(frame, (0, 0), (WINDOW_W, 4), accent, -1)

        cx, cy = WINDOW_W // 2, WINDOW_H // 2
        cw2, ch2 = 500, 320

        # centre card
        _rounded_rect(frame, (cx - cw2 // 2, cy - ch2 // 2),
                      (cx + cw2 // 2, cy + ch2 // 2), _SURFACE, radius=20, alpha=0.96)
        # top stripe inside card
        _rounded_rect(frame, (cx - cw2 // 2, cy - ch2 // 2),
                      (cx + cw2 // 2, cy - ch2 // 2 + 6), accent, radius=0, alpha=1.0)

        # "Session Complete"
        t1 = "Session Complete"
        t1w, _ = _text_size(t1, 0.78, 2)
        _text(frame, t1, (cx - t1w // 2, cy - 100), 0.78, _ON_SURFACE, thickness=2)

        # jump type line
        jt_line = f"{meta['name_ko']}   {meta['name_en']}"
        jtw, _ = _text_size(jt_line, 0.56, 1)
        _text(frame, jt_line, (cx - jtw // 2, cy - 62), 0.56, accent, thickness=1, shadow=False)

        # big count
        cnt_str = str(final_count)
        cnt_w, _ = _text_size(cnt_str, 2.8, 3)
        _text(frame, cnt_str, (cx - cnt_w // 2, cy + 52), 2.8, _ON_SURFACE, thickness=3)

        lbl = "jumps"
        lw, _ = _text_size(lbl, 0.65, 1)
        _text(frame, lbl, (cx - lw // 2, cy + 90), 0.65, _ON_MUTED, thickness=1, shadow=False)

        # duration
        mm = int(duration_sec) // 60
        ss = int(duration_sec) % 60
        dur = f"Duration   {mm:02d}:{ss:02d}"
        dw, _ = _text_size(dur, 0.55, 1)
        _text(frame, dur, (cx - dw // 2, cy + 126), 0.55, _ON_MUTED, thickness=1, shadow=False)

        # auto-return progress bar
        rem = max(0.0, deadline - time.monotonic())
        prog = 1.0 - (rem / 3.5)
        _progress_bar(frame, (cx - 190, cy + 148), (380, 6), prog, accent)

        hint = "Press any key to return"
        hw, _ = _text_size(hint, 0.46, 1)
        _text(frame, hint, (cx - hw // 2, cy + 168), 0.46, _ON_MUTED, thickness=1, shadow=False)

        cv2.imshow(WINDOW_NAME, frame)
        key = cv2.waitKey(16) & 0xFF
        if key != 255:
            break


# ═══════════════════════════════════════════════════════════════════════════
#  DETECTOR RUNNER
# ═══════════════════════════════════════════════════════════════════════════

def run_detector(jump_type: JumpType, source: str = "0") -> tuple[int, float]:
    """
    Load the correct counter engine, run the detection loop,
    and return (final_count, duration_sec).
    """
    # ── dynamic import ────────────────────────────────────────────────────
    if jump_type == JumpType.BASIC:
        from basic_jump.counter_engine import (
            EngineConfig, PoseSignalExtractor, RealtimeCounterEngine,
            RealtimeStartGate, core_landmarks_visible,
        )
        _phase_hook   = None          # no special hook on COUNTING entry
        _uses_signal  = False

    elif jump_type == JumpType.ALTERNATING:
        from alternating_jump.counter_engine import (
            EngineConfig, PoseSignalExtractor, RealtimeCounterEngine,
            RealtimeStartGate, core_landmarks_visible,
        )
        _phase_hook   = "begin_count_phase"
        _uses_signal  = False

    else:  # DOUBLE
        from double_jump.counter_engine import (
            EngineConfig, PoseSignalExtractor, RealtimeCounterEngine,
            RealtimeStartGate, core_landmarks_visible,
        )
        _phase_hook   = "arm_for_counting"
        _uses_signal  = True          # also checks signal.detected

    # ── open capture ─────────────────────────────────────────────────────
    if source.isdigit():
        cap = cv2.VideoCapture(int(source))
        is_camera = True
    else:
        cap = cv2.VideoCapture(source)
        is_camera = False

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video source: {source!r}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)

    config    = EngineConfig()
    extractor = PoseSignalExtractor(config)
    gate      = RealtimeStartGate(
        ready_hold_seconds=1.0,
        countdown_seconds=3.0,
        ready_dropout_seconds=0.35,
    )

    mp_draw   = mp.solutions.drawing_utils
    mp_styles = mp.solutions.drawing_styles
    mp_pose   = mp.solutions.pose

    engine: RealtimeCounterEngine | None = None
    accepted_count   = 0
    pulse_total      = 10
    pulse_remaining  = 0
    frame_idx        = 0
    stream_start     = time.monotonic()
    count_start: float | None = None
    last_phase       = gate.phase

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            ts = (
                time.monotonic() - stream_start
                if is_camera
                else (frame_idx / fps if fps > 0 else 0.0)
            )

            signal, result = extractor.process_bgr_frame(frame, frame_idx, ts)

            if _uses_signal:
                count_ready = signal.detected or core_landmarks_visible(result, 0.30, 0.80)
            else:
                count_ready = core_landmarks_visible(result, 0.30, 0.80)

            stream_state = gate.update(count_ready, ts)
            phase_changed = stream_state.phase != last_phase

            # engine lifecycle
            if stream_state.phase == "SEARCHING" and not count_ready:
                engine = None
            elif engine is None and count_ready:
                engine = RealtimeCounterEngine(config)

            if phase_changed:
                if stream_state.phase == "COUNTING":
                    if engine is not None and _phase_hook:
                        getattr(engine, _phase_hook)()
                    accepted_count = 0
                    count_start = time.monotonic()
                last_phase = stream_state.phase

            if engine is not None:
                if stream_state.phase != "COUNTING":
                    engine.warmup(signal)
                else:
                    event = engine.step(signal)
                    if event is not None:
                        accepted_count    = event.running_count
                        pulse_remaining   = pulse_total

            # ── render ───────────────────────────────────────────────────
            display = frame.copy()
            if result.pose_landmarks:
                mp_draw.draw_landmarks(
                    display,
                    result.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_styles.get_default_pose_landmarks_style(),
                )

            elapsed = (
                (time.monotonic() - count_start) if count_start
                else (time.monotonic() - stream_start)
            )
            pulse_prog = pulse_remaining / pulse_total if pulse_total > 0 else 0.0

            _draw_counting_overlay(
                display, stream_state, accepted_count, count_ready,
                3.0, pulse_prog, jump_type, elapsed,
            )

            cv2.imshow(WINDOW_NAME, display)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break

            if pulse_remaining > 0:
                pulse_remaining -= 1
            frame_idx += 1

    finally:
        extractor.close()
        cap.release()

    duration = (
        (time.monotonic() - count_start) if count_start
        else (time.monotonic() - stream_start)
    )
    return accepted_count, duration


# ═══════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Jump Rope Detector — Unified Launcher"
    )
    parser.add_argument(
        "--source", default="0",
        help="Camera index (e.g. 0) or path to a video file",
    )
    args = parser.parse_args()

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)

    while True:
        jump_type = run_menu()
        if jump_type is None:
            break

        try:
            final_count, duration = run_detector(jump_type, args.source)
        except RuntimeError as e:
            # show a brief error overlay then go back to menu
            err_frame = np.full((WINDOW_H, WINDOW_W, 3), _BG, dtype=np.uint8)
            msg = f"Error: {e}"
            mw, _ = _text_size(msg, 0.60, 1)
            _text(err_frame, msg, ((WINDOW_W - mw) // 2, WINDOW_H // 2), 0.60, (80, 80, 220), thickness=1)
            cv2.imshow(WINDOW_NAME, err_frame)
            cv2.waitKey(2500)
            continue

        show_results(final_count, jump_type, duration)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
