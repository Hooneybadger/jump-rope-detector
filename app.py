"""
Jump Rope Detector — Streamlit App
Material Design 3 dark theme  ·  streamlit-webrtc 기반 실시간 카운팅
"""
from __future__ import annotations

import sys
import threading
import time
from pathlib import Path
from typing import Optional

import av
import cv2
import mediapipe as mp
import numpy as np
import streamlit as st
from streamlit_webrtc import VideoProcessorBase, WebRtcMode, webrtc_streamer

# ── project root on sys.path ──────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# ── reuse drawing helpers & constants from run.py ─────────────────────────────
from run import (
    JumpType,
    _JUMP_META,
    _draw_counting_overlay,
)

# ─────────────────────────────────────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _bgr_to_hex(bgr: tuple[int, int, int]) -> str:
    b, g, r = bgr
    return f"#{r:02x}{g:02x}{b:02x}"


RTC_CONFIG = {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}


# ─────────────────────────────────────────────────────────────────────────────
#  PAGE CONFIG  (반드시 가장 먼저 호출)
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Jump Rope Detector",
    page_icon="💪",
    layout="centered",
    initial_sidebar_state="collapsed",
)


# ─────────────────────────────────────────────────────────────────────────────
#  CSS  — Material Design 3 dark theme
# ─────────────────────────────────────────────────────────────────────────────
def _inject_css() -> None:
    st.markdown("""
<style>
/* ── base ──────────────────────────────────────────────────────────────── */
html, body,
[data-testid="stAppViewContainer"],
[data-testid="stMain"],
.main .block-container {
    background-color: #121212 !important;
    color: #E6E1E5;
}
[data-testid="stHeader"]   { background: transparent !important; }
[data-testid="stToolbar"]  { display: none !important; }
[data-testid="stDecoration"] { display: none !important; }
footer     { visibility: hidden !important; }
#MainMenu  { visibility: hidden !important; }

.block-container { padding-top: 2.5rem !important; }

/* ── app header ─────────────────────────────────────────────────────────── */
.app-title {
    text-align: center;
    font-size: 2rem;
    font-weight: 700;
    color: #E6E1E5;
    letter-spacing: 0.02em;
    margin: 0 0 4px;
}
.app-subtitle {
    text-align: center;
    font-size: 0.85rem;
    color: #938F99;
    margin: 0 0 2rem;
}

/* ── jump cards (menu) ───────────────────────────────────────────────────── */
.jump-card {
    background: #1C1B22;
    border-radius: 16px;
    border: 1px solid #3A3845;
    padding: 22px 18px 14px;
    margin-bottom: 6px;
    min-height: 198px;
    display: flex;
    flex-direction: column;
}
.card-badge {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 34px; height: 34px;
    border-radius: 50%;
    font-size: 1.05rem;
    font-weight: 700;
    color: #121212;
    margin-bottom: 10px;
}
.card-name-ko {
    font-size: 1.1rem;
    font-weight: 700;
    color: #E6E1E5;
    line-height: 1.25;
}
.card-name-en {
    font-size: 0.78rem;
    font-weight: 600;
}
.card-desc {
    font-size: 0.70rem;
    color: #79747E;
    line-height: 1.55;
    margin-top: 6px;
    flex-grow: 1;
}

/* ── generic button reset ───────────────────────────────────────────────── */
.stButton > button {
    width: 100% !important;
    background-color: #2C2A36 !important;
    color: #E6E1E5 !important;
    border: 1px solid #49454F !important;
    border-radius: 10px !important;
    font-size: 0.82rem !important;
    padding: 8px 14px !important;
    transition: background 0.18s, border-color 0.18s !important;
    margin-top: 6px !important;
}
.stButton > button:hover {
    background-color: #3A3846 !important;
    border-color: #79747E !important;
}
.stButton > button:active {
    background-color: #46435A !important;
}

/* ── stop button ─────────────────────────────────────────────────────────── */
.stop-wrap .stButton > button {
    background-color: #2D1218 !important;
    border-color: #7A2040 !important;
    color: #FFB3C1 !important;
    font-weight: 600 !important;
    font-size: 0.90rem !important;
    border-radius: 12px !important;
    padding: 10px 18px !important;
    margin-top: 10px !important;
}
.stop-wrap .stButton > button:hover {
    background-color: #42192A !important;
    border-color: #C03060 !important;
}

/* ── back / retry buttons (results page) ────────────────────────────────── */
.back-wrap .stButton > button {
    background-color: #1A2840 !important;
    border-color: #365EA3 !important;
    color: #B3C8F5 !important;
    border-radius: 12px !important;
    padding: 10px 18px !important;
    font-size: 0.88rem !important;
}
.retry-wrap .stButton > button {
    background-color: #1C2A1E !important;
    border-color: #3A6B42 !important;
    color: #AEDDB6 !important;
    border-radius: 12px !important;
    padding: 10px 18px !important;
    font-size: 0.88rem !important;
}

/* ── results card ────────────────────────────────────────────────────────── */
.result-card {
    background: #1C1B22;
    border-radius: 20px;
    border: 1px solid #3A3845;
    padding: 36px 40px 28px;
    text-align: center;
    max-width: 440px;
    margin: 0 auto 24px;
}
.result-count {
    font-size: 5.5rem;
    font-weight: 800;
    color: #E6E1E5;
    line-height: 1;
    margin: 8px 0 4px;
}
.result-unit {
    font-size: 0.95rem;
    color: #938F99;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    margin-bottom: 18px;
}
.result-meta {
    display: flex;
    justify-content: center;
    gap: 36px;
    font-size: 0.80rem;
    color: #79747E;
}
.result-meta strong { display: block; font-size: 1.1rem; color: #CAC4D0; }

/* ── divider ─────────────────────────────────────────────────────────────── */
hr { border-color: #2C2A36 !important; margin: 14px 0 !important; }

/* ── info box ────────────────────────────────────────────────────────────── */
.info-box {
    margin-top: 14px;
    padding: 12px 16px;
    background: #1C1B22;
    border-radius: 12px;
    border: 1px solid #2C2A36;
    font-size: 0.77rem;
    color: #79747E;
    text-align: center;
    line-height: 1.7;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
#  SESSION STATE
# ─────────────────────────────────────────────────────────────────────────────
def _init_state() -> None:
    defaults: dict = {
        "app_state":    "menu",   # "menu" | "counting" | "results"
        "jump_type":    None,
        "final_count":  0,
        "duration":     0.0,
        "session_seq":  0,        # 카운팅 세션마다 증가 → processor 재생성 강제
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


# ─────────────────────────────────────────────────────────────────────────────
#  VIDEO PROCESSOR  (worker thread에서 recv() 호출)
# ─────────────────────────────────────────────────────────────────────────────
class JumpRopeProcessor(VideoProcessorBase):
    """
    선택된 jump_type에 맞는 counter engine을 로드하고,
    av.VideoFrame을 받아 처리 후 overlay가 그려진 프레임을 반환한다.
    accepted_count / duration은 main thread에서 읽으므로 lock 보호.
    """

    def __init__(self, jump_type: JumpType) -> None:
        self._lock = threading.Lock()
        self.jump_type        = jump_type
        self.accepted_count   = 0
        self._frame_idx       = 0
        self._start_time      = time.monotonic()
        self._count_start: Optional[float] = None
        self._pulse_remaining = 0
        self._pulse_total     = 10
        self._engine          = None
        self._last_phase: Optional[str] = None

        # ── 엔진 모듈 동적 임포트 ────────────────────────────────────────────
        if jump_type == JumpType.BASIC:
            from basic_jump.counter_engine import (
                EngineConfig, PoseSignalExtractor, RealtimeCounterEngine,
                RealtimeStartGate, core_landmarks_visible,
            )
            self._phase_hook  = None           # COUNTING 진입 시 특별 호출 없음
            self._uses_signal = False

        elif jump_type == JumpType.ALTERNATING:
            from alternating_jump.counter_engine import (
                EngineConfig, PoseSignalExtractor, RealtimeCounterEngine,
                RealtimeStartGate, core_landmarks_visible,
            )
            self._phase_hook  = "begin_count_phase"
            self._uses_signal = False

        else:  # DOUBLE
            from double_jump.counter_engine import (
                EngineConfig, PoseSignalExtractor, RealtimeCounterEngine,
                RealtimeStartGate, core_landmarks_visible,
            )
            self._phase_hook  = "arm_for_counting"
            self._uses_signal = True            # signal.detected도 체크

        # 클래스 참조를 인스턴스에 보관 (recv()에서 동적 생성에 사용)
        self._RCE             = RealtimeCounterEngine
        self._CLV             = core_landmarks_visible

        self._config    = EngineConfig()
        self._extractor = PoseSignalExtractor(self._config)
        self._gate      = RealtimeStartGate(
            ready_hold_seconds=1.0,
            countdown_seconds=3.0,
            ready_dropout_seconds=0.35,
        )
        self._last_phase = self._gate.phase

        # MediaPipe drawing (recv()에서 매 프레임 사용)
        self._mp_draw   = mp.solutions.drawing_utils
        self._mp_styles = mp.solutions.drawing_styles
        self._mp_pose   = mp.solutions.pose

    # ── main thread에서 읽는 프로퍼티 ────────────────────────────────────────
    @property
    def count(self) -> int:
        with self._lock:
            return self.accepted_count

    @property
    def duration(self) -> float:
        ref = self._count_start if self._count_start else self._start_time
        return time.monotonic() - ref

    # ── 프레임 처리 (worker thread) ───────────────────────────────────────────
    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        ts  = time.monotonic() - self._start_time

        signal, result = self._extractor.process_bgr_frame(img, self._frame_idx, ts)

        if self._uses_signal:
            count_ready = signal.detected or self._CLV(result, 0.30, 0.80)
        else:
            count_ready = self._CLV(result, 0.30, 0.80)

        stream_state  = self._gate.update(count_ready, ts)
        phase_changed = stream_state.phase != self._last_phase

        # 엔진 생명주기 관리
        if stream_state.phase == "SEARCHING" and not count_ready:
            self._engine = None
        elif self._engine is None and count_ready:
            self._engine = self._RCE(self._config)

        if phase_changed:
            if stream_state.phase == "COUNTING":
                if self._engine and self._phase_hook:
                    getattr(self._engine, self._phase_hook)()
                with self._lock:
                    self.accepted_count = 0
                self._count_start = time.monotonic()
            self._last_phase = stream_state.phase

        if self._engine is not None:
            if stream_state.phase != "COUNTING":
                self._engine.warmup(signal)
            else:
                event = self._engine.step(signal)
                if event is not None:
                    with self._lock:
                        self.accepted_count = event.running_count
                    self._pulse_remaining = self._pulse_total

        # ── 오버레이 렌더링 ──────────────────────────────────────────────────
        display = img.copy()
        if result.pose_landmarks:
            self._mp_draw.draw_landmarks(
                display,
                result.pose_landmarks,
                self._mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self._mp_styles.get_default_pose_landmarks_style(),
            )

        elapsed    = self.duration
        pulse_prog = self._pulse_remaining / self._pulse_total if self._pulse_total > 0 else 0.0
        _draw_counting_overlay(
            display, stream_state, self.accepted_count, count_ready,
            3.0, pulse_prog, self.jump_type, elapsed,
        )

        if self._pulse_remaining > 0:
            self._pulse_remaining -= 1
        self._frame_idx += 1

        return av.VideoFrame.from_ndarray(display, format="bgr24")

    def __del__(self) -> None:
        try:
            self._extractor.close()
        except Exception:
            pass


# ─────────────────────────────────────────────────────────────────────────────
#  MENU PAGE
# ─────────────────────────────────────────────────────────────────────────────
def _render_menu() -> None:
    st.markdown('<p class="app-title">Jump Rope Detector</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="app-subtitle">뛰기 종목을 선택하세요 &nbsp;·&nbsp; Choose a jump type</p>',
        unsafe_allow_html=True,
    )

    cols = st.columns(3, gap="medium")
    for col, jt in zip(cols, JumpType):
        meta   = _JUMP_META[jt]
        acc    = _bgr_to_hex(meta["accent"])
        with col:
            st.markdown(f"""
<div class="jump-card" style="border-top: 3px solid {acc};">
  <div class="card-badge" style="background-color:{acc};">{meta['key']}</div>
  <div class="card-name-ko">{meta['name_ko']}</div>
  <div class="card-name-en" style="color:{acc};">{meta['name_en']}</div>
  <div class="card-desc">{meta['desc']}</div>
</div>""", unsafe_allow_html=True)

            if st.button("시작  ·  Start", key=f"sel_{jt.name}"):
                st.session_state.jump_type   = jt
                st.session_state.app_state   = "counting"
                st.session_state.session_seq += 1
                st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
#  COUNTING PAGE
# ─────────────────────────────────────────────────────────────────────────────
def _render_counting() -> None:
    jt   = st.session_state.jump_type
    meta = _JUMP_META[jt]
    acc  = _bgr_to_hex(meta["accent"])
    seq  = st.session_state.session_seq

    # 상단 종목 표시
    st.markdown(f"""
<div style="display:flex;align-items:center;gap:12px;margin-bottom:14px;">
  <div style="width:4px;height:30px;background:{acc};border-radius:2px;flex-shrink:0;"></div>
  <div>
    <span style="font-size:1.05rem;font-weight:700;color:#E6E1E5;">{meta['name_ko']}</span>
    <span style="font-size:0.80rem;color:{acc};margin-left:10px;">{meta['name_en']}</span>
  </div>
</div>""", unsafe_allow_html=True)

    # WebRTC 스트림 (processor가 프레임마다 overlay 그림)
    webrtc_ctx = webrtc_streamer(
        key=f"jump-{jt.name}-{seq}",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIG,
        video_processor_factory=lambda: JumpRopeProcessor(jt),
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

    # Stop 버튼
    st.markdown("<div class='stop-wrap'>", unsafe_allow_html=True)
    stop = st.button("Stop  /  결과 보기", key="stop_btn")
    st.markdown("</div>", unsafe_allow_html=True)

    if stop:
        if webrtc_ctx.video_processor:
            st.session_state.final_count = webrtc_ctx.video_processor.count
            st.session_state.duration    = webrtc_ctx.video_processor.duration
        else:
            st.session_state.final_count = 0
            st.session_state.duration    = 0.0
        st.session_state.app_state = "results"
        st.rerun()

    # 사용 안내
    st.markdown("""
<div class="info-box">
  카메라를 허용한 뒤 화면 정면에 서세요 &nbsp;·&nbsp; Allow camera and stand in front of it<br>
  <span style="color:#49454F;">카운트·페이즈·타이머는 영상 위 오버레이에 표시됩니다
  &nbsp;·&nbsp; Count, phase and timer are shown on the video overlay</span>
</div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
#  RESULTS PAGE
# ─────────────────────────────────────────────────────────────────────────────
def _render_results() -> None:
    jt       = st.session_state.jump_type
    meta     = _JUMP_META[jt]
    acc      = _bgr_to_hex(meta["accent"])
    count    = st.session_state.final_count
    duration = st.session_state.duration
    mm, ss   = int(duration) // 60, int(duration) % 60

    st.markdown(f"""
<div class="result-card" style="border-top:4px solid {acc};">
  <div style="font-size:0.78rem;color:#938F99;text-transform:uppercase;letter-spacing:.1em;margin-bottom:6px;">
    Session Complete
  </div>
  <div style="font-size:0.88rem;font-weight:600;color:{acc};margin-bottom:8px;">
    {meta['name_ko']} &nbsp;·&nbsp; {meta['name_en']}
  </div>
  <div class="result-count">{count}</div>
  <div class="result-unit">jumps</div>
  <hr>
  <div class="result-meta">
    <span><strong>{mm:02d}:{ss:02d}</strong>Duration</span>
  </div>
</div>""", unsafe_allow_html=True)

    col_back, col_retry = st.columns(2, gap="small")

    with col_back:
        st.markdown("<div class='back-wrap'>", unsafe_allow_html=True)
        if st.button("← 메뉴로  ·  Back to Menu", key="back_btn"):
            st.session_state.app_state   = "menu"
            st.session_state.jump_type   = None
            st.session_state.final_count = 0
            st.session_state.duration    = 0.0
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

    with col_retry:
        st.markdown("<div class='retry-wrap'>", unsafe_allow_html=True)
        if st.button(f"↺  다시  ·  Retry", key="retry_btn"):
            st.session_state.app_state   = "counting"
            st.session_state.session_seq += 1
            st.session_state.final_count = 0
            st.session_state.duration    = 0.0
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN ROUTER
# ─────────────────────────────────────────────────────────────────────────────
_init_state()
_inject_css()

match st.session_state.app_state:
    case "menu":
        _render_menu()
    case "counting":
        _render_counting()
    case "results":
        _render_results()
