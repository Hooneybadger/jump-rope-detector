from __future__ import annotations

import importlib
import time
from dataclasses import dataclass

import cv2
import numpy as np


MODES = {
    "basic": "basic_jump.counter_engine",
    "alternating": "alternating_jump.counter_engine",
    "double": "double_jump.counter_engine",
}


@dataclass(frozen=True)
class FrameResult:
    count: int
    phase: str
    ready: bool
    ready_progress: float
    countdown: float
    elapsed: float
    landmarks: list[list[float]]
    processing_ms: float


class JumpCounterSession:
    """Small adapter around the three preserved counting engines."""

    def __init__(self, mode: str) -> None:
        if mode not in MODES:
            raise ValueError("unsupported jump mode")
        module = importlib.import_module(MODES[mode])
        self.mode = mode
        self.config = module.EngineConfig()
        self.extractor = module.PoseSignalExtractor(self.config)
        self.engine_type = module.RealtimeCounterEngine
        self.visible = module.core_landmarks_visible
        self.gate = module.RealtimeStartGate(
            ready_hold_seconds=1.0,
            countdown_seconds=3.0,
            ready_dropout_seconds=0.35,
        )
        self.engine = None
        self.phase_hook = {
            "basic": None,
            "alternating": "begin_count_phase",
            "double": "arm_for_counting",
        }[mode]
        self.uses_signal_detection = mode == "double"
        self.count = 0
        self.frame_index = 0
        self.started_at = time.monotonic()
        self.count_started_at: float | None = None
        self.last_phase = self.gate.phase

    @property
    def elapsed(self) -> float:
        started = self.count_started_at or self.started_at
        return max(0.0, time.monotonic() - started)

    def process(self, payload: bytes, max_frame_bytes: int) -> FrameResult:
        processing_started_at = time.perf_counter()
        if not payload or len(payload) > max_frame_bytes:
            raise ValueError("frame size is not allowed")
        data = np.frombuffer(payload, dtype=np.uint8)
        frame = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError("invalid image frame")
        height, width = frame.shape[:2]
        if width < 160 or height < 120 or width > 1920 or height > 1080:
            raise ValueError("frame dimensions are not allowed")

        timestamp = time.monotonic() - self.started_at
        signal, pose_result = self.extractor.process_bgr_frame(frame, self.frame_index, timestamp)
        ready = self.visible(pose_result, 0.30, 0.80)
        if self.uses_signal_detection:
            ready = bool(signal.detected or ready)
        stream_state = self.gate.update(ready, timestamp)
        phase_changed = stream_state.phase != self.last_phase

        if stream_state.phase == "SEARCHING" and not ready:
            self.engine = None
        elif self.engine is None and ready:
            self.engine = self.engine_type(self.config, enable_realtime_compensation=True)

        if phase_changed:
            if stream_state.phase == "COUNTING":
                if self.engine is not None and self.phase_hook:
                    getattr(self.engine, self.phase_hook)()
                self.count = 0
                self.count_started_at = time.monotonic()
            self.last_phase = stream_state.phase

        if self.engine is not None:
            if stream_state.phase != "COUNTING":
                self.engine.warmup(signal)
            else:
                event = self.engine.step(signal)
                if event is not None:
                    self.count = int(event.running_count)

        landmarks = []
        if pose_result.pose_landmarks:
            landmarks = [
                [round(point.x, 4), round(point.y, 4), round(point.visibility, 3)]
                for point in pose_result.pose_landmarks.landmark
            ]
        self.frame_index += 1
        return FrameResult(
            count=self.count,
            phase=stream_state.phase,
            ready=ready,
            ready_progress=float(stream_state.ready_progress),
            countdown=float(stream_state.countdown_remaining_sec),
            elapsed=self.elapsed,
            landmarks=landmarks,
            processing_ms=(time.perf_counter() - processing_started_at) * 1000,
        )

    def close(self) -> None:
        self.extractor.close()

