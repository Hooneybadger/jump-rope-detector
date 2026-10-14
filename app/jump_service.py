from __future__ import annotations

import importlib
import time
from dataclasses import dataclass

import cv2
import numpy as np

from .rope_trip import RopeTripPolicy
from .signal_resampler import SignalResampler
from .pose_protocol import PoseSamplingClock, decode_pose, finite_number, plausible_pose


MODES = {
    "basic": "basic_jump.counter_engine",
    "alternating": "alternating_jump.counter_engine",
    "double": "double_jump.counter_engine",
}

# Browser inference/transport target.
ANALYSIS_FPS = {
    "basic": 60,
    "alternating": 60,
    "double": 60,
}

# Counter input rate per engine (None = every frame the browser sends). Measured offline on the
# 20 labelled clips with landmarks cached at native fps (tools/eval/evaluate_engines.py):
# the old 20 Hz input lost alternate-foot steps (67.9% -> 98.4% at native 60 fps, 90.5% at 30 Hz)
# and double-unders (85.1% -> 89.5% at 30 Hz); two-foot jumps were unaffected (90.2% vs 90.0%).
COUNTER_INPUT_HZ = {
    "basic": 30.0,
    "alternating": None,
    "double": 30.0,
}

# Engines whose 30 fps frame thresholds need the stream expanded to 30 Hz. Only the
# double-under engine was validated with this; the other engines count better on raw frames.
RESAMPLED_MODES = {"double"}


@dataclass(frozen=True)
class FrameResult:
    count: int
    phase: str
    ready: bool
    framed: bool
    ready_progress: float
    countdown: float
    elapsed: float
    landmarks: list[list[float]]
    processing_ms: float
    inference_ms: float = 0.0


class JumpCounterSession:
    """Small adapter around the three preserved counting engines."""

    def __init__(self, mode: str, countdown_seconds: float = 3.0, input_backend: str = "legacy") -> None:
        if mode not in MODES:
            raise ValueError("unsupported jump mode")
        module = importlib.import_module(MODES[mode])
        if input_backend not in {"legacy", "landmarks"}:
            raise ValueError("unsupported input backend")
        self.input_backend = input_backend
        self.client_timestamp_origin = None
        self.last_client_timestamp = None
        self.client_server_origin = None
        rate = COUNTER_INPUT_HZ[mode]
        self.pose_sampling_clock = None if rate is None else PoseSamplingClock(rate)
        self.trip_policy = RopeTripPolicy()
        self.mode = mode
        self.config = module.EngineConfig()
        self.extractor = (module.PoseSignalExtractor(self.config) if input_backend == "legacy"
                          else module.PoseSignalExtractor(self.config, initialize_pose=False))
        self.engine_type = module.RealtimeCounterEngine
        self.visible = module.core_landmarks_visible
        self.gate = module.RealtimeStartGate(
            ready_hold_seconds=1.0,
            countdown_seconds=countdown_seconds,
            ready_dropout_seconds=0.35,
        )
        self.engine = None
        self.resampler = SignalResampler() if mode in RESAMPLED_MODES else None
        self.phase_hook = {
            "basic": None,
            "alternating": "begin_count_phase",
            "double": "arm_for_counting",
        }[mode]
        self.count = 0
        self.frame_index = 0
        self.started_at = time.monotonic()
        self.count_started_at: float | None = None
        self.last_phase = self.gate.phase

    @property
    def elapsed(self) -> float:
        if self.count_started_at is None:
            return 0.0
        return max(0.0, time.monotonic() - self.count_started_at)

    def process(self, payload: bytes, max_frame_bytes: int) -> FrameResult:
        processing_started_at = time.perf_counter()
        if self.input_backend != "legacy":
            raise ValueError("expected browser landmarks")
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
        return self._advance(signal, pose_result, timestamp, processing_started_at)

    def process_landmarks(self, message: dict) -> FrameResult:
        processing_started_at = time.perf_counter()
        if self.input_backend != "landmarks":
            raise ValueError("expected legacy frame")
        timestamp_ms, inference_ms, pose_result = decode_pose(message)
        frame_height = message.get("frameHeight")  # height of the image the pose model saw (optional)
        if frame_height is not None:
            frame_height = finite_number(frame_height, 16, 4320)
        server_timestamp = time.monotonic() - self.started_at
        if self.last_client_timestamp is not None and timestamp_ms <= self.last_client_timestamp:
            raise ValueError("pose timestamps must increase")
        if self.client_timestamp_origin is None:
            self.client_timestamp_origin = timestamp_ms
            self.client_server_origin = max(0.0, server_timestamp - inference_ms / 1000)
        timestamp = self.client_server_origin + (timestamp_ms - self.client_timestamp_origin) / 1000
        if abs(timestamp + inference_ms / 1000 - server_timestamp) > 5.0:
            raise ValueError("pose timestamp drift too large")
        self.last_client_timestamp = timestamp_ms
        signal = None
        if self.pose_sampling_clock is None or self.pose_sampling_clock.accept(timestamp):
            signal, _ = self.extractor.process_pose_result(plausible_pose(pose_result), self.frame_index, timestamp)
        if frame_height is not None and self.engine is not None and hasattr(self.engine, "frame_height_px"):
            self.engine.frame_height_px = frame_height
        return self._advance(signal, pose_result, server_timestamp, processing_started_at, inference_ms)

    def _advance(self, signal, pose_result, timestamp, processing_started_at, inference_ms=0.0):
        ready = self.visible(pose_result, 0.30, 0.80)
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
                self.trip_policy.reset()
                self.count_started_at = time.monotonic()
            self.last_phase = stream_state.phase

        engine_frames = [] if signal is None else ([signal] if self.resampler is None else self.resampler.push(signal))
        if self.engine is not None:
            for engine_frame in engine_frames:
                if stream_state.phase != "COUNTING":
                    self.engine.warmup(engine_frame)
                    continue
                self.trip_policy.expire(engine_frame.time_sec)
                event = self.engine.step(engine_frame)
                if event is not None:
                    self.count = self.trip_policy.observe(event, engine_frame.time_sec)

        landmarks = []
        if pose_result.pose_landmarks:
            landmarks = [
                [round(point.x, 4), round(point.y, 4), round(point.visibility, 3)]
                for point in pose_result.pose_landmarks.landmark
            ]
        if signal is not None:
            self.frame_index += 1
        return FrameResult(
            count=self.count,
            phase=stream_state.phase,
            ready=ready,
            framed=ready,
            ready_progress=float(stream_state.ready_progress),
            countdown=float(stream_state.countdown_remaining_sec),
            elapsed=self.elapsed,
            landmarks=landmarks,
            processing_ms=(time.perf_counter() - processing_started_at) * 1000,
            inference_ms=inference_ms,
        )

    def close(self) -> None:
        self.extractor.close()

