from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from double_jump.cycle_features import FEATURE_NAMES, build_cycle_feature_tensor, flatten_cycle_feature_tensor


CLASS_NAMES = ("no_jump", "basic_jump", "double_under")


@dataclass(frozen=True)
class CyclePrediction:
    label: str
    confidence: float
    probabilities: dict[str, float]
    source: str


class CycleClassifier:
    def __init__(
        self,
        target_frames: int = 32,
        model_path: str | Path | None = None,
    ) -> None:
        self.target_frames = target_frames
        self.model_path = None if model_path in {None, ""} else Path(model_path)
        self.weights: np.ndarray | None = None
        self.bias: np.ndarray | None = None
        self.class_names = list(CLASS_NAMES)
        self.source = "heuristic"
        if self.model_path is not None and self.model_path.exists():
            payload = json.loads(self.model_path.read_text(encoding="utf-8"))
            self.target_frames = int(payload["target_frames"])
            self.class_names = [str(name) for name in payload["class_names"]]
            self.weights = np.asarray(payload["weights"], dtype=np.float32)
            self.bias = np.asarray(payload["bias"], dtype=np.float32)
            self.source = f"softmax:{self.model_path.name}"

    def _softmax(self, logits: np.ndarray) -> np.ndarray:
        shifted = logits - np.max(logits)
        exp = np.exp(shifted)
        return exp / np.sum(exp)

    def _heuristic_probabilities(self, tensor: np.ndarray) -> dict[str, float]:
        wrist_rot = tensor[:, FEATURE_NAMES.index("wrist_rotation_ratio")]
        wrist_sync = tensor[:, FEATURE_NAMES.index("wrist_sync_ratio")]
        ankle_flow = tensor[:, FEATURE_NAMES.index("ankle_flow_ratio")]
        left_forearm = tensor[:, FEATURE_NAMES.index("left_forearm_angle")]
        right_forearm = tensor[:, FEATURE_NAMES.index("right_forearm_angle")]
        left_vel = np.abs(np.diff(left_forearm, prepend=left_forearm[0]))
        right_vel = np.abs(np.diff(right_forearm, prepend=right_forearm[0]))
        rotation_energy = float(np.mean(wrist_rot))
        peak_rotation = float(np.max(wrist_rot))
        sync_score = float(np.max(wrist_sync))
        ankle_energy = float(np.mean(ankle_flow))
        forearm_velocity = float(np.mean(np.maximum(left_vel, right_vel)))

        double_score = (
            (2.8 * rotation_energy)
            + (2.0 * peak_rotation)
            + (1.7 * sync_score)
            + (1.5 * forearm_velocity)
            + (0.6 * ankle_energy)
            - 1.8
        )
        basic_score = (
            (0.9 * ankle_energy)
            + (0.5 * sync_score)
            + (0.5 * peak_rotation)
            - (0.2 * forearm_velocity)
            + 0.2
        )
        no_jump_score = 0.35 - (0.8 * ankle_energy) - (0.8 * rotation_energy)
        raw = np.asarray([no_jump_score, basic_score, double_score], dtype=np.float32)
        probs = self._softmax(raw)
        return {name: float(prob) for name, prob in zip(CLASS_NAMES, probs)}

    def predict(self, cycle_frames: Sequence[object]) -> CyclePrediction:
        tensor = build_cycle_feature_tensor(cycle_frames, target_frames=self.target_frames)
        if self.weights is not None and self.bias is not None:
            flat = flatten_cycle_feature_tensor(tensor)
            logits = (self.weights @ flat) + self.bias
            probs = self._softmax(logits)
            probabilities = {name: float(prob) for name, prob in zip(self.class_names, probs)}
            label = self.class_names[int(np.argmax(probs))]
            confidence = float(np.max(probs))
            return CyclePrediction(label=label, confidence=confidence, probabilities=probabilities, source=self.source)

        probabilities = self._heuristic_probabilities(tensor)
        label = max(probabilities, key=probabilities.get)
        return CyclePrediction(
            label=label,
            confidence=float(probabilities[label]),
            probabilities=probabilities,
            source=self.source,
        )
