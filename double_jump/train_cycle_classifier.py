from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from double_jump.counter_engine import (  # noqa: E402
    EngineConfig,
    SignalFrame,
    _AirborneStateMachine,
    extract_signal_stream,
    load_ground_truth,
)
from double_jump.cycle_classifier import CLASS_NAMES  # noqa: E402
from double_jump.cycle_features import FEATURE_NAMES, build_cycle_feature_tensor, flatten_cycle_feature_tensor  # noqa: E402


@dataclass
class LabeledCycle:
    stem: str
    start_frame: int
    end_frame: int
    label: str
    feature_vector: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a cycle-based double-under classifier.")
    parser.add_argument("--video-dir", default="videos/double_jump_video")
    parser.add_argument("--label-dir", default="videos/double_jump_video")
    parser.add_argument("--target-frames", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--learning-rate", type=float, default=0.08)
    parser.add_argument("--l2", type=float, default=1e-4)
    parser.add_argument("--match-tolerance-frames", type=int, default=2)
    parser.add_argument("--output", default="double_jump/artifacts/cycle_classifier.json")
    return parser.parse_args()


def _collect_cycles(signals: list[SignalFrame], config: EngineConfig) -> list[tuple[int, int, list[SignalFrame]]]:
    machine = _AirborneStateMachine(config)
    active_frames: list[SignalFrame] = []
    cycles: list[tuple[int, int, list[SignalFrame]]] = []
    for signal in signals:
        was_in_air = machine.in_air
        machine.advance(signal)
        is_in_air = machine.in_air
        if not was_in_air and is_in_air:
            active_frames = [signal]
            continue
        if is_in_air and active_frames:
            active_frames.append(signal)
            continue
        if was_in_air and not is_in_air and active_frames:
            active_frames.append(signal)
            cycles.append((active_frames[0].frame_idx, signal.frame_idx, active_frames[:]))
            active_frames = []
    return cycles


def _label_cycle(start_frame: int, end_frame: int, gt_frames: list[int], tolerance: int) -> str:
    for gt_frame in gt_frames:
        if (start_frame - tolerance) <= gt_frame <= (end_frame + tolerance):
            return "double_under"
    return "basic_jump"


def _build_dataset(
    video_dir: Path,
    ground_truth: dict[str, list[object]],
    config: EngineConfig,
    target_frames: int,
    tolerance: int,
) -> list[LabeledCycle]:
    dataset: list[LabeledCycle] = []
    for stem in sorted(ground_truth):
        _, signals = extract_signal_stream(video_dir / f"{stem}.mp4", config)
        gt_frames = [event.frame_idx for event in ground_truth[stem]]
        for start_frame, end_frame, cycle_frames in _collect_cycles(signals, config):
            tensor = build_cycle_feature_tensor(cycle_frames, target_frames=target_frames)
            dataset.append(
                LabeledCycle(
                    stem=stem,
                    start_frame=start_frame,
                    end_frame=end_frame,
                    label=_label_cycle(start_frame, end_frame, gt_frames, tolerance),
                    feature_vector=flatten_cycle_feature_tensor(tensor),
                )
            )
    return dataset


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=1, keepdims=True)


def _train_softmax_classifier(
    X: np.ndarray,
    y: np.ndarray,
    epochs: int,
    learning_rate: float,
    l2: float,
) -> tuple[np.ndarray, np.ndarray]:
    weights = np.zeros((len(CLASS_NAMES), X.shape[1]), dtype=np.float32)
    bias = np.zeros((len(CLASS_NAMES),), dtype=np.float32)
    one_hot = np.eye(len(CLASS_NAMES), dtype=np.float32)[y]
    for _ in range(epochs):
        logits = X @ weights.T + bias
        probs = _softmax(logits)
        error = probs - one_hot
        grad_w = (error.T @ X) / X.shape[0]
        grad_b = np.mean(error, axis=0)
        grad_w += l2 * weights
        weights -= learning_rate * grad_w.astype(np.float32)
        bias -= learning_rate * grad_b.astype(np.float32)
    return weights, bias


def main() -> None:
    args = parse_args()
    config = EngineConfig()
    video_dir = Path(args.video_dir)
    ground_truth = load_ground_truth(args.label_dir, args.video_dir)
    dataset = _build_dataset(
        video_dir,
        ground_truth,
        config,
        target_frames=args.target_frames,
        tolerance=args.match_tolerance_frames,
    )
    if not dataset:
        raise RuntimeError("No jump cycles found for training.")

    label_to_idx = {name: idx for idx, name in enumerate(CLASS_NAMES)}
    X = np.stack([item.feature_vector for item in dataset], axis=0)
    y = np.asarray([label_to_idx.get(item.label, label_to_idx["basic_jump"]) for item in dataset], dtype=np.int64)
    weights, bias = _train_softmax_classifier(
        X,
        y,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        l2=args.l2,
    )
    logits = X @ weights.T + bias
    pred = np.argmax(_softmax(logits), axis=1)
    accuracy = float(np.mean(pred == y))

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "class_names": list(CLASS_NAMES),
        "target_frames": args.target_frames,
        "feature_names": FEATURE_NAMES,
        "weights": weights.tolist(),
        "bias": bias.tolist(),
        "training_accuracy": accuracy,
        "sample_count": int(X.shape[0]),
        "config": asdict(config),
    }
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[saved] {output_path}")
    print(f"[train] cycles={X.shape[0]} accuracy={accuracy:.4f}")


if __name__ == "__main__":
    main()
