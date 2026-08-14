from __future__ import annotations

from dataclasses import dataclass
import math
import random
from typing import Iterable, Sequence

import numpy as np
from sklearn.metrics import accuracy_score, f1_score

from ..dataset import DatasetSplit, PublicDecisionExample
from .action_model import legal_action_mask
from .schema import ACTION_CLASSES


@dataclass(frozen=True, slots=True)
class ReliabilityBin:
    lower: float
    upper: float
    count: int
    mean_confidence: float
    empirical_accuracy: float


@dataclass(frozen=True, slots=True)
class ActionMetrics:
    rows: int
    log_loss: float
    mean_negative_log_likelihood: float
    accuracy: float
    macro_f1: float
    multiclass_brier: float
    expected_calibration_error: float
    reliability: tuple[ReliabilityBin, ...]
    per_class_frequency: tuple[tuple[str, float, float], ...]


@dataclass(frozen=True, slots=True)
class MetricDifference:
    point_estimate: float
    confidence_interval_95: tuple[float, float]
    bootstrap_samples: int


class LegalFrequencyBaseline:
    """Training action frequency conditional only on the five-bit legal mask."""

    def __init__(self) -> None:
        self.counts: dict[tuple[int, ...], np.ndarray] = {}
        self.global_counts = np.ones(len(ACTION_CLASSES), dtype=float)

    def fit(self, examples: Iterable[PublicDecisionExample]) -> LegalFrequencyBaseline:
        for example in examples:
            mask = _legal_key(example)
            counts = self.counts.setdefault(mask, np.ones(len(ACTION_CLASSES)))
            index = ACTION_CLASSES.index(example.chosen_action_family)
            counts[index] += 1
            self.global_counts[index] += 1
        return self

    def predict_probabilities(
        self, examples: Iterable[PublicDecisionExample]
    ) -> np.ndarray:
        rows = []
        for example in examples:
            counts = self.counts.get(_legal_key(example), self.global_counts)
            rows.append(
                [
                    legal_action_mask(counts / counts.sum(), example.features)[action]
                    for action in ACTION_CLASSES
                ]
            )
        return np.asarray(rows)


def evaluate_action_predictions(
    examples: Sequence[PublicDecisionExample],
    probabilities: np.ndarray,
    *,
    calibration_bins: int = 10,
) -> ActionMetrics:
    if probabilities.shape != (len(examples), len(ACTION_CLASSES)):
        raise ValueError("probability matrix does not align with held-out examples")
    probabilities = np.asarray(probabilities, dtype=float)
    if np.any(probabilities < 0) or not np.allclose(probabilities.sum(axis=1), 1):
        raise ValueError("each probability row must be normalized")
    truth = np.asarray(
        [ACTION_CLASSES.index(example.chosen_action_family) for example in examples]
    )
    selected = np.clip(probabilities[np.arange(len(truth)), truth], 1e-15, 1)
    predictions = probabilities.argmax(axis=1)
    one_hot = np.eye(len(ACTION_CLASSES))[truth]
    log_loss = float(-np.log(selected).mean())
    reliability, ece = calibration_diagnostics(
        truth, probabilities, bins=calibration_bins
    )
    per_class = tuple(
        (
            action,
            float(probabilities[:, index].mean()),
            float((truth == index).mean()),
        )
        for index, action in enumerate(ACTION_CLASSES)
    )
    return ActionMetrics(
        len(examples),
        log_loss,
        log_loss,
        float(accuracy_score(truth, predictions)),
        float(f1_score(truth, predictions, average="macro", zero_division=0)),
        float(np.square(probabilities - one_hot).sum(axis=1).mean()),
        ece,
        reliability,
        per_class,
    )


def calibration_diagnostics(
    truth: np.ndarray, probabilities: np.ndarray, *, bins: int = 10
) -> tuple[tuple[ReliabilityBin, ...], float]:
    confidence = probabilities.max(axis=1)
    correct = probabilities.argmax(axis=1) == truth
    rows = []
    ece = 0.0
    for index in range(bins):
        lower, upper = index / bins, (index + 1) / bins
        selected = (confidence >= lower) & (
            confidence <= upper if index == bins - 1 else confidence < upper
        )
        count = int(selected.sum())
        if not count:
            continue
        mean_confidence = float(confidence[selected].mean())
        empirical = float(correct[selected].mean())
        rows.append(ReliabilityBin(lower, upper, count, mean_confidence, empirical))
        ece += count / len(truth) * abs(mean_confidence - empirical)
    return tuple(rows), ece


def metrics_by_slice(
    examples: Sequence[PublicDecisionExample], probabilities: np.ndarray
) -> dict[str, ActionMetrics]:
    slices: dict[str, list[int]] = {}
    for index, example in enumerate(examples):
        feature = example.features
        labels = (
            feature.street,
            f"position:{feature.position}",
            "facing_bet" if feature.to_call_bb > 0 else "checked_to",
            "heads_up" if feature.active_players == 2 else "multiway",
        )
        for label in labels:
            slices.setdefault(label, []).append(index)
    return {
        label: evaluate_action_predictions(
            [examples[index] for index in indices], probabilities[indices]
        )
        for label, indices in slices.items()
    }


def temporal_subject_split(
    examples: Iterable[PublicDecisionExample],
    *,
    validation_fraction: float = 0.15,
    test_fraction: float = 0.15,
) -> DatasetSplit:
    values = tuple(examples)
    sessions: dict[str, dict[str, list[PublicDecisionExample]]] = {}
    for example in values:
        sessions.setdefault(example.dataset_session_id, {}).setdefault(
            example.correlation_group_id, []
        ).append(example)
    train, validation, test = [], [], []
    for groups in sessions.values():
        ordered = sorted(
            groups,
            key=lambda key: min(item.hand_index for item in groups[key]),
        )
        validation_count = round(len(ordered) * validation_fraction)
        test_count = round(len(ordered) * test_fraction)
        train_count = len(ordered) - validation_count - test_count
        for index, key in enumerate(ordered):
            destination = (
                train
                if index < train_count
                else validation if index < train_count + validation_count else test
            )
            destination.extend(groups[key])
    return DatasetSplit(tuple(train), tuple(validation), tuple(test))


def grouped_log_loss_difference_bootstrap(
    examples: Sequence[PublicDecisionExample],
    first: np.ndarray,
    second: np.ndarray,
    *,
    samples: int = 1_000,
    seed: int = 0,
) -> MetricDifference:
    groups: dict[str, list[int]] = {}
    for index, example in enumerate(examples):
        groups.setdefault(example.correlation_group_id, []).append(index)
    keys = sorted(groups)
    losses_first = _row_log_losses(examples, first)
    losses_second = _row_log_losses(examples, second)
    differences = losses_first - losses_second
    rng = random.Random(seed)
    bootstrap = []
    for _ in range(samples):
        selected = [rng.choice(keys) for _ in keys]
        indices = [index for key in selected for index in groups[key]]
        bootstrap.append(float(differences[indices].mean()))
    ordered = sorted(bootstrap)
    return MetricDifference(
        float(differences.mean()),
        (
            ordered[math.floor(0.025 * (samples - 1))],
            ordered[math.ceil(0.975 * (samples - 1))],
        ),
        samples,
    )


def _row_log_losses(
    examples: Sequence[PublicDecisionExample], probabilities: np.ndarray
) -> np.ndarray:
    truth = [ACTION_CLASSES.index(item.chosen_action_family) for item in examples]
    return -np.log(np.clip(probabilities[np.arange(len(examples)), truth], 1e-15, 1))


def _legal_key(example: PublicDecisionExample) -> tuple[int, ...]:
    feature = example.features
    return (
        feature.can_fold,
        feature.can_check,
        feature.can_call,
        feature.can_bet,
        feature.can_raise,
    )
