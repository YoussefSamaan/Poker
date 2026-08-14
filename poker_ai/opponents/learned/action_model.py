from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from ..dataset import OpponentFeatureVector, PublicDecisionExample
from .preprocessing import build_preprocessor, feature_matrix, transformed_feature_names
from .schema import (
    ACTION_CLASSES,
    HistoryAwareExample,
    OpponentHistoryFeatures,
    context_feature_mapping,
)


@dataclass(frozen=True, slots=True)
class Coefficient:
    action: str
    feature: str
    value: float


class _LogisticActionModel:
    MODEL_TYPE = "base"

    def __init__(self, *, seed: int = 0) -> None:
        self.seed = seed
        self.pipeline: Pipeline | None = None
        self.feature_names: tuple[str, ...] = ()

    def _fit(
        self, rows: Sequence[Mapping[str, object]], targets: Sequence[str]
    ) -> _LogisticActionModel:
        if len(rows) != len(targets) or len(rows) < 2:
            raise ValueError("training requires matching rows and at least two examples")
        if len(set(targets)) < 2:
            raise ValueError("training requires at least two observed action classes")
        self.feature_names = tuple(sorted(rows[0]))
        self.pipeline = Pipeline(
            (
                ("preprocess", build_preprocessor(self.feature_names)),
                (
                    "classifier",
                    LogisticRegression(
                        solver="lbfgs",
                        max_iter=1_000,
                        random_state=self.seed,
                    ),
                ),
            )
        )
        self.pipeline.fit(feature_matrix(rows, self.feature_names), targets)
        return self

    @property
    def fitted(self) -> bool:
        return self.pipeline is not None

    @property
    def classes(self) -> tuple[str, ...]:
        self._require_fitted()
        return tuple(map(str, self.pipeline.classes_))

    def _raw_probabilities(
        self, rows: Sequence[Mapping[str, object]]
    ) -> np.ndarray:
        self._require_fitted()
        return self.pipeline.predict_proba(feature_matrix(rows, self.feature_names))

    def _aligned_probabilities(
        self, rows: Sequence[Mapping[str, object]]
    ) -> np.ndarray:
        raw = self._raw_probabilities(rows)
        aligned = np.zeros((len(rows), len(ACTION_CLASSES)), dtype=float)
        for source, action in enumerate(self.classes):
            aligned[:, ACTION_CLASSES.index(action)] = raw[:, source]
        return aligned

    def inspect_coefficients(self, limit: int = 10) -> tuple[Coefficient, ...]:
        self._require_fitted()
        classifier = self.pipeline.named_steps["classifier"]
        names = transformed_feature_names(self.pipeline, self.feature_names)
        coefficients = classifier.coef_
        if len(self.classes) == 2 and coefficients.shape[0] == 1:
            coefficients = np.vstack((-coefficients[0], coefficients[0]))
        values = []
        for action, row in zip(self.classes, coefficients):
            ranked = sorted(
                zip(names, row), key=lambda item: abs(item[1]), reverse=True
            )[:limit]
            values.extend(Coefficient(action, name, float(value)) for name, value in ranked)
        return tuple(values)

    def _require_fitted(self) -> None:
        if self.pipeline is None:
            raise RuntimeError("learned action model is not fitted")


class ContextActionModel(_LogisticActionModel):
    MODEL_TYPE = "context_logistic"

    def fit(self, examples: Iterable[PublicDecisionExample]) -> ContextActionModel:
        values = tuple(examples)
        return self._fit(
            [context_feature_mapping(value.features) for value in values],
            [value.chosen_action_family for value in values],
        )

    def predict_probabilities(
        self, examples: Iterable[PublicDecisionExample]
    ) -> np.ndarray:
        values = tuple(examples)
        raw = self._aligned_probabilities(
            [context_feature_mapping(value.features) for value in values]
        )
        return _masked_matrix(raw, [value.features for value in values])

    def predict_distribution(
        self, features: OpponentFeatureVector
    ) -> dict[str, float]:
        row = context_feature_mapping(features)
        probabilities = self._aligned_probabilities([row])[0]
        return legal_action_mask(probabilities, features)


class HistoryAwareActionModel(_LogisticActionModel):
    MODEL_TYPE = "history_logistic"

    def fit(
        self, examples: Iterable[HistoryAwareExample]
    ) -> HistoryAwareActionModel:
        values = tuple(examples)
        return self._fit(
            [_history_mapping(value.public.features, value.history) for value in values],
            [value.public.chosen_action_family for value in values],
        )

    def predict_probabilities(
        self, examples: Iterable[HistoryAwareExample]
    ) -> np.ndarray:
        values = tuple(examples)
        raw = self._aligned_probabilities(
            [_history_mapping(value.public.features, value.history) for value in values]
        )
        return _masked_matrix(raw, [value.public.features for value in values])

    def predict_distribution(
        self,
        features: OpponentFeatureVector,
        history: OpponentHistoryFeatures,
    ) -> dict[str, float]:
        probabilities = self._aligned_probabilities(
            [_history_mapping(features, history)]
        )[0]
        return legal_action_mask(probabilities, features)


def legal_action_mask(
    probabilities: Sequence[float], features: OpponentFeatureVector
) -> dict[str, float]:
    legal = {
        "fold": features.can_fold,
        "check": features.can_check,
        "call": features.can_call,
        "bet": features.can_bet,
        "raise": features.can_raise,
    }
    masked = np.asarray(
        [probabilities[index] if legal[action] else 0.0 for index, action in enumerate(ACTION_CLASSES)],
        dtype=float,
    )
    total = float(masked.sum())
    if total <= 0:
        count = sum(bool(value) for value in legal.values())
        if not count:
            raise ValueError("decision has no legal action family")
        return {action: (1 / count if available else 0.0) for action, available in legal.items()}
    return {action: float(masked[index] / total) for index, action in enumerate(ACTION_CLASSES)}


def _history_mapping(
    context: OpponentFeatureVector, history: OpponentHistoryFeatures
) -> dict[str, object]:
    values = context_feature_mapping(context)
    values.update(history.as_dict())
    return values


def _masked_matrix(
    values: np.ndarray, features: Sequence[OpponentFeatureVector]
) -> np.ndarray:
    return np.asarray(
        [
            [legal_action_mask(row, feature)[action] for action in ACTION_CLASSES]
            for row, feature in zip(values, features)
        ]
    )
