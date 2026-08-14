from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import math
from typing import Iterable, Mapping, Sequence

import numpy as np

from ...agents.features import canonical_hand_class
from ...cards import Card
from ...evaluation import evaluate_holdem
from ...experiments.simulator import ExperimentResult
from ...ranges import WeightedCombo, WeightedRange
from ..dataset import PublicDecisionExample, PublicObservationDataset, public_decision_features
from ..model import RangeBelief
from ..observation import ObservedDecision
from .action_model import _LogisticActionModel, _history_mapping, _masked_matrix
from .history_features import causal_history_examples
from .schema import (
    ACTION_CLASSES,
    CandidateHandFeatures,
    OpponentHistoryFeatures,
    ResearchHandConditionedExample,
)

_STRAIGHT_WINDOWS = tuple(frozenset(range(start, start + 5)) for start in range(1, 11))


@dataclass(frozen=True, slots=True)
class ResearchHandConditionedDataset:
    """Privileged synthetic dataset; never accepted by public inference APIs."""

    schema_version: int
    examples: tuple[ResearchHandConditionedExample, ...]
    privileged_profile_composition: tuple[tuple[str, int], ...]


@dataclass(frozen=True, slots=True)
class RangeEvaluation:
    true_combo_negative_log_probability: float
    true_combo_percentile: float
    true_hand_class_probability: float
    entropy: float


@dataclass(frozen=True, slots=True)
class EquityErrorMetrics:
    mean_absolute_error: float
    root_mean_squared_error: float
    spots: int


@dataclass(frozen=True, slots=True)
class RangeComparisonReport:
    spots: int
    archetype_mean_true_combo_nll: float
    learned_mean_true_combo_nll: float
    archetype_mean_true_class_probability: float
    learned_mean_true_class_probability: float
    archetype_mean_entropy: float
    learned_mean_entropy: float


def candidate_hand_features(
    cards: tuple[Card, Card], board: tuple[Card, ...]
) -> CandidateHandFeatures:
    high, low = sorted((card.rank_value for card in cards), reverse=True)
    all_cards = cards + board
    category = evaluate_holdem(all_cards).name if len(all_cards) >= 5 else "preflop"
    suits = Counter(card.suit for card in all_cards)
    flush_draw = int(len(all_cards) < 7 and max(suits.values(), default=0) == 4)
    completions = _straight_completion_ranks(all_cards)
    board_ranks = {card.rank for card in board}
    return CandidateHandFeatures(
        high,
        low,
        int(high == low),
        int(cards[0].suit == cards[1].suit),
        max(0, high - low - 1),
        canonical_hand_class(cards),
        category,
        flush_draw,
        int(len(completions) >= 2),
        int(len(completions) == 1),
        sum(card.rank in board_ranks for card in cards),
    )


def build_research_hand_conditioned_dataset(
    results: Iterable[ExperimentResult],
) -> ResearchHandConditionedDataset:
    results = tuple(results)
    histories = causal_history_examples(results)
    history_map = {
        (
            value.public.dataset_session_id,
            value.public.hand_index,
            value.public.decision_sequence,
            value.public.public_subject_id,
        ): value.history
        for value in histories
    }
    rows = []
    composition: Counter[str] = Counter()
    for result in results:
        public = PublicObservationDataset.from_experiment(result)
        labels = tuple(
            label for record in result.records for label in record.research_labels
        )
        if len(public.examples) != len(labels):
            raise AssertionError("public decisions and research labels lost alignment")
        for example, label in zip(public.examples, labels):
            if example.public_subject_id != label.public_subject_id:
                raise AssertionError("research/public subject mapping mismatch")
            history = history_map[
                (
                    example.dataset_session_id,
                    example.hand_index,
                    example.decision_sequence,
                    example.public_subject_id,
                )
            ]
            cards = tuple(label.true_hole_cards)
            rows.append(
                ResearchHandConditionedExample(
                    example,
                    history,
                    candidate_hand_features(cards, _board_for_example(result, example)),
                    tuple(map(str, cards)),
                )
            )
            composition[label.true_profile_name] += 1
    return ResearchHandConditionedDataset(1, tuple(rows), tuple(sorted(composition.items())))


class HandConditionedActionModel(_LogisticActionModel):
    MODEL_TYPE = "hand_conditioned_logistic"

    def fit(
        self, examples: Iterable[ResearchHandConditionedExample]
    ) -> HandConditionedActionModel:
        values = tuple(examples)
        return self._fit(
            [_conditioned_mapping(value.public, value.history, value.candidate) for value in values],
            [value.public.chosen_action_family for value in values],
        )

    def predict_candidate_probabilities(
        self,
        public: PublicDecisionExample,
        history: OpponentHistoryFeatures,
        candidates: Sequence[CandidateHandFeatures],
    ) -> np.ndarray:
        base = _history_mapping(public.features, history)
        rows = []
        for candidate in candidates:
            row = dict(base)
            row.update(_candidate_mapping(candidate))
            rows.append(row)
        raw = self._aligned_probabilities(rows)
        return _masked_matrix(raw, [public.features] * len(candidates))


class LearnedRangeBelief:
    """Blocker-aware range updated by one batched learned likelihood call/action."""

    def __init__(
        self,
        model: HandConditionedActionModel,
        known_cards: Iterable[Card] = (),
        weights: Mapping[tuple[Card, Card], float] | None = None,
    ) -> None:
        self.model = model
        self.known_cards = tuple(known_cards)
        self.range = RangeBelief(weights, self.known_cards)
        self.conditioned_actions: list[str] = []

    def update(
        self,
        decision: ObservedDecision,
        history: OpponentHistoryFeatures,
    ) -> float:
        blockers = (*self.known_cards, *decision.board)
        legal = {
            cards: weight
            for cards, weight in self.range.weights.items()
            if not set(cards).intersection(blockers)
        }
        total = sum(legal.values())
        legal = {cards: weight / total for cards, weight in legal.items()}
        cards = tuple(legal)
        public = _public_example_from_decision(decision)
        candidates = [candidate_hand_features(combo, decision.board) for combo in cards]
        probabilities = self.model.predict_candidate_probabilities(public, history, candidates)
        action_index = ACTION_CLASSES.index(decision.action_family)
        likelihoods = probabilities[:, action_index]
        evidence = float(sum(legal[combo] * value for combo, value in zip(cards, likelihoods)))
        posterior = {
            combo: legal[combo] * max(float(value), 1e-12)
            for combo, value in zip(cards, likelihoods)
        }
        posterior_total = sum(posterior.values())
        self.range.weights = {
            combo: value / posterior_total for combo, value in posterior.items()
        }
        self.conditioned_actions.append(decision.action_family)
        return evidence

    def weighted_range(self) -> WeightedRange:
        return WeightedRange(
            WeightedCombo(cards, weight)
            for cards, weight in self.range.weights.items()
        )


def evaluate_learned_range(
    belief: LearnedRangeBelief, true_cards: tuple[Card, Card]
) -> RangeEvaluation:
    return evaluate_weighted_range(belief.weighted_range(), true_cards)


def evaluate_weighted_range(
    weighted_range: WeightedRange, true_cards: tuple[Card, Card]
) -> RangeEvaluation:
    canonical = tuple(sorted(true_cards, key=str))
    weights = {
        tuple(sorted(cards, key=str)): value
        for cards, value in (
            (combo.cards, combo.weight) for combo in weighted_range.combos
        )
    }
    probability = max(weights.get(canonical, 0.0), 1e-15)
    ordered = sorted(weights.values(), reverse=True)
    rank = 1 + sum(value > probability for value in ordered)
    true_class = canonical_hand_class(true_cards)
    class_probability = sum(
        value
        for cards, value in weights.items()
        if canonical_hand_class(cards) == true_class
    )
    entropy = -sum(value * math.log(value) for value in weights.values() if value > 0)
    return RangeEvaluation(
        -math.log(probability),
        1 - (rank - 1) / max(1, len(ordered) - 1),
        class_probability,
        entropy,
    )


def equity_error_metrics(
    estimated: Sequence[float], truth: Sequence[float]
) -> EquityErrorMetrics:
    if len(estimated) != len(truth) or not estimated:
        raise ValueError("equity arrays must be non-empty and aligned")
    errors = np.asarray(estimated) - np.asarray(truth)
    return EquityErrorMetrics(
        float(np.abs(errors).mean()),
        float(np.sqrt(np.square(errors).mean())),
        len(errors),
    )


def compare_range_evaluations(
    archetype: Sequence[RangeEvaluation],
    learned: Sequence[RangeEvaluation],
) -> RangeComparisonReport:
    if len(archetype) != len(learned) or not archetype:
        raise ValueError("range evaluations must be non-empty and aligned")
    return RangeComparisonReport(
        len(archetype),
        float(np.mean([value.true_combo_negative_log_probability for value in archetype])),
        float(np.mean([value.true_combo_negative_log_probability for value in learned])),
        float(np.mean([value.true_hand_class_probability for value in archetype])),
        float(np.mean([value.true_hand_class_probability for value in learned])),
        float(np.mean([value.entropy for value in archetype])),
        float(np.mean([value.entropy for value in learned])),
    )


def _conditioned_mapping(
    public: PublicDecisionExample,
    history: OpponentHistoryFeatures,
    candidate: CandidateHandFeatures,
) -> dict[str, object]:
    values = _history_mapping(public.features, history)
    values.update(_candidate_mapping(candidate))
    return values


def _candidate_mapping(candidate: CandidateHandFeatures) -> dict[str, object]:
    return {
        f"candidate_{name}": getattr(candidate, name)
        for name in candidate.__dataclass_fields__
    }


def _public_example_from_decision(decision: ObservedDecision) -> PublicDecisionExample:
    return PublicDecisionExample(
        decision.hand_key.session_id,
        decision.hand_index,
        len(decision.history),
        decision.public_subject_id,
        f"{decision.hand_key.session_id}:hand:{decision.hand_index}",
        public_decision_features(decision),
        decision.action_family,
    )


def _board_for_example(
    result: ExperimentResult, example: PublicDecisionExample
) -> tuple[Card, ...]:
    record = result.records[example.hand_index]
    decision = next(
        item
        for item in record.observed_decisions
        if len(item.history) == example.decision_sequence
        and item.public_subject_id == example.public_subject_id
    )
    return decision.board


def _straight_completion_ranks(cards: tuple[Card, ...]) -> frozenset[int]:
    values = {card.rank_value for card in cards}
    normalized = values | ({1} if 14 in values else set())
    if any(window <= normalized for window in _STRAIGHT_WINDOWS):
        return frozenset()
    completions = set()
    for window in _STRAIGHT_WINDOWS:
        missing = window - normalized
        if len(missing) == 1:
            rank = next(iter(missing))
            completions.add(14 if rank == 1 else rank)
    return frozenset(completions)
