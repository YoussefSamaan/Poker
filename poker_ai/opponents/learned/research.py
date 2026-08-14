from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from ..dataset import PublicDecisionExample, grouped_train_validation_test_split
from ..model import OpponentHandBelief, OpponentModel
from ..observation import ObservedDecision, ObserverContext
from .action_model import ContextActionModel, HistoryAwareActionModel
from .evaluation import (
    ActionMetrics,
    LegalFrequencyBaseline,
    MetricDifference,
    evaluate_action_predictions,
    grouped_log_loss_difference_bootstrap,
)
from .generation import SyntheticOpponentDatasetBundle
from .history_features import causal_history_examples
from .schema import ACTION_CLASSES


@dataclass(frozen=True, slots=True)
class ActionComparisonReport:
    held_out_rows: int
    frequency: ActionMetrics
    context: ActionMetrics
    history: ActionMetrics
    bayesian_archetype: ActionMetrics
    context_minus_frequency: MetricDifference
    history_minus_context: MetricDifference
    history_minus_bayesian: MetricDifference


@dataclass(frozen=True, slots=True)
class DomainShiftReport:
    training_rows: int
    held_out_rows: int
    context: ActionMetrics
    history: ActionMetrics


@dataclass(frozen=True, slots=True)
class LearningCurveRow:
    training_rows: int
    training_log_loss: float
    validation_log_loss: float


def compare_action_models(
    bundle: SyntheticOpponentDatasetBundle,
    *,
    seed: int = 0,
    bootstrap_samples: int = 500,
) -> ActionComparisonReport:
    split = grouped_train_validation_test_split(bundle.public_examples, seed=seed)
    held = split.test or split.validation
    frequency_model = LegalFrequencyBaseline().fit(split.train)
    context_model = ContextActionModel(seed=seed).fit(split.train)
    histories = tuple(
        value
        for value in causal_history_examples(bundle.results)
        if value.public.public_subject_id == "public_player_1"
    )
    train_keys = {_key(value) for value in split.train}
    held_keys = {_key(value) for value in held}
    history_training = tuple(
        value for value in histories if _key(value.public) in train_keys
    )
    history_held = tuple(
        value for value in histories if _key(value.public) in held_keys
    )
    history_by_key = {_key(value.public): value for value in history_held}
    ordered_history = tuple(history_by_key[_key(value)] for value in held)
    history_model = HistoryAwareActionModel(seed=seed).fit(history_training)

    frequency = frequency_model.predict_probabilities(held)
    context = context_model.predict_probabilities(held)
    history = history_model.predict_probabilities(ordered_history)
    bayesian = bayesian_archetype_probabilities(
        bundle, split.train, held
    )
    return ActionComparisonReport(
        len(held),
        evaluate_action_predictions(held, frequency),
        evaluate_action_predictions(held, context),
        evaluate_action_predictions(held, history),
        evaluate_action_predictions(held, bayesian),
        grouped_log_loss_difference_bootstrap(
            held, context, frequency, samples=bootstrap_samples, seed=seed
        ),
        grouped_log_loss_difference_bootstrap(
            held, history, context, samples=bootstrap_samples, seed=seed + 1
        ),
        grouped_log_loss_difference_bootstrap(
            held, history, bayesian, samples=bootstrap_samples, seed=seed + 2
        ),
    )


def evaluate_domain_shift(
    training_bundle: SyntheticOpponentDatasetBundle,
    held_out_bundle: SyntheticOpponentDatasetBundle,
    *,
    seed: int = 0,
) -> DomainShiftReport:
    context_model = ContextActionModel(seed=seed).fit(
        training_bundle.public_examples
    )
    training_history = tuple(
        value
        for value in causal_history_examples(training_bundle.results)
        if value.public.public_subject_id == "public_player_1"
    )
    held_history = tuple(
        value
        for value in causal_history_examples(held_out_bundle.results)
        if value.public.public_subject_id == "public_player_1"
    )
    history_model = HistoryAwareActionModel(seed=seed).fit(training_history)
    held = held_out_bundle.public_examples
    return DomainShiftReport(
        len(training_bundle.public_examples),
        len(held),
        evaluate_action_predictions(held, context_model.predict_probabilities(held)),
        evaluate_action_predictions(
            tuple(value.public for value in held_history),
            history_model.predict_probabilities(held_history),
        ),
    )


def context_learning_curve(
    training: Iterable[PublicDecisionExample],
    validation: Iterable[PublicDecisionExample],
    *,
    sizes: Iterable[int],
    seed: int = 0,
) -> tuple[LearningCurveRow, ...]:
    training = tuple(training)
    validation = tuple(validation)
    rows = []
    for size in sorted(set(sizes)):
        subset = training[: min(size, len(training))]
        if len(subset) < 2 or len({item.chosen_action_family for item in subset}) < 2:
            continue
        model = ContextActionModel(seed=seed).fit(subset)
        rows.append(
            LearningCurveRow(
                len(subset),
                evaluate_action_predictions(
                    subset, model.predict_probabilities(subset)
                ).log_loss,
                evaluate_action_predictions(
                    validation, model.predict_probabilities(validation)
                ).log_loss,
            )
        )
    return tuple(rows)


def bayesian_archetype_probabilities(
    bundle: SyntheticOpponentDatasetBundle,
    training: Iterable[PublicDecisionExample],
    held: Iterable[PublicDecisionExample],
) -> np.ndarray:
    training_keys = {_key(value) for value in training}
    held_values = tuple(held)
    held_keys = {_key(value) for value in held_values}
    decisions = _target_decisions(bundle)
    model = OpponentModel("evaluation_observer", "public_player_1")
    for decision in decisions:
        if _decision_key(decision) in training_keys:
            model.observe(
                decision,
                observer_context=ObserverContext(
                    "evaluation_observer", decision.hand_key, ()
                ),
            )
    beliefs: dict[object, OpponentHandBelief] = {}
    predictions = {}
    for decision in decisions:
        key = _decision_key(decision)
        if key not in held_keys:
            continue
        context = ObserverContext("evaluation_observer", decision.hand_key, ())
        belief = beliefs.setdefault(decision.hand_key, model.start_hand(context))
        legal = {
            "fold": decision.can_fold,
            "check": decision.can_check,
            "call": decision.can_call,
            "bet": decision.can_bet,
            "raise": decision.can_raise,
        }
        distribution = model.action_distribution(
            decision, observer_context=context, hand_belief=belief
        )
        row = np.asarray(
            [distribution.get(action, 0.0) if legal[action] else 0.0 for action in ACTION_CLASSES]
        )
        row /= row.sum()
        predictions[key] = row
        belief.observe(decision)
        model.observe(decision, observer_context=context)
    return np.asarray([predictions[_key(value)] for value in held_values])


def _target_decisions(
    bundle: SyntheticOpponentDatasetBundle,
) -> tuple[ObservedDecision, ...]:
    return tuple(
        decision
        for result in bundle.results
        for record in result.records
        for decision in record.observed_decisions
        if decision.public_subject_id == "public_player_1"
    )


def _key(value: PublicDecisionExample) -> tuple[object, ...]:
    return (
        value.dataset_session_id,
        value.hand_index,
        value.decision_sequence,
        value.public_subject_id,
    )


def _decision_key(value: ObservedDecision) -> tuple[object, ...]:
    return (
        value.hand_key.session_id,
        value.hand_index,
        len(value.history),
        value.public_subject_id,
    )
