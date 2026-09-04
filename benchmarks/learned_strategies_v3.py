"""Milestone 8 nonlinear learned-strategy decision gate."""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import statistics
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from poker_ai.opponents.dataset import grouped_train_validation_test_split
from poker_ai.opponents.learned import (
    BoostedContextActionModel,
    BoostedHistoryActionModel,
    ContextActionModel,
    HistoryAwareActionModel,
    LegalFrequencyBaseline,
    causal_history_examples,
    evaluate_action_predictions,
    generate_balanced_synthetic_dataset,
    parameterized_ood_profiles,
    temporal_subject_split,
)


def run(
    hands_per_personality: int = 200,
    seeds: tuple[int, ...] = (811, 812, 813),
) -> dict[str, object]:
    started = time.perf_counter()
    trials = []
    for seed in seeds:
        bundle = generate_balanced_synthetic_dataset(
            hands_per_personality=hands_per_personality,
            sessions_per_personality=2,
            seed=seed,
        )
        histories = tuple(
            value
            for value in causal_history_examples(bundle.results)
            if value.public.public_subject_id == "public_player_1"
        )
        grouped = grouped_train_validation_test_split(bundle.public_examples, seed=seed)
        temporal = temporal_subject_split(bundle.public_examples)
        trials.append(
            {
                "seed": seed,
                "physical_hands": hands_per_personality * 6,
                "public_rows": len(bundle.public_examples),
                "grouped": _evaluate_split(grouped, histories, seed),
                "temporal": _evaluate_split(temporal, histories, seed),
                "ood": _evaluate_ood(bundle.public_examples, histories, seed),
            }
        )
    model_names = tuple(trials[0]["grouped"])
    summary = {
        split_name: {
            model_name: {
                "mean_log_loss": statistics.fmean(
                    trial[split_name][model_name]["log_loss"] for trial in trials
                ),
                "mean_accuracy": statistics.fmean(
                    trial[split_name][model_name]["accuracy"] for trial in trials
                ),
            }
            for model_name in model_names
        }
        for split_name in ("grouped", "temporal", "ood")
    }
    return {
        "configuration": {
            "hands_per_personality": hands_per_personality,
            "seeds": seeds,
            "total_physical_hands": hands_per_personality * 6 * len(seeds),
            "probability_floor": 1e-6,
        },
        "trials": trials,
        "summary": summary,
        "runtime_seconds": time.perf_counter() - started,
    }


def _evaluate_split(split, histories, seed: int) -> dict[str, dict[str, object]]:
    held = split.test or split.validation
    train_keys = {_key(value) for value in split.train}
    held_keys = {_key(value) for value in held}
    history_train = tuple(
        value for value in histories if _key(value.public) in train_keys
    )
    history_by_key = {
        _key(value.public): value
        for value in histories
        if _key(value.public) in held_keys
    }
    history_held = tuple(history_by_key[_key(value)] for value in held)
    predictions = {
        "frequency": LegalFrequencyBaseline().fit(split.train).predict_probabilities(held),
        "context_logistic": ContextActionModel(seed=seed)
        .fit(split.train)
        .predict_probabilities(held),
        "history_logistic": HistoryAwareActionModel(seed=seed)
        .fit(history_train)
        .predict_probabilities(history_held),
        "context_boosted": BoostedContextActionModel(seed=seed)
        .fit(split.train)
        .predict_probabilities(held),
        "history_boosted": BoostedHistoryActionModel(seed=seed)
        .fit(history_train)
        .predict_probabilities(history_held),
    }
    return {
        name: _compact_metrics(evaluate_action_predictions(held, probabilities))
        for name, probabilities in predictions.items()
    }


def _evaluate_ood(training, histories, seed: int) -> dict[str, dict[str, object]]:
    ood = generate_balanced_synthetic_dataset(
        hands_per_personality=50,
        sessions_per_personality=1,
        seed=seed + 10_000,
        profiles=parameterized_ood_profiles(),
    )
    held = ood.public_examples
    held_history = tuple(
        value
        for value in causal_history_examples(ood.results)
        if value.public.public_subject_id == "public_player_1"
    )
    predictions = {
        "frequency": LegalFrequencyBaseline().fit(training).predict_probabilities(held),
        "context_logistic": ContextActionModel(seed=seed)
        .fit(training)
        .predict_probabilities(held),
        "history_logistic": HistoryAwareActionModel(seed=seed)
        .fit(histories)
        .predict_probabilities(held_history),
        "context_boosted": BoostedContextActionModel(seed=seed)
        .fit(training)
        .predict_probabilities(held),
        "history_boosted": BoostedHistoryActionModel(seed=seed)
        .fit(histories)
        .predict_probabilities(held_history),
    }
    return {
        name: _compact_metrics(evaluate_action_predictions(held, probabilities))
        for name, probabilities in predictions.items()
    }


def _compact_metrics(metrics) -> dict[str, object]:
    values = asdict(metrics)
    return {
        name: values[name]
        for name in (
            "rows",
            "log_loss",
            "accuracy",
            "macro_f1",
            "multiclass_brier",
            "expected_calibration_error",
        )
    }


def _key(value) -> tuple[object, ...]:
    return (
        value.dataset_session_id,
        value.hand_index,
        value.decision_sequence,
        value.public_subject_id,
    )


if __name__ == "__main__":
    print(json.dumps(run(), indent=2, sort_keys=True))
