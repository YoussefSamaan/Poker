from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path

import numpy as np

from ..dataset import grouped_train_validation_test_split
from .action_model import (
    BoostedContextActionModel,
    BoostedHistoryActionModel,
    ContextActionModel,
    HistoryAwareActionModel,
)
from .evaluation import evaluate_action_predictions
from .generation import generate_balanced_synthetic_dataset
from .hand_conditioned import HandConditionedActionModel
from .history_features import causal_history_examples
from .persistence import build_metadata, load_trusted_local_artifact, save_learned_artifact


def train_opponent_model(
    *,
    hands: int,
    sessions_per_profile: int,
    seed: int,
    output: str | Path,
    model_type: str = "history",
) -> dict[str, object]:
    bundle = generate_balanced_synthetic_dataset(
        hands_per_personality=hands,
        sessions_per_personality=sessions_per_profile,
        seed=seed,
    )
    split = grouped_train_validation_test_split(bundle.public_examples, seed=seed)
    if model_type in {"context", "boosted-context"}:
        model_class = (
            ContextActionModel if model_type == "context" else BoostedContextActionModel
        )
        model = model_class(seed=seed).fit(split.train)
        held = split.validation or split.test
        probabilities = model.predict_probabilities(held)
        evaluation_examples = held
    elif model_type in {"history", "boosted-history"}:
        histories = causal_history_examples(bundle.results)
        target = tuple(
            value
            for value in histories
            if value.public.public_subject_id == "public_player_1"
        )
        train_keys = {_example_key(value) for value in split.train}
        held_keys = {_example_key(value) for value in (split.validation or split.test)}
        training = tuple(value for value in target if _example_key(value.public) in train_keys)
        held = tuple(value for value in target if _example_key(value.public) in held_keys)
        model_class = (
            HistoryAwareActionModel
            if model_type == "history"
            else BoostedHistoryActionModel
        )
        model = model_class(seed=seed).fit(training)
        probabilities = model.predict_probabilities(held)
        evaluation_examples = tuple(value.public for value in held)
    elif model_type == "hand":
        train_keys = {_example_key(value) for value in split.train}
        held_keys = {_example_key(value) for value in (split.validation or split.test)}
        training = tuple(
            value
            for value in bundle.research.examples
            if _example_key(value.public) in train_keys
        )
        held = tuple(
            value
            for value in bundle.research.examples
            if _example_key(value.public) in held_keys
        )
        model = HandConditionedActionModel(seed=seed).fit(training)
        probabilities = np.vstack(
            [
                model.predict_candidate_probabilities(
                    value.public, value.history, [value.candidate]
                )[0]
                for value in held
            ]
        )
        evaluation_examples = tuple(value.public for value in held)
    else:
        raise ValueError(
            "model_type must be context, history, boosted-context, boosted-history, or hand"
        )
    metrics = evaluate_action_predictions(evaluation_examples, probabilities)
    dataset_payload = json.dumps(
        [asdict(value) for value in split.train], sort_keys=True
    )
    metadata = build_metadata(
        model,
        dataset_payload=dataset_payload,
        training_rows=len(split.train),
        training_correlation_groups=len(
            {value.correlation_group_id for value in split.train}
        ),
        metrics_summary={
            "validation_log_loss": metrics.log_loss,
            "validation_macro_f1": metrics.macro_f1,
            "validation_brier": metrics.multiclass_brier,
            "validation_ece": metrics.expected_calibration_error,
        },
        seed=seed,
    )
    save_learned_artifact(output, model, metadata)
    return {"artifact": str(output), "metadata": asdict(metadata), "metrics": asdict(metrics)}


def evaluate_opponent_model(
    *, path: str | Path, hands: int, sessions_per_profile: int, seed: int
) -> dict[str, object]:
    model, metadata = load_trusted_local_artifact(path)
    bundle = generate_balanced_synthetic_dataset(
        hands_per_personality=hands,
        sessions_per_personality=sessions_per_profile,
        seed=seed,
    )
    if model.MODEL_TYPE in {
        ContextActionModel.MODEL_TYPE,
        BoostedContextActionModel.MODEL_TYPE,
    }:
        examples = bundle.public_examples
        probabilities = model.predict_probabilities(examples)
    elif model.MODEL_TYPE in {
        HistoryAwareActionModel.MODEL_TYPE,
        BoostedHistoryActionModel.MODEL_TYPE,
    }:
        histories = causal_history_examples(bundle.results)
        rows = tuple(
            value
            for value in histories
            if value.public.public_subject_id == "public_player_1"
        )
        examples = tuple(value.public for value in rows)
        probabilities = model.predict_probabilities(rows)
    else:
        rows = bundle.research.examples
        examples = tuple(value.public for value in rows)
        probabilities = np.vstack(
            [
                model.predict_candidate_probabilities(
                    value.public, value.history, [value.candidate]
                )[0]
                for value in rows
            ]
        )
    metrics = evaluate_action_predictions(examples, probabilities)
    return {"metadata": asdict(metadata), "metrics": asdict(metrics)}


def _example_key(value) -> tuple[object, ...]:
    return (
        value.dataset_session_id,
        value.hand_index,
        value.decision_sequence,
        value.public_subject_id,
    )
