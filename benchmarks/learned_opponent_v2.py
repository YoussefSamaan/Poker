"""Development-only Milestone 7 scientific comparison and runtime benchmark."""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import sys
import tempfile
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from poker_ai.equity import EquityCalculator
from poker_ai.cards import parse_cards
from poker_ai.opponents import ObserverContext, OpponentModel
from poker_ai.opponents.dataset import grouped_train_validation_test_split
from poker_ai.opponents.learned import (
    ContextActionModel,
    HandConditionedActionModel,
    LearnedRangeBelief,
    build_metadata,
    compare_action_models,
    compare_range_evaluations,
    context_learning_curve,
    equity_error_metrics,
    evaluate_domain_shift,
    evaluate_learned_range,
    evaluate_weighted_range,
    generate_balanced_synthetic_dataset,
    parameterized_ood_profiles,
    save_learned_artifact,
)
from poker_ai.ranges import WeightedRange


def run(hands_per_personality: int = 100, seed: int = 701) -> dict[str, object]:
    generation_start = time.perf_counter()
    bundle = generate_balanced_synthetic_dataset(
        hands_per_personality=hands_per_personality,
        sessions_per_personality=2,
        seed=seed,
    )
    generation_seconds = time.perf_counter() - generation_start
    action = compare_action_models(bundle, seed=seed, bootstrap_samples=200)

    split = grouped_train_validation_test_split(bundle.public_examples, seed=seed)
    learning_curve = context_learning_curve(
        split.train,
        split.validation or split.test,
        sizes=(50, 100, 250, 500, 1_000, 5_000),
        seed=seed,
    )
    train_keys = {_key(value) for value in split.train}
    held_keys = {_key(value) for value in (split.test or split.validation)}
    research_train = tuple(
        value for value in bundle.research.examples if _key(value.public) in train_keys
    )
    research_held = tuple(
        value for value in bundle.research.examples if _key(value.public) in held_keys
    )
    training_start = time.perf_counter()
    hand_model = HandConditionedActionModel(seed=seed).fit(research_train)
    hand_training_seconds = time.perf_counter() - training_start
    decisions = {
        (
            value.hand_key.session_id,
            value.hand_index,
            len(value.history),
            value.public_subject_id,
        ): value
        for result in bundle.results
        for record in result.records
        for value in record.observed_decisions
        if value.public_subject_id == "public_player_1"
    }
    archetype_evaluations = []
    learned_evaluations = []
    range_update_seconds = 0.0
    equity_learned, equity_archetype, equity_truth = [], [], []
    calculator = EquityCalculator()
    for example in research_held[:25]:
        decision = decisions[_key(example.public)]
        true_cards = tuple(parse_cards(example.true_hole_cards))
        learned = LearnedRangeBelief(hand_model)
        started = time.perf_counter()
        learned.update(decision, example.history)
        range_update_seconds += time.perf_counter() - started
        learned_evaluations.append(evaluate_learned_range(learned, true_cards))
        baseline = OpponentModel("benchmark", "public_player_1")
        inferred = baseline.infer_range_for_hand(
            (decision,),
            observer_context=ObserverContext("benchmark", decision.hand_key, ()),
        )
        archetype_evaluations.append(
            evaluate_weighted_range(inferred.weighted_range, true_cards)
        )

        record = next(
            record
            for result in bundle.results
            for record in result.records
            if record.hand_index == decision.hand_index
            and any(item.hand_key == decision.hand_key for item in record.observed_decisions)
        )
        hero_label = next(
            (
                label
                for label in record.research_labels
                if label.public_subject_id == "public_player_0"
            ),
            None,
        )
        if hero_label is not None and len(equity_truth) < 8:
            hero = hero_label.true_hole_cards
            true_range = WeightedRange.from_mapping(
                {"".join(map(str, true_cards)): 1.0}
            )
            kwargs = {"samples": 1_000, "seed": seed, "exact": False}
            equity_truth.append(
                calculator.calculate(hero, decision.board, true_range, **kwargs).equity
            )
            equity_learned.append(
                calculator.calculate(
                    hero, decision.board, learned.weighted_range(), **kwargs
                ).equity
            )
            equity_archetype.append(
                calculator.calculate(
                    hero, decision.board, inferred.weighted_range, **kwargs
                ).equity
            )
    ranges = compare_range_evaluations(archetype_evaluations, learned_evaluations)

    ood = generate_balanced_synthetic_dataset(
        hands_per_personality=max(20, hands_per_personality // 3),
        sessions_per_personality=1,
        seed=seed + 1,
        profiles=parameterized_ood_profiles(),
    )
    domain_shift = evaluate_domain_shift(bundle, ood, seed=seed)

    context = ContextActionModel(seed=seed)
    context_start = time.perf_counter()
    context.fit(split.train)
    context_training_seconds = time.perf_counter() - context_start
    prediction_start = time.perf_counter()
    context.predict_probabilities(split.test or split.validation)
    prediction_seconds = time.perf_counter() - prediction_start
    with tempfile.TemporaryDirectory() as directory:
        artifact = Path(directory) / "hand.joblib"
        metadata = build_metadata(
            hand_model,
            dataset_payload="milestone-7-benchmark",
            training_rows=len(research_train),
            training_correlation_groups=len(
                {value.public.correlation_group_id for value in research_train}
            ),
            metrics_summary={},
            seed=seed,
        )
        save_learned_artifact(artifact, hand_model, metadata)
        artifact_bytes = artifact.stat().st_size

    equity = {
        "learned": asdict(equity_error_metrics(equity_learned, equity_truth))
        if equity_truth
        else None,
        "archetype": asdict(equity_error_metrics(equity_archetype, equity_truth))
        if equity_truth
        else None,
    }
    return {
        "configuration": {
            "hands_per_personality": hands_per_personality,
            "sessions_per_personality": 2,
            "seed": seed,
            "public_rows": len(bundle.public_examples),
        },
        "action_prediction": asdict(action),
        "context_learning_curve": [asdict(value) for value in learning_curve],
        "range_inference": asdict(ranges),
        "downstream_equity_error": equity,
        "ood_parameterized_profiles": asdict(domain_shift),
        "runtime": {
            "dataset_generation_seconds": generation_seconds,
            "context_training_seconds": context_training_seconds,
            "hand_conditioned_training_seconds": hand_training_seconds,
            "context_prediction_decisions_per_second": len(split.test or split.validation)
            / prediction_seconds,
            "mean_range_update_ms": 1_000 * range_update_seconds / len(learned_evaluations),
            "hand_conditioned_artifact_bytes": artifact_bytes,
        },
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
