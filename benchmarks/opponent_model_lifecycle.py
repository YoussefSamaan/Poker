"""Development-only Milestone 6.2 state and throughput benchmark."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import time
import tracemalloc

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from poker_ai.agents import PRESETS
from poker_ai.holdem import PlayerStatus, PublicPlayerState, Street
from poker_ai.opponents import (
    HandKey,
    ObservedDecision,
    ObserverContext,
    OpponentModel,
    PublicDecisionExample,
    PublicObservationDataset,
    public_decision_features,
)


def synthetic_decision(hand: int) -> ObservedDecision:
    players = (
        PublicPlayerState("hero", 0, 200, PlayerStatus.ACTIVE, 1, 1),
        PublicPlayerState("villain", 1, 198, PlayerStatus.ACTIVE, 2, 2),
    )
    return ObservedDecision(
        HandKey("benchmark", hand), "villain", "villain", Street.PREFLOP,
        "BB", (), "hero", "hero", "villain", players, (), 3, 2, 1,
        True, False, True, False, True, 0, None, 2, "call", None, None,
    )


def run(points: tuple[int, ...] = (100, 1_000, 10_000)) -> list[dict[str, float]]:
    model = OpponentModel(
        "hero", "villain", archetypes={"tag": PRESETS["tag"]}
    )
    rows = []
    completed = 0
    tracemalloc.start()
    started = time.perf_counter()
    for target in points:
        for hand in range(completed, target):
            model.commit_hand(
                (),
                observer_context=ObserverContext(
                    "hero", HandKey("benchmark", hand), ()
                ),
            )
            model.finish_hand()
        completed = target
        current, peak = tracemalloc.get_traced_memory()
        serialized = model.to_json()
        rows.append(
            {
                "hands": target,
                "elapsed_seconds": time.perf_counter() - started,
                "tracemalloc_current_bytes": current,
                "tracemalloc_peak_bytes": peak,
                "serialized_json_bytes": len(serialized.encode()),
            }
        )
    samples = tuple(synthetic_decision(index) for index in range(1_000))
    feature_start = time.perf_counter()
    for value in samples:
        public_decision_features(value)
    feature_seconds = time.perf_counter() - feature_start
    examples = tuple(
        PublicDecisionExample(
            "benchmark",
            value.hand_index,
            0,
            "public_player_1",
            f"benchmark:hand:{value.hand_index}",
            public_decision_features(value),
            value.action_family,
        )
        for value in samples
    )
    dataset_start = time.perf_counter()
    dataset_json = PublicObservationDataset(1, "benchmark", examples).to_json()
    dataset_seconds = time.perf_counter() - dataset_start
    update_model = OpponentModel(
        "hero", "villain", archetypes={"tag": PRESETS["tag"]}
    )
    update_start = time.perf_counter()
    for value in samples[:100]:
        update_model.commit_hand(
            (value,),
            observer_context=ObserverContext("hero", value.hand_key, ()),
        )
        update_model.finish_hand()
    update_seconds = time.perf_counter() - update_start
    inference_start = time.perf_counter()
    sample = samples[0]
    model.infer_range_for_hand(
        (sample,),
        observer_context=ObserverContext("hero", sample.hand_key, ()),
    )
    inference_seconds = time.perf_counter() - inference_start
    rows[-1].update(
        {
            "feature_decisions_per_second": len(samples) / feature_seconds,
            "dataset_serialization_decisions_per_second": len(samples)
            / dataset_seconds,
            "dataset_json_bytes_per_1000_decisions": len(dataset_json.encode()),
            "model_updates_per_second": 100 / update_seconds,
            "current_hand_inference_per_second": 1 / inference_seconds,
        }
    )
    tracemalloc.stop()
    return rows


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
