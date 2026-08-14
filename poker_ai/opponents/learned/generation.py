from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Mapping

from ...agents import PRESETS, StrategyProfile
from ...experiments.schedule import Participant
from ...experiments.simulator import ExperimentResult, SimulationConfig, SimulationRunner
from ..dataset import PublicDecisionExample, PublicObservationDataset
from .hand_conditioned import (
    ResearchHandConditionedDataset,
    build_research_hand_conditioned_dataset,
)


@dataclass(frozen=True, slots=True)
class SyntheticOpponentDatasetBundle:
    public_examples: tuple[PublicDecisionExample, ...]
    research: ResearchHandConditionedDataset
    results: tuple[ExperimentResult, ...]
    privileged_composition: tuple[tuple[str, int, int], ...]


def generate_balanced_synthetic_dataset(
    *,
    hands_per_personality: int = 1_000,
    sessions_per_personality: int = 2,
    seed: int = 0,
    profiles: Mapping[str, StrategyProfile] = PRESETS,
) -> SyntheticOpponentDatasetBundle:
    if hands_per_personality < 1 or sessions_per_personality < 1:
        raise ValueError("dataset sizes must be positive")
    results = []
    composition = []
    for profile_index, (name, profile) in enumerate(profiles.items()):
        base, remainder = divmod(hands_per_personality, sessions_per_personality)
        for session in range(sessions_per_personality):
            hands = base + int(session < remainder)
            if not hands:
                continue
            observer = Participant("research_observer", "Observer", PRESETS["tag"])
            target = Participant("research_target", "Target", profile)
            results.append(
                SimulationRunner(
                    SimulationConfig(
                        (observer.profile, target.profile),
                        hands=hands,
                        master_seed=seed + profile_index * 10_000 + session,
                        participants=(observer, target),
                        session_id=f"research-{seed}-{profile_index}-{session}",
                    )
                ).run()
            )
        composition.append((name, hands_per_personality, sessions_per_personality))
    public = tuple(
        example
        for result in results
        for example in PublicObservationDataset.from_experiment(result).examples
        if example.public_subject_id == "public_player_1"
    )
    research = build_research_hand_conditioned_dataset(results)
    research = replace(
        research,
        examples=tuple(
            example
            for example in research.examples
            if example.public.public_subject_id == "public_player_1"
        ),
    )
    return SyntheticOpponentDatasetBundle(
        public, research, tuple(results), tuple(composition)
    )


def parameterized_ood_profiles() -> dict[str, StrategyProfile]:
    """Synthetic perturbations that are not claims about human player types."""
    tag = PRESETS["tag"].with_parameter(
        "bluff_frequency", min(1.0, PRESETS["tag"].bluff_frequency * 1.2)
    )
    lag = PRESETS["lag"].with_parameter(
        "three_bet_frequency", PRESETS["lag"].three_bet_frequency * 0.7
    )
    caller = PRESETS["calling_station"].with_parameter(
        "call_open_frequency",
        min(1.0, PRESETS["calling_station"].call_open_frequency * 1.15),
    )
    return {
        "tag_bluff_plus_20pct": replace(tag, name="Synthetic TAG bluff +20%"),
        "lag_three_bet_minus_30pct": replace(
            lag, name="Synthetic LAG 3-bet -30%"
        ),
        "calling_station_call_plus_15pct": replace(
            caller, name="Synthetic Calling Station call +15%"
        ),
    }
