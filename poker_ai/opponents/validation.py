from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

from ..agents import PRESETS, StrategyProfile
from ..experiments.schedule import Participant
from ..experiments.simulator import SimulationConfig, SimulationRunner
from .adaptive import AdaptiveExploitPolicy
from .model import OpponentModel
from .observation import ObservedDecision


@dataclass(frozen=True, slots=True)
class CalibrationRow:
    true_profile: str
    hands_observed: int
    true_profile_probability: float
    predicted_archetype: str
    log_loss: float


@dataclass(frozen=True, slots=True)
class HoldoutResult:
    training_hands: int
    holdout_hands: int
    uniform_mixture_log_loss: float
    fixed_tag_log_loss: float
    adaptive_mixture_log_loss: float


@dataclass(frozen=True, slots=True)
class AdaptivePerformanceResult:
    opponent: str
    physical_hands: int
    duplicate_blocks: int
    fixed_bb_per_100: float
    adaptive_bb_per_100: float
    difference_bb_per_100: float
    paired_ci_95: tuple[float, float]


def calibration_experiment(
    *, hands: int = 1_000, checkpoints: Iterable[int] = (10, 50, 250, 1_000), seed: int = 0
) -> tuple[CalibrationRow, ...]:
    rows = []
    for index, (name, profile) in enumerate(PRESETS.items()):
        decisions = _synthetic_decisions(profile, hands, seed + index)
        for checkpoint in sorted(value for value in checkpoints if value <= hands):
            model = OpponentModel("observer", "villain")
            for decision in decisions:
                if decision.hand_index < checkpoint:
                    model.observe(decision)
            posterior = model.archetype_posterior
            probability = posterior[name]
            rows.append(
                CalibrationRow(
                    name,
                    checkpoint,
                    probability,
                    max(posterior, key=posterior.get),
                    -math.log(max(probability, 1e-12)),
                )
            )
    return tuple(rows)


def confusion_matrix(
    *, hands: int = 50, seed: int = 0
) -> tuple[tuple[str, tuple[int, ...]], ...]:
    names = tuple(PRESETS)
    matrix = {name: [0] * len(names) for name in names}
    for row in calibration_experiment(hands=hands, checkpoints=(hands,), seed=seed):
        matrix[row.true_profile][names.index(row.predicted_archetype)] += 1
    return tuple((name, tuple(matrix[name])) for name in names)


def holdout_predictive_evaluation(
    true_profile: StrategyProfile,
    *,
    training_hands: int = 50,
    holdout_hands: int = 50,
    seed: int = 0,
) -> HoldoutResult:
    decisions = _synthetic_decisions(
        true_profile, training_hands + holdout_hands, seed
    )
    adaptive = OpponentModel("observer", "villain")
    uniform = OpponentModel("observer", "villain")
    fixed_tag = OpponentModel(
        "observer", "villain", archetypes={"tag": PRESETS["tag"]}
    )
    adaptive_losses = []
    uniform_losses = []
    tag_losses = []
    for decision in decisions:
        if decision.hand_index < training_hands:
            adaptive.observe(decision)
            continue
        adaptive_losses.append(-math.log(adaptive.action_probability(decision)))
        uniform_losses.append(-math.log(uniform.action_probability(decision)))
        tag_losses.append(-math.log(fixed_tag.action_probability(decision)))
        adaptive.observe(decision)
        uniform_prior = dict(uniform.log_posterior)
        uniform.observe(decision)
        uniform.log_posterior = uniform_prior
        fixed_tag.observe(decision)
    count = len(adaptive_losses)
    return HoldoutResult(
        training_hands,
        holdout_hands,
        sum(uniform_losses) / count,
        sum(tag_losses) / count,
        sum(adaptive_losses) / count,
    )


def adaptive_vs_fixed_experiment(
    opponent: StrategyProfile,
    *,
    training_hands: int = 100,
    evaluation_hands: int = 1_000,
    seed: int = 0,
) -> AdaptivePerformanceResult:
    if evaluation_hands % 2:
        raise ValueError("adaptive paired evaluation needs an even hand count")
    model = OpponentModel("hero", "villain")
    for decision in _synthetic_decisions(opponent, training_hands, seed):
        model.observe(decision)
    hero = Participant("hero", "TAG baseline", PRESETS["tag"])
    villain = Participant("villain", opponent.name, opponent)
    config = SimulationConfig(
        (hero.profile, villain.profile),
        hands=evaluation_hands,
        master_seed=seed + 10_000,
        duplicate_deals=True,
        participants=(hero, villain),
    )
    from ..agents import PersonalityAgent

    def fixed_factory(participant: Participant, hand: int, seat: int):
        return PersonalityAgent(participant.profile, seed + hand * 2 + seat)

    fixed = SimulationRunner(config, policy_factory=fixed_factory).run()

    def factory(participant: Participant, hand: int, seat: int):
        if participant.participant_id == "hero":
            return AdaptiveExploitPolicy(
                participant.profile, model, seed=seed + hand * 2 + seat
            )
        return PersonalityAgent(participant.profile, seed + hand * 2 + seat)

    adaptive = SimulationRunner(config, policy_factory=factory).run()
    fixed_blocks = _participant_blocks(fixed, "hero")
    adaptive_blocks = _participant_blocks(adaptive, "hero")
    differences = [a - f for a, f in zip(adaptive_blocks, fixed_blocks)]
    mean = sum(differences) / len(differences)
    variance = (
        sum((value - mean) ** 2 for value in differences) / (len(differences) - 1)
        if len(differences) > 1 else 0
    )
    se100 = math.sqrt(variance / len(differences)) * 100
    fixed100 = sum(fixed_blocks) / len(fixed_blocks) * 100
    adaptive100 = sum(adaptive_blocks) / len(adaptive_blocks) * 100
    difference100 = mean * 100
    return AdaptivePerformanceResult(
        opponent.name,
        evaluation_hands,
        len(differences),
        fixed100,
        adaptive100,
        difference100,
        (difference100 - 1.96 * se100, difference100 + 1.96 * se100),
    )


def _synthetic_decisions(
    profile: StrategyProfile, hands: int, seed: int
) -> tuple[ObservedDecision, ...]:
    hero = Participant("observer", "Observer", PRESETS["tag"])
    villain = Participant("villain", profile.name, profile)
    result = SimulationRunner(
        SimulationConfig(
            (hero.profile, villain.profile),
            hands=hands,
            master_seed=seed,
            participants=(hero, villain),
        )
    ).run()
    return tuple(
        decision
        for record in result.records
        for decision in record.observed_decisions
        if decision.participant_id == "villain"
    )


def _participant_blocks(result, participant_id: str) -> list[float]:
    blocks = []
    block_count = result.metadata["independent_duplicate_blocks"]
    for block in range(block_count):
        values = [
            seat.net_bb
            for record in result.records
            if record.duplicate_block_id == block
            for seat in record.seats
            if seat.participant_id == participant_id
        ]
        blocks.append(sum(values) / len(values))
    return blocks
