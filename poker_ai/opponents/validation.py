from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

from ..agents import PRESETS, PersonalityAgent, StrategyProfile
from ..experiments.schedule import Participant
from ..experiments.simulator import SimulationConfig, SimulationRunner
from .adaptive import AdaptiveExploitPolicy
from .model import OpponentHandBelief, OpponentModel
from .observation import ObservedDecision, ObserverContext


@dataclass(frozen=True, slots=True)
class CalibrationRow:
    true_profile: str
    trial: int
    hands_observed: int
    true_profile_probability: float
    predicted_archetype: str
    log_loss: float


@dataclass(frozen=True, slots=True)
class CalibrationSummary:
    true_profile: str
    hands_observed: int
    trials: int
    accuracy: float
    mean_true_profile_probability: float
    mean_log_loss: float
    confusion_counts: tuple[tuple[str, int], ...]


@dataclass(frozen=True, slots=True)
class PredictionTrace:
    hand_key: str
    model_version_before_prediction: int
    model_version_after_observation: int


@dataclass(frozen=True, slots=True)
class HoldoutResult:
    training_hands: int
    holdout_hands: int
    uniform_frozen_log_loss: float
    fixed_tag_frozen_log_loss: float
    adaptive_frozen_log_loss: float
    adaptive_prequential_log_loss: float
    prequential_trace: tuple[PredictionTrace, ...]


@dataclass(frozen=True, slots=True)
class TendencyConvergenceRow:
    true_profile: str
    tendency: str
    hands_observed: int
    posterior_mean: float
    credible_interval_95: tuple[float, float]
    empirical_reference_rate: float
    reference_opportunities: int


@dataclass(frozen=True, slots=True)
class AdaptivePerformanceResult:
    opponent: str
    historical_training_hands: int
    physical_hands: int
    duplicate_blocks: int
    fixed_bb_per_100: float
    frozen_pretrained_bb_per_100: float
    online_persistent_bb_per_100: float
    reset_each_hand_bb_per_100: float
    persistent_minus_fixed_bb_per_100: float
    persistent_paired_ci_95: tuple[float, float]
    reset_minus_fixed_bb_per_100: float
    reset_paired_ci_95: tuple[float, float]
    online_observations: int


@dataclass(frozen=True, slots=True)
class _SyntheticEvent:
    decision: ObservedDecision
    observer_context: ObserverContext


def calibration_experiment(
    *,
    hands: int = 1_000,
    checkpoints: Iterable[int] = (10, 50, 250, 1_000),
    trials: int = 1,
    seed: int = 0,
) -> tuple[CalibrationRow, ...]:
    rows = []
    for profile_index, (name, profile) in enumerate(PRESETS.items()):
        for trial in range(trials):
            events = _synthetic_events(
                profile, hands, seed + profile_index * trials + trial
            )
            for checkpoint in sorted(value for value in checkpoints if value <= hands):
                model = OpponentModel("observer", "public_player_1")
                for event in events:
                    if event.decision.hand_index < checkpoint:
                        model.observe(
                            event.decision, observer_context=event.observer_context
                        )
                posterior = model.archetype_posterior
                probability = posterior[name]
                rows.append(
                    CalibrationRow(
                        name,
                        trial,
                        checkpoint,
                        probability,
                        max(posterior, key=lambda key: posterior[key]),
                        -math.log(max(probability, 1e-12)),
                    )
                )
    return tuple(rows)


def calibration_summary(
    *, hands: int = 250, trials: int = 20, seed: int = 0
) -> tuple[CalibrationSummary, ...]:
    rows = calibration_experiment(
        hands=hands, checkpoints=(hands,), trials=trials, seed=seed
    )
    summaries = []
    for true_profile in PRESETS:
        values = [row for row in rows if row.true_profile == true_profile]
        confusion = {
            name: sum(row.predicted_archetype == name for row in values)
            for name in PRESETS
        }
        summaries.append(
            CalibrationSummary(
                true_profile,
                hands,
                trials,
                sum(row.predicted_archetype == true_profile for row in values) / trials,
                sum(row.true_profile_probability for row in values) / trials,
                sum(row.log_loss for row in values) / trials,
                tuple(confusion.items()),
            )
        )
    return tuple(summaries)


def confusion_matrix(
    *, hands: int = 50, trials: int = 1, seed: int = 0
) -> tuple[tuple[str, tuple[int, ...]], ...]:
    names = tuple(PRESETS)
    matrix = {name: [0] * len(names) for name in names}
    for row in calibration_experiment(
        hands=hands, checkpoints=(hands,), trials=trials, seed=seed
    ):
        matrix[row.true_profile][names.index(row.predicted_archetype)] += 1
    return tuple((name, tuple(matrix[name])) for name in names)


def holdout_predictive_evaluation(
    true_profile: StrategyProfile,
    *,
    training_hands: int = 50,
    holdout_hands: int = 50,
    seed: int = 0,
) -> HoldoutResult:
    events = _synthetic_events(
        true_profile, training_hands + holdout_hands, seed
    )
    trained = OpponentModel("observer", "public_player_1")
    for event in events:
        if event.decision.hand_index < training_hands:
            trained.observe(event.decision, observer_context=event.observer_context)
    frozen_adaptive = OpponentModel.from_json(trained.to_json())
    prequential = OpponentModel.from_json(trained.to_json())
    uniform = OpponentModel("observer", "public_player_1")
    fixed_tag = OpponentModel(
        "observer", "public_player_1", archetypes={"tag": PRESETS["tag"]}
    )
    holdout = [
        event for event in events if event.decision.hand_index >= training_hands
    ]
    uniform_loss = _frozen_log_loss(
        uniform, holdout, update_archetype_weights=False
    )
    tag_loss = _frozen_log_loss(fixed_tag, holdout)
    adaptive_frozen_loss = _frozen_log_loss(frozen_adaptive, holdout)

    losses = []
    traces = []
    beliefs: dict[object, OpponentHandBelief] = {}
    for event in holdout:
        decision, context = event.decision, event.observer_context
        belief = beliefs.setdefault(decision.hand_key, prequential.start_hand(context))
        before = prequential.model_version
        losses.append(
            -math.log(
                prequential.action_probability(
                    decision,
                    observer_context=context,
                    hand_belief=belief,
                )
            )
        )
        prequential.observe(decision, observer_context=context)
        belief.observe(decision)
        traces.append(
            PredictionTrace(
                f"{decision.hand_key.session_id}:{decision.hand_key.hand_index}",
                before,
                prequential.model_version,
            )
        )
    return HoldoutResult(
        training_hands,
        holdout_hands,
        uniform_loss,
        tag_loss,
        adaptive_frozen_loss,
        sum(losses) / len(losses),
        tuple(traces),
    )


def tendency_convergence_experiment(
    true_profile: StrategyProfile,
    *,
    tendency: str = "vpip",
    reference_hands: int = 2_000,
    checkpoints: Iterable[int] = (25, 100, 500),
    seed: int = 0,
) -> tuple[TendencyConvergenceRow, ...]:
    events = _synthetic_events(true_profile, reference_hands, seed)
    reference = OpponentModel("observer", "public_player_1")
    for event in events:
        reference.observe(event.decision, observer_context=event.observer_context)
    reference_estimate = reference.stats.estimate(tendency)
    empirical = (
        reference_estimate.successes / reference_estimate.opportunities
        if reference_estimate.opportunities
        else 0.0
    )
    rows = []
    for checkpoint in checkpoints:
        if checkpoint > reference_hands:
            continue
        model = OpponentModel("observer", "public_player_1")
        for event in events:
            if event.decision.hand_index < checkpoint:
                model.observe(event.decision, observer_context=event.observer_context)
        estimate = model.stats.estimate(tendency)
        rows.append(
            TendencyConvergenceRow(
                true_profile.name,
                tendency,
                checkpoint,
                estimate.mean,
                estimate.credible_interval_95,
                empirical,
                reference_estimate.opportunities,
            )
        )
    return tuple(rows)


def adaptive_vs_fixed_experiment(
    opponent: StrategyProfile,
    *,
    training_hands: int = 100,
    evaluation_hands: int = 1_000,
    seed: int = 0,
) -> AdaptivePerformanceResult:
    if evaluation_hands % 2:
        raise ValueError("adaptive paired evaluation needs an even hand count")
    training_model = OpponentModel("hero", "public_player_1")
    for event in _synthetic_events(
        opponent, training_hands, seed, observer_id="hero"
    ):
        training_model.observe(
            event.decision, observer_context=event.observer_context
        )
    hero = Participant("hero", "TAG baseline", PRESETS["tag"])
    villain = Participant("villain", opponent.name, opponent)
    config = SimulationConfig(
        (hero.profile, villain.profile),
        hands=evaluation_hands,
        master_seed=seed + 10_000,
        duplicate_deals=True,
        participants=(hero, villain),
        session_id=f"adaptive-eval-{seed}",
    )

    def fixed_factory(participant: Participant, hand: int, seat: int):
        return PersonalityAgent(participant.profile, seed + hand * 2 + seat)

    fixed = SimulationRunner(config, policy_factory=fixed_factory).run()
    frozen_model = OpponentModel.from_json(training_model.to_json())
    frozen = _adaptive_run(config, frozen_model, seed, online=False)
    persistent_model = OpponentModel.from_json(training_model.to_json())
    persistent = _adaptive_run(config, persistent_model, seed, online=True)
    reset = _reset_each_hand_run(config, seed)

    fixed_blocks = _participant_blocks(fixed, "hero")
    frozen_blocks = _participant_blocks(frozen, "hero")
    persistent_blocks = _participant_blocks(persistent, "hero")
    reset_blocks = _participant_blocks(reset, "hero")
    persistent_difference, persistent_ci = _paired_difference(
        persistent_blocks, fixed_blocks
    )
    reset_difference, reset_ci = _paired_difference(reset_blocks, fixed_blocks)
    return AdaptivePerformanceResult(
        opponent.name,
        training_hands,
        evaluation_hands,
        len(fixed_blocks),
        _bb100(fixed_blocks),
        _bb100(frozen_blocks),
        _bb100(persistent_blocks),
        _bb100(reset_blocks),
        persistent_difference,
        persistent_ci,
        reset_difference,
        reset_ci,
        persistent_model.observation_count - training_model.observation_count,
    )


def _frozen_log_loss(
    model: OpponentModel,
    events: list[_SyntheticEvent],
    *,
    update_archetype_weights: bool = True,
) -> float:
    losses = []
    beliefs: dict[object, OpponentHandBelief] = {}
    for event in events:
        decision, context = event.decision, event.observer_context
        if decision.hand_key not in beliefs:
            beliefs[decision.hand_key] = model.start_hand(
                context, update_archetype_weights=update_archetype_weights
            )
        belief = beliefs[decision.hand_key]
        losses.append(
            -math.log(
                model.action_probability(
                    decision,
                    observer_context=context,
                    hand_belief=belief,
                )
            )
        )
        belief.observe(decision)
    return sum(losses) / len(losses)


def _adaptive_run(
    config: SimulationConfig, model: OpponentModel, seed: int, *, online: bool
):
    def factory(participant: Participant, hand: int, seat: int):
        if participant.participant_id == "hero":
            return AdaptiveExploitPolicy(
                participant.profile, model, seed=seed + hand * 2 + seat
            )
        return PersonalityAgent(participant.profile, seed + hand * 2 + seat)

    def observer(decision, private_context):
        if online and decision.public_subject_id == "public_player_1":
            model.observe(decision, observer_context=private_context)

    return SimulationRunner(
        config,
        policy_factory=factory,
        decision_observer=observer,
        observer_participant_id="hero",
        defer_observer_by_duplicate_block=online,
    ).run()


def _reset_each_hand_run(config: SimulationConfig, seed: int):
    holder = {
        "hand": None,
        "model": OpponentModel("hero", "public_player_1"),
    }

    def factory(participant: Participant, hand: int, seat: int):
        if holder["hand"] != hand:
            holder["hand"] = hand
            holder["model"] = OpponentModel("hero", "public_player_1")
        if participant.participant_id == "hero":
            return AdaptiveExploitPolicy(
                participant.profile,
                holder["model"],
                seed=seed + hand * 2 + seat,
            )
        return PersonalityAgent(participant.profile, seed + hand * 2 + seat)

    def observer(decision, private_context):
        if decision.public_subject_id == "public_player_1":
            holder["model"].observe(
                decision, observer_context=private_context
            )

    return SimulationRunner(
        config,
        policy_factory=factory,
        decision_observer=observer,
        observer_participant_id="hero",
        defer_observer_by_duplicate_block=True,
    ).run()


def _synthetic_events(
    profile: StrategyProfile,
    hands: int,
    seed: int,
    *,
    observer_id: str = "observer",
) -> tuple[_SyntheticEvent, ...]:
    observer = Participant(observer_id, "Observer", PRESETS["tag"])
    villain = Participant("villain", profile.name, profile)
    events = []

    def collect(decision, private_context):
        if decision.public_subject_id == "public_player_1":
            events.append(_SyntheticEvent(decision, private_context))

    SimulationRunner(
        SimulationConfig(
            (observer.profile, villain.profile),
            hands=hands,
            master_seed=seed,
            participants=(observer, villain),
            session_id=f"synthetic-{observer_id}-{seed}",
        ),
        decision_observer=collect,
        observer_participant_id=observer_id,
    ).run()
    return tuple(events)


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


def _bb100(values: list[float]) -> float:
    return sum(values) / len(values) * 100


def _paired_difference(
    first: list[float], second: list[float]
) -> tuple[float, tuple[float, float]]:
    differences = [a - b for a, b in zip(first, second)]
    mean = sum(differences) / len(differences)
    variance = (
        sum((value - mean) ** 2 for value in differences) / (len(differences) - 1)
        if len(differences) > 1
        else 0
    )
    difference100 = mean * 100
    se100 = math.sqrt(variance / len(differences)) * 100
    return difference100, (
        difference100 - 1.96 * se100,
        difference100 + 1.96 * se100,
    )
