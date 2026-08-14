from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
import hashlib
import json
from typing import Callable, Iterable, Protocol
import uuid

from ..agents import PersonalityAgent, StrategyProfile, position_name
from ..holdem import (
    Action,
    ActionType,
    HoldemGame,
    LegalActions,
    PlayerObservation,
    Street,
    TableConfig,
)
from ..opponents.observation import (
    HandKey,
    ObservedDecision,
    ObserverContext,
    ResearchDecisionLabels,
    observe_decision,
)
from .metrics import StrategyMetrics, summarize_metrics
from .records import HandExperimentRecord, SeatHandStats
from .schedule import (
    Participant,
    ScheduledHand,
    build_schedule,
    participants_from_profiles,
)


@dataclass(frozen=True, slots=True)
class SimulationConfig:
    profiles: tuple[StrategyProfile, ...]
    hands: int = 1_000
    stack_bb: int = 100
    small_blind: int = 1
    big_blind: int = 2
    master_seed: int = 0
    rotate_profiles: bool = True
    duplicate_deals: bool = False
    record_full_history: bool = False
    participants: tuple[Participant, ...] | None = None
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))


@dataclass(frozen=True, slots=True)
class ExperimentResult:
    config: SimulationConfig
    records: tuple[HandExperimentRecord, ...]
    metrics: tuple[StrategyMetrics, ...]
    position_metrics: tuple[tuple[str, str, StrategyMetrics], ...]
    metadata: dict[str, object]

    def to_json(self, indent: int = 2) -> str:
        """Full privileged research export; use public_observation_json for ML data."""
        return self.research_json(indent)

    def research_json(self, indent: int = 2) -> str:
        return json.dumps(
            asdict(self),
            indent=indent,
            sort_keys=True,
            default=_json_default,
        )

    def public_observation_json(self, indent: int = 2) -> str:
        payload = {
            "schema_version": 1,
            "dataset_type": "public_observed_decisions",
            "decisions": [
                asdict(decision)
                for record in self.records
                for decision in record.observed_decisions
            ],
        }
        return json.dumps(
            payload, indent=indent, sort_keys=True, default=_json_default
        )

    def metrics_csv(self) -> str:
        fields = tuple(asdict(self.metrics[0])) if self.metrics else ()
        lines = [",".join(fields)]
        for metric in self.metrics:
            row = asdict(metric)
            lines.append(
                ",".join(
                    json.dumps(row[field], separators=(",", ":")) for field in fields
                )
            )
        return "\n".join(lines)

    def cumulative_net_bb(self) -> tuple[dict[str, float | int], ...]:
        totals = {metric.profile: 0.0 for metric in self.metrics}
        rows = []
        for record in self.records:
            for seat in record.seats:
                totals[seat.participant_id] += seat.net_bb
            rows.append({"hand": record.hand_index + 1, **totals})
        return tuple(rows)


class DecisionPolicy(Protocol):
    def decide(
        self, observation: PlayerObservation, legal_actions: LegalActions
    ) -> Action: ...


PolicyFactory = Callable[[Participant, int, int], DecisionPolicy]
DecisionObserver = Callable[[ObservedDecision, ObserverContext], None]


class SimulationRunner:
    """Independent cash-hand evaluator with deterministic seed hierarchy."""

    def __init__(
        self,
        config: SimulationConfig,
        *,
        policy_factory: PolicyFactory | None = None,
        decision_observer: DecisionObserver | None = None,
        observer_participant_id: str | None = None,
    ) -> None:
        if not 2 <= len(config.profiles) <= 6 or config.hands < 1:
            raise ValueError("simulation requires 2–6 profiles and positive hands")
        self.config = config
        self.policy_factory = policy_factory
        self.decision_observer = decision_observer
        self.observer_participant_id = observer_participant_id
        self.participants = config.participants or participants_from_profiles(
            config.profiles
        )
        if len(self.participants) != len(config.profiles):
            raise ValueError("participants must match profile seat count")
        identities = tuple(item.participant_id for item in self.participants)
        if len(set(identities)) != len(identities):
            raise ValueError("participant IDs must be unique")
        if decision_observer is not None and observer_participant_id not in identities:
            raise ValueError(
                "decision observers require a valid observer_participant_id"
            )
        self.schedule = build_schedule(
            config.hands,
            len(config.profiles),
            config.duplicate_deals,
            config.rotate_profiles,
        )
        self.session_id = config.session_id

    def run(self) -> ExperimentResult:
        records = tuple(self._run_hand(item) for item in self.schedule)
        identities = tuple(
            participant.participant_id for participant in self.participants
        )
        metrics = tuple(
            summarize_metrics(identity, self._seats(records, identity))
            for identity in identities
        )
        position = []
        for identity in identities:
            positions = sorted(
                {seat.position for seat in self._seats(records, identity)}
            )
            for label in positions:
                position.append(
                    (
                        identity,
                        label,
                        summarize_metrics(
                            identity,
                            (
                                seat
                                for seat in self._seats(records, identity)
                                if seat.position == label
                            ),
                        ),
                    )
                )
        metadata: dict[str, object] = {
            "schedule_type": "cyclic_duplicate_blocks"
            if self.config.duplicate_deals
            else "balanced_button_blocks",
            "duplicate_mode": self.config.duplicate_deals,
            "duplicate_block_size": len(self.participants)
            if self.config.duplicate_deals
            else None,
            "physical_hands": self.config.hands,
            "independent_duplicate_blocks": self.config.hands // len(self.participants)
            if self.config.duplicate_deals
            else 0,
            "master_seed": self.config.master_seed,
            "participants": tuple(
                {
                    "id": p.participant_id,
                    "label": p.label,
                    "fingerprint": p.profile_fingerprint,
                }
                for p in self.participants
            ),
            "button_schedule": tuple(item.button for item in self.schedule),
        }
        return ExperimentResult(
            self.config, records, metrics, tuple(position), metadata
        )

    def _run_hand(self, scheduled: ScheduledHand) -> HandExperimentRecord:
        count = len(self.config.profiles)
        index = scheduled.hand_index
        hand_key = HandKey(self.session_id, index)
        assigned = tuple(
            self.participants[item] for item in scheduled.participant_indices_by_seat
        )
        profiles = tuple(item.profile for item in assigned)
        button = scheduled.button
        deal_index = (
            scheduled.duplicate_block_id if self.config.duplicate_deals else index
        )
        deal_seed = _seed(self.config.master_seed, "deck", deal_index)
        players = tuple(f"P{seat + 1}" for seat in range(count))
        stack = self.config.stack_bb * self.config.big_blind
        game = HoldemGame(
            TableConfig(
                players,
                (stack,) * count,
                self.config.small_blind,
                self.config.big_blind,
                button,
            ),
            seed=deal_seed,
        )
        agents = {
            player: (
                self.policy_factory(assigned[seat], index, seat)
                if self.policy_factory is not None
                else PersonalityAgent(
                    profiles[seat], _seed(self.config.master_seed, "policy", index, seat)
                )
            )
            for seat, player in enumerate(players)
        }
        observed_decisions: list[ObservedDecision] = []
        research_labels: list[ResearchDecisionLabels] = []
        game.start_hand()
        while not game.is_terminal:
            actor = game.current_player
            if actor is None:
                raise AssertionError("non-terminal hand must have a current actor")
            observation = game.observation_for(actor)
            legal = game.legal_actions(actor)
            action = agents[actor].decide(observation, legal)
            participant = assigned[players.index(actor)]
            public_decision = observe_decision(
                hand_key, participant.participant_id, observation, legal, action
            )
            observed_decisions.append(public_decision)
            if self.decision_observer is not None:
                observer_seat = next(
                    seat
                    for seat, item in enumerate(assigned)
                    if item.participant_id == self.observer_participant_id
                )
                observer_player = players[observer_seat]
                private_context = ObserverContext(
                    self.observer_participant_id,
                    hand_key,
                    game.observation_for(observer_player).hole_cards,
                )
                self.decision_observer(public_decision, private_context)
            research_labels.append(
                ResearchDecisionLabels(
                    hand_key,
                    actor,
                    participant.participant_id,
                    participant.profile.name,
                    observation.hole_cards,
                )
            )
            game.step(action, actor)
        result = game.result
        if result is None:
            raise AssertionError("terminal hand must have a result")
        seats = tuple(
            _seat_stats(
                game,
                player,
                profiles[seat].name,
                assigned[seat].participant_id,
                seat,
                stack,
                self.config.big_blind,
                tuple(observed_decisions),
            )
            for seat, player in enumerate(players)
        )
        history = (
            tuple(
                {
                    **asdict(record),
                    "street": record.street.value,
                    "action_type": record.action_type.value,
                }
                for record in game.history
            )
            if self.config.record_full_history
            else None
        )
        return HandExperimentRecord(
            index,
            deal_seed,
            button,
            tuple((player, profiles[seat].name) for seat, player in enumerate(players)),
            tuple(
                (player, assigned[seat].participant_id)
                for seat, player in enumerate(players)
            ),
            scheduled.duplicate_block_id,
            scheduled.duplicate_leg,
            seats,
            result.winners,
            result.showdown,
            len(game.history),
            tuple(observed_decisions),
            tuple(research_labels),
            history,
        )

    @staticmethod
    def _seats(
        records: tuple[HandExperimentRecord, ...], participant_id: str
    ) -> tuple[SeatHandStats, ...]:
        return tuple(
            seat
            for record in records
            for seat in record.seats
            if seat.participant_id == participant_id
        )


def sweep_parameter(
    base_profile: StrategyProfile,
    parameter: str,
    values: Iterable[float],
    opponents: tuple[StrategyProfile, ...],
    *,
    hands: int = 1_000,
    seed: int = 0,
) -> tuple[tuple[float, ExperimentResult], ...]:
    results = []
    for value in values:
        variant = base_profile.with_parameter(parameter, value)
        profiles = (variant, *opponents)
        value_label = format(value, ".12g")
        participants = (
            Participant(
                f"sweep_{parameter}_{value_label}",
                f"{base_profile.name} | {parameter}={value_label}",
                variant,
            ),
            *tuple(
                Participant(f"opponent_{index}", profile.name, profile)
                for index, profile in enumerate(opponents, start=1)
            ),
        )
        result = SimulationRunner(
            SimulationConfig(
                profiles,
                hands=hands,
                master_seed=seed,
                participants=participants,
            )
        ).run()
        results.append((value, result))
    return tuple(results)


def _seed(master: int, *parts: object) -> int:
    payload = ":".join(map(str, (master, *parts))).encode()
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big")


def _json_default(value: object) -> object:
    return value.value if isinstance(value, Enum) else str(value)


def _seat_stats(
    game: HoldemGame,
    player: str,
    profile: str,
    participant_id: str,
    seat: int,
    starting: int,
    big_blind: int,
    observed_decisions: tuple[ObservedDecision, ...],
) -> SeatHandStats:
    observation = game.observation_for(player)
    records = tuple(record for record in game.history if record.player_id == player)
    preflop = tuple(record for record in records if record.street == Street.PREFLOP)
    voluntary = tuple(
        record
        for record in preflop
        if record.action_type
        not in {
            ActionType.SMALL_BLIND,
            ActionType.BIG_BLIND,
            ActionType.CHECK,
            ActionType.FOLD,
        }
    )
    opportunities, three_bets = _three_bet_counts(observed_decisions, player)
    actions = tuple(
        record
        for record in records
        if record.action_type not in {ActionType.SMALL_BLIND, ActionType.BIG_BLIND}
    )
    post = tuple(record for record in actions if record.street != Street.PREFLOP)
    if game.result is None:
        raise AssertionError("metrics require a terminal hand")
    final_stack = game.result.final_stacks[player]
    return SeatHandStats(
        player,
        profile,
        participant_id,
        seat,
        position_name(observation, player),
        final_stack - starting,
        (final_stack - starting) / big_blind,
        bool(voluntary),
        any(record.action_type == ActionType.RAISE for record in preflop),
        opportunities,
        three_bets,
        sum(record.action_type == ActionType.FOLD for record in actions),
        sum(record.action_type == ActionType.CHECK for record in actions),
        sum(record.action_type == ActionType.CALL for record in actions),
        sum(
            record.action_type in {ActionType.BET, ActionType.RAISE}
            for record in actions
        ),
        len(post),
        sum(
            record.action_type in {ActionType.BET, ActionType.RAISE} for record in post
        ),
        game.result.showdown
        and not any(record.action_type == ActionType.FOLD for record in records),
    )


def _three_bet_counts(
    decisions: Iterable[ObservedDecision], player: str
) -> tuple[int, int]:
    """Count only decisions where the first re-raise was actually legal."""
    opportunities = tuple(
        decision
        for decision in decisions
        if decision.player_id == player
        and decision.street == Street.PREFLOP
        and decision.prior_voluntary_raises == 1
        and decision.can_raise
    )
    return len(opportunities), sum(
        decision.action_family == "raise" for decision in opportunities
    )
