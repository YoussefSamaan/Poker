from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import random
from typing import TYPE_CHECKING, Iterable

from ..holdem import ActionType
from .observation import ObservedDecision

if TYPE_CHECKING:
    from ..experiments.simulator import ExperimentResult


@dataclass(frozen=True, slots=True)
class OpponentFeatureVector:
    street: str
    position: str
    active_players: int
    pot_bb: float
    to_call_bb: float
    pot_odds: float
    effective_stack_bb: float
    stack_to_pot: float
    can_fold: int
    can_check: int
    can_call: int
    can_bet: int
    can_raise: int
    prior_voluntary_raises: int
    previous_aggressor_relationship: str
    board_paired: int
    board_max_suit_count: int
    history_bets: int
    history_raises: int
    history_calls: int
    history_checks: int


@dataclass(frozen=True, slots=True)
class PublicDecisionExample:
    dataset_session_id: str
    hand_index: int
    decision_sequence: int
    public_subject_id: str
    correlation_group_id: str
    features: OpponentFeatureVector
    chosen_action_family: str


@dataclass(frozen=True, slots=True)
class PublicObservationDataset:
    schema_version: int
    dataset_id: str
    examples: tuple[PublicDecisionExample, ...]

    SCHEMA_VERSION = 1

    @classmethod
    def from_experiment(cls, result: ExperimentResult) -> PublicObservationDataset:
        examples = []
        first_decision = next(
            (
                decision
                for record in result.records
                for decision in record.observed_decisions
            ),
            None,
        )
        public_session_id = (
            first_decision.hand_key.session_id
            if first_decision is not None
            else "empty_dataset"
        )
        for record in result.records:
            group = (
                f"{public_session_id}:duplicate:{record.duplicate_block_id}"
                if record.duplicate_block_id is not None
                else f"{public_session_id}:hand:{record.hand_index}"
            )
            for decision in record.observed_decisions:
                examples.append(
                    PublicDecisionExample(
                        public_session_id,
                        record.hand_index,
                        len(decision.history),
                        decision.public_subject_id,
                        group,
                        public_decision_features(decision),
                        decision.action_family,
                    )
                )
        return cls(cls.SCHEMA_VERSION, public_session_id, tuple(examples))

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(asdict(self), indent=indent, sort_keys=True)


@dataclass(frozen=True, slots=True)
class DatasetSplit:
    train: tuple[PublicDecisionExample, ...]
    validation: tuple[PublicDecisionExample, ...]
    test: tuple[PublicDecisionExample, ...]


def public_decision_features(decision: ObservedDecision) -> OpponentFeatureVector:
    """Deterministic ML inputs derived exclusively from one public decision."""
    big_blind = max(
        (
            record.amount_paid
            for record in decision.history
            if record.action_type == ActionType.BIG_BLIND
        ),
        default=1,
    )
    actor = next(player for player in decision.players if player.player_id == decision.player_id)
    opponents = [
        player
        for player in decision.players
        if player.player_id != decision.player_id and player.status.value == "active"
    ]
    effective = min(actor.stack, max((player.stack for player in opponents), default=0))
    suits = {card.suit: sum(item.suit == card.suit for item in decision.board) for card in decision.board}
    ranks = [card.rank for card in decision.board]
    history = decision.history
    relationship = (
        "self"
        if decision.previous_aggressor == decision.player_id
        else "other" if decision.previous_aggressor is not None else "none"
    )
    return OpponentFeatureVector(
        decision.street.value,
        decision.position,
        decision.active_players,
        decision.pot / big_blind,
        decision.to_call / big_blind,
        decision.to_call / (decision.pot + decision.to_call)
        if decision.to_call > 0
        else 0.0,
        effective / big_blind,
        effective / decision.pot if decision.pot else 0.0,
        int(decision.can_fold),
        int(decision.can_check),
        int(decision.can_call),
        int(decision.can_bet),
        int(decision.can_raise),
        decision.prior_voluntary_raises,
        relationship,
        int(len(set(ranks)) < len(ranks)),
        max(suits.values(), default=0),
        sum(record.action_type == ActionType.BET for record in history),
        sum(record.action_type == ActionType.RAISE for record in history),
        sum(record.action_type == ActionType.CALL for record in history),
        sum(record.action_type == ActionType.CHECK for record in history),
    )


def grouped_train_validation_test_split(
    examples: Iterable[PublicDecisionExample],
    *,
    validation_fraction: float = 0.15,
    test_fraction: float = 0.15,
    seed: int = 0,
) -> DatasetSplit:
    """Split correlation groups atomically; duplicate legs and hands never cross."""
    if not 0 <= validation_fraction < 1 or not 0 <= test_fraction < 1:
        raise ValueError("split fractions must be in [0, 1)")
    if validation_fraction + test_fraction >= 1:
        raise ValueError("validation and test fractions must sum to less than one")
    groups: dict[str, list[PublicDecisionExample]] = {}
    for example in examples:
        groups.setdefault(example.correlation_group_id, []).append(example)
    keys = sorted(groups)
    random.Random(seed).shuffle(keys)
    validation_count = round(len(keys) * validation_fraction)
    test_count = round(len(keys) * test_fraction)
    validation_keys = set(keys[:validation_count])
    test_keys = set(keys[validation_count : validation_count + test_count])
    train_keys = set(keys) - validation_keys - test_keys

    def rows(selected: set[str]) -> tuple[PublicDecisionExample, ...]:
        return tuple(item for key in keys if key in selected for item in groups[key])

    return DatasetSplit(rows(train_keys), rows(validation_keys), rows(test_keys))
