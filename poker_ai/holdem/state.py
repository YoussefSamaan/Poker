from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Mapping

from ..cards import Card
from .actions import LegalActions


class Street(Enum):
    PREFLOP = "preflop"
    FLOP = "flop"
    TURN = "turn"
    RIVER = "river"
    SHOWDOWN = "showdown"


class PlayerStatus(Enum):
    ACTIVE = "active"
    FOLDED = "folded"
    ALL_IN = "all_in"
    OUT = "out"


class ActionType(Enum):
    SMALL_BLIND = "small_blind"
    BIG_BLIND = "big_blind"
    FOLD = "fold"
    CHECK = "check"
    CALL = "call"
    BET = "bet"
    RAISE = "raise"


def _require_chip_amount(name: str, value: int, *, positive: bool = False) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer chip amount")
    if value < (1 if positive else 0):
        qualifier = "positive" if positive else "non-negative"
        raise ValueError(f"{name} must be {qualifier}")


@dataclass(frozen=True, slots=True)
class TableConfig:
    player_ids: tuple[str, ...]
    starting_stacks: tuple[int, ...]
    small_blind: int = 1
    big_blind: int = 2
    button: int = 0

    def __post_init__(self) -> None:
        if not 2 <= len(self.player_ids) <= 6:
            raise ValueError("Hold'em requires 2 to 6 seats")
        if len(set(self.player_ids)) != len(self.player_ids):
            raise ValueError("player ids must be unique")
        if len(self.starting_stacks) != len(self.player_ids):
            raise ValueError("starting_stacks must match player_ids")
        _require_chip_amount("small_blind", self.small_blind, positive=True)
        _require_chip_amount("big_blind", self.big_blind, positive=True)
        if self.small_blind >= self.big_blind:
            raise ValueError("small_blind must be smaller than big_blind")
        for stack in self.starting_stacks:
            _require_chip_amount("starting stack", stack)
        if not 0 <= self.button < len(self.player_ids):
            raise ValueError("button index is outside the table")

    @classmethod
    def from_mapping(
        cls,
        stacks: Mapping[str, int],
        *,
        small_blind: int = 1,
        big_blind: int = 2,
        button: int = 0,
    ) -> TableConfig:
        return cls(
            tuple(stacks), tuple(stacks.values()), small_blind, big_blind, button
        )


@dataclass(slots=True)
class PlayerState:
    player_id: str
    seat: int
    stack: int
    status: PlayerStatus = PlayerStatus.ACTIVE
    hole_cards: tuple[Card, ...] = ()
    street_contribution: int = 0
    total_contribution: int = 0


@dataclass(frozen=True, slots=True)
class PublicPlayerState:
    player_id: str
    seat: int
    stack: int
    status: PlayerStatus
    street_contribution: int
    total_contribution: int


@dataclass(frozen=True, slots=True)
class InternalPlayerState(PublicPlayerState):
    hole_cards: tuple[Card, ...]


@dataclass(frozen=True, slots=True)
class ActionRecord:
    sequence: int
    street: Street
    player_id: str
    action_type: ActionType
    amount_paid: int
    target_to: int | None
    contribution_before: int
    contribution_after: int
    amount_to_call_before: int
    pot_before: int
    pot_after: int
    stack_before: int
    stack_after: int
    caused_all_in: bool


@dataclass(frozen=True, slots=True)
class PotResult:
    amount: int
    eligible_players: tuple[str, ...]
    winners: tuple[str, ...]
    payouts: Mapping[str, int]


@dataclass(frozen=True, slots=True)
class HandResult:
    reason: str
    showdown: bool
    winners: tuple[str, ...]
    payouts: Mapping[str, int]
    pots: tuple[PotResult, ...]
    final_stacks: Mapping[str, int]


@dataclass(frozen=True, slots=True)
class PlayerObservation:
    player_id: str
    hole_cards: tuple[Card, ...]
    board: tuple[Card, ...]
    street: Street
    button_player: str
    small_blind_player: str
    big_blind_player: str
    current_player: str | None
    pot: int
    current_bet: int
    players: tuple[PublicPlayerState, ...]
    history: tuple[ActionRecord, ...]
    legal_actions: LegalActions | None
    is_terminal: bool


@dataclass(frozen=True, slots=True)
class InternalState:
    players: tuple[InternalPlayerState, ...]
    board: tuple[Card, ...]
    remaining_deck: tuple[Card, ...]
    street: Street
    button_player: str
    small_blind_player: str
    big_blind_player: str
    current_player: str | None
    current_bet: int
    last_full_raise_size: int
    pending_players: tuple[str, ...]
    raise_rights: tuple[str, ...]
    history: tuple[ActionRecord, ...]
    is_terminal: bool


@dataclass(frozen=True, slots=True)
class Transition:
    action_record: ActionRecord
    street_changed: bool
    cards_revealed: tuple[Card, ...]
    hand_terminated: bool
    next_player: str | None
