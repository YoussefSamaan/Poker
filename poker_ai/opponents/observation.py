from __future__ import annotations

from dataclasses import dataclass

from ..agents import position_name
from ..cards import Card
from ..holdem import (
    Action,
    ActionRecord,
    BetTo,
    Call,
    Check,
    Fold,
    LegalActions,
    PlayerObservation,
    PublicPlayerState,
    RaiseTo,
    Street,
)


@dataclass(frozen=True, slots=True, order=True)
class HandKey:
    session_id: str
    hand_index: int

    def __post_init__(self) -> None:
        if not self.session_id:
            raise ValueError("hand session_id cannot be empty")
        if self.hand_index < 0:
            raise ValueError("hand index cannot be negative")


@dataclass(frozen=True, slots=True)
class ObserverContext:
    observer_id: str
    hand_key: HandKey
    observer_known_cards: tuple[Card, ...]


@dataclass(frozen=True, slots=True)
class ObservedDecision:
    """Public information available immediately before and after one action."""

    hand_key: HandKey
    player_id: str
    public_subject_id: str
    street: Street
    position: str
    board: tuple[Card, ...]
    button_player: str
    small_blind_player: str
    big_blind_player: str
    players: tuple[PublicPlayerState, ...]
    history: tuple[ActionRecord, ...]
    pot: int
    current_bet: int
    to_call: int
    can_fold: bool
    can_check: bool
    can_call: bool
    can_bet: bool
    can_raise: bool
    prior_voluntary_raises: int
    previous_aggressor: str | None
    active_players: int
    action_family: str
    action_amount: int | None
    bet_fraction_of_pot: float | None

    @property
    def hand_index(self) -> int:
        return self.hand_key.hand_index

    @property
    def participant_id(self) -> str:
        """Deprecated alias for public_subject_id; never a research participant ID."""
        return self.public_subject_id

    def legal_actions(self) -> LegalActions:
        player = next(item for item in self.players if item.player_id == self.player_id)
        maximum = player.street_contribution + player.stack
        return LegalActions(
            self.player_id,
            self.can_fold,
            self.can_check,
            self.to_call if self.can_call else None,
            self.current_bet + 1 if self.can_bet else None,
            maximum if self.can_bet else None,
            self.current_bet + 1 if self.can_raise else None,
            maximum if self.can_raise else None,
        )

    def observation_with_hole_cards(
        self, cards: tuple[Card, Card]
    ) -> PlayerObservation:
        return PlayerObservation(
            self.player_id,
            cards,
            self.board,
            self.street,
            self.button_player,
            self.small_blind_player,
            self.big_blind_player,
            self.player_id,
            self.pot,
            self.current_bet,
            self.players,
            self.history,
            self.legal_actions(),
            False,
        )


@dataclass(frozen=True, slots=True)
class ResearchDecisionLabels:
    """Privileged synthetic-only labels, never accepted by opponent models."""

    hand_key: HandKey
    player_id: str
    public_subject_id: str
    participant_id: str
    true_profile_name: str
    true_hole_cards: tuple[Card, ...]


def observe_decision(
    hand_key: HandKey,
    public_subject_id: str,
    observation: PlayerObservation,
    legal: LegalActions,
    action: Action,
) -> ObservedDecision:
    aggressive = tuple(
        record
        for record in observation.history
        if record.street == observation.street
        and record.action_type.value in {"bet", "raise"}
    )
    family = action_family(action)
    amount = action.amount if isinstance(action, (BetTo, RaiseTo)) else None
    actor = next(
        player for player in observation.players if player.player_id == observation.player_id
    )
    incremental_amount = (
        amount - actor.street_contribution if amount is not None else None
    )
    return ObservedDecision(
        hand_key,
        observation.player_id,
        public_subject_id,
        observation.street,
        position_name(observation, observation.player_id),
        observation.board,
        observation.button_player,
        observation.small_blind_player,
        observation.big_blind_player,
        observation.players,
        observation.history,
        observation.pot,
        observation.current_bet,
        legal.call_amount or 0,
        legal.can_fold,
        legal.can_check,
        legal.can_call,
        legal.can_bet,
        legal.can_raise,
        sum(
            record.action_type.value == "raise"
            for record in observation.history
            if record.street == Street.PREFLOP
        ),
        aggressive[-1].player_id if aggressive else None,
        sum(player.status.value == "active" for player in observation.players),
        family,
        amount,
        (
            incremental_amount / observation.pot
            if incremental_amount is not None and observation.pot
            else None
        ),
    )


def action_family(action: Action) -> str:
    if isinstance(action, Fold):
        return "fold"
    if isinstance(action, Check):
        return "check"
    if isinstance(action, Call):
        return "call"
    if isinstance(action, BetTo):
        return "bet"
    if isinstance(action, RaiseTo):
        return "raise"
    raise TypeError(f"unsupported action type {type(action).__name__}")
