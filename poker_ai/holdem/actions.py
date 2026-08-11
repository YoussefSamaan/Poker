from __future__ import annotations

from dataclasses import dataclass


class IllegalAction(ValueError):
    """Raised when an action is not legal in the current game state."""


class Action:
    """Marker base class for typed player actions."""


@dataclass(frozen=True, slots=True)
class Fold(Action):
    pass


@dataclass(frozen=True, slots=True)
class Check(Action):
    pass


@dataclass(frozen=True, slots=True)
class Call(Action):
    pass


@dataclass(frozen=True, slots=True)
class BetTo(Action):
    """Make the player's total contribution on this street equal ``amount``."""

    amount: int


@dataclass(frozen=True, slots=True)
class RaiseTo(Action):
    """Raise the player's total contribution on this street to ``amount``."""

    amount: int


@dataclass(frozen=True, slots=True)
class LegalActions:
    player_id: str
    can_fold: bool
    can_check: bool
    call_amount: int | None
    min_bet_to: int | None
    max_bet_to: int | None
    min_raise_to: int | None
    max_raise_to: int | None

    @property
    def can_call(self) -> bool:
        return self.call_amount is not None

    @property
    def can_bet(self) -> bool:
        return self.min_bet_to is not None

    @property
    def can_raise(self) -> bool:
        return self.max_raise_to is not None
