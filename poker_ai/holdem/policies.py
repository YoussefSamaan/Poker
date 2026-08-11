from __future__ import annotations

import random
from typing import Protocol

from .actions import Action, BetTo, Call, Check, Fold, LegalActions, RaiseTo
from .state import PlayerObservation


class Policy(Protocol):
    def decide(
        self, observation: PlayerObservation, legal_actions: LegalActions
    ) -> Action: ...


class CheckCallPolicy:
    """Deterministic testing baseline that never bets or raises."""

    def decide(
        self, observation: PlayerObservation, legal_actions: LegalActions
    ) -> Action:
        if legal_actions.can_check:
            return Check()
        if legal_actions.can_call:
            return Call()
        return Fold()


class RandomLegalPolicy:
    """Seeded testing policy that samples only from engine-advertised actions."""

    def __init__(self, seed: int | None = None) -> None:
        self._rng = random.Random(seed)

    def decide(
        self, observation: PlayerObservation, legal_actions: LegalActions
    ) -> Action:
        choices: list[Action] = []
        if legal_actions.can_fold:
            choices.append(Fold())
        if legal_actions.can_check:
            choices.append(Check())
        if legal_actions.can_call:
            choices.append(Call())
        if legal_actions.can_bet:
            min_bet = legal_actions.min_bet_to
            max_bet = legal_actions.max_bet_to
            if min_bet is None or max_bet is None:
                raise AssertionError("can_bet requires concrete bounds")
            choices.append(BetTo(self._rng.randint(min_bet, max_bet)))
        if legal_actions.can_raise:
            min_raise = legal_actions.min_raise_to
            max_raise = legal_actions.max_raise_to
            if min_raise is None or max_raise is None:
                raise AssertionError("can_raise requires concrete bounds")
            choices.append(RaiseTo(self._rng.randint(min_raise, max_raise)))
        if not choices:
            raise RuntimeError("engine exposed no legal action")
        return self._rng.choice(choices)
