from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from ..cards import Card, full_deck, parse_cards, require_unique
from .actions import Action
from .engine import HoldemGame
from .state import TableConfig


@dataclass(slots=True)
class ScenarioBuilder:
    """Construct exact decision points by building a deck and replaying legal actions."""

    config: TableConfig
    _hole_cards: dict[str, tuple[Card, Card]] = field(default_factory=dict, init=False)
    _board_runout: tuple[Card, ...] = field(default=(), init=False)
    _actions: list[tuple[str, Action]] = field(default_factory=list, init=False)

    def set_hole_cards(
        self, player_id: str, cards: Iterable[Card | str] | str
    ) -> ScenarioBuilder:
        if player_id not in self.config.player_ids:
            raise KeyError(f"unknown player {player_id!r}")
        seat = self.config.player_ids.index(player_id)
        if self.config.starting_stacks[seat] == 0:
            raise ValueError("cannot assign hole cards to an empty seat")
        parsed = parse_cards(cards)
        if len(parsed) != 2:
            raise ValueError("each specified Hold'em hand must contain two cards")
        self._hole_cards[player_id] = (parsed[0], parsed[1])
        self._validate_known_cards()
        return self

    def set_board_runout(self, cards: Iterable[Card | str] | str) -> ScenarioBuilder:
        parsed = parse_cards(cards)
        if len(parsed) not in (0, 3, 4, 5):
            raise ValueError("board runout must contain 0, 3, 4, or 5 cards")
        self._board_runout = parsed
        self._validate_known_cards()
        return self

    def action(self, player_id: str, action: Action) -> ScenarioBuilder:
        if player_id not in self.config.player_ids:
            raise KeyError(f"unknown player {player_id!r}")
        self._actions.append((player_id, action))
        return self

    def build(self) -> HoldemGame:
        self._validate_known_cards()
        deck = self._build_deck()
        game = HoldemGame(self.config, preset_deck=deck)
        game.start_hand()
        for player_id, action in self._actions:
            game.step(action, player_id)
        revealed = tuple(game.board)
        comparable = min(len(revealed), len(self._board_runout))
        if revealed[:comparable] != self._board_runout[:comparable]:
            raise AssertionError(
                "scenario replay did not reveal the requested board prefix"
            )
        return game

    @property
    def known_hole_cards(self) -> dict[str, tuple[Card, Card]]:
        return dict(self._hole_cards)

    @property
    def board_runout(self) -> tuple[Card, ...]:
        return self._board_runout

    @property
    def scripted_actions(self) -> tuple[tuple[str, Action], ...]:
        return tuple(self._actions)

    def _build_deck(self) -> tuple[Card, ...]:
        live_seats = [
            seat for seat, stack in enumerate(self.config.starting_stacks) if stack > 0
        ]
        button = self.config.button
        if button not in live_seats:
            button = self._next_live(button, live_seats)
        first_dealt = self._next_live(button, live_seats)
        deal_order = tuple(
            seat
            for offset in range(len(self.config.player_ids))
            if (seat := (first_dealt + offset) % len(self.config.player_ids))
            in live_seats
        )
        slots: list[Card | None] = [None] * 52
        position = 0
        for card_index in range(2):
            for seat in deal_order:
                player_id = self.config.player_ids[seat]
                specified = self._hole_cards.get(player_id)
                if specified is not None:
                    slots[position] = specified[card_index]
                position += 1
        for card in self._board_runout:
            slots[position] = card
            position += 1

        known = {card for card in slots if card is not None}
        remaining = iter(card for card in full_deck() if card not in known)
        return tuple(card if card is not None else next(remaining) for card in slots)

    def _validate_known_cards(self) -> None:
        known = (
            tuple(card for cards in self._hole_cards.values() for card in cards)
            + self._board_runout
        )
        require_unique(known)

    def _next_live(self, seat: int, live_seats: list[int]) -> int:
        for offset in range(1, len(self.config.player_ids) + 1):
            candidate = (seat + offset) % len(self.config.player_ids)
            if candidate in live_seats:
                return candidate
        raise ValueError("scenario needs at least one live seat")
