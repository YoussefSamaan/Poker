from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Iterable, Mapping

from .cards import Card, full_deck, parse_cards


@dataclass(frozen=True, slots=True)
class WeightedCombo:
    cards: tuple[Card, Card]
    weight: float = 1.0

    def __post_init__(self) -> None:
        if len(self.cards) != 2 or self.cards[0] == self.cards[1]:
            raise ValueError("an opponent combo must contain two distinct cards")
        if self.weight <= 0:
            raise ValueError("combo weights must be positive")
        object.__setattr__(self, "cards", tuple(sorted(self.cards, key=str)))


class WeightedRange:
    """An explicit distribution over two-card opponent combinations."""

    def __init__(self, combos: Iterable[WeightedCombo]):
        merged: dict[frozenset[Card], WeightedCombo] = {}
        for combo in combos:
            key = frozenset(combo.cards)
            previous = merged.get(key)
            merged[key] = WeightedCombo(
                combo.cards,
                combo.weight + (previous.weight if previous else 0.0),
            )
        self.combos = tuple(merged.values())
        if not self.combos:
            raise ValueError("a range must contain at least one combo")

    @classmethod
    def from_mapping(cls, combos: Mapping[str, float]) -> WeightedRange:
        parsed: list[WeightedCombo] = []
        for text, weight in combos.items():
            compact = text.replace(" ", "").replace(",", "")
            if len(compact) != 4:
                raise ValueError(f"explicit combo {text!r} must look like 'AsKh'")
            cards = parse_cards((compact[:2], compact[2:]))
            parsed.append(WeightedCombo((cards[0], cards[1]), float(weight)))
        return cls(parsed)

    @classmethod
    def random(cls, dead_cards: Iterable[Card] = ()) -> WeightedRange:
        dead = set(dead_cards)
        available = [card for card in full_deck() if card not in dead]
        return cls(WeightedCombo(combo) for combo in combinations(available, 2))

    def without_blocked(self, dead_cards: Iterable[Card]) -> WeightedRange:
        dead = set(dead_cards)
        legal = [combo for combo in self.combos if not dead.intersection(combo.cards)]
        if not legal:
            raise ValueError("every opponent combo is blocked by known cards")
        return WeightedRange(legal)
