from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
import re
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

    @property
    def total_weight(self) -> float:
        return sum(combo.weight for combo in self.combos)


RANGE_RANKS = "AKQJT98765432"
_CLASS_PATTERN = re.compile(r"^([AKQJT2-9])([AKQJT2-9])([so]?)(\+?)$")


@dataclass(frozen=True, slots=True)
class RangeStats:
    raw_combo_count: int
    legal_combo_count: int
    blocked_combo_count: int
    legal_total_weight: float
    raw_preflop_coverage: float
    legal_fraction_of_original_range: float

    @property
    def total_weight(self) -> float:
        """Backward-compatible alias for legal_total_weight."""
        return self.legal_total_weight

    @property
    def coverage(self) -> float:
        """Backward-compatible alias for raw_preflop_coverage."""
        return self.raw_preflop_coverage


@dataclass(frozen=True, slots=True)
class PreflopRange:
    """Weighted conventional hand classes expanded to concrete card combos.

    ``+`` on a pair includes that pair and every higher pair. On a non-pair it
    keeps the first (higher) rank fixed and raises the second rank through the
    rank immediately below it: ``ATs+`` is ``ATs,AJs,AQs,AKs``.
    """

    class_weights: tuple[tuple[str, float], ...]

    @classmethod
    def parse(cls, text: str) -> PreflopRange:
        tokens = [token for token in re.split(r"[\s,]+", text.strip()) if token]
        if not tokens:
            raise ValueError("range expression cannot be empty")
        expanded: dict[str, float] = {}
        for token in tokens:
            pieces = token.split(":")
            if len(pieces) > 2:
                raise ValueError(f"invalid weighted range token {token!r}")
            expression = pieces[0]
            try:
                weight = float(pieces[1]) if len(pieces) == 2 else 1.0
            except ValueError as error:
                raise ValueError(f"invalid weight in {token!r}") from error
            if weight <= 0:
                raise ValueError("range weights must be positive")
            for hand_class in _expand_class(expression):
                if hand_class in expanded:
                    raise ValueError(
                        f"duplicate or overlapping hand class {hand_class!r}"
                    )
                expanded[hand_class] = weight
        ordered = sorted(expanded.items(), key=lambda item: _class_sort_key(item[0]))
        return cls(tuple(ordered))

    def to_weighted_range(self, dead_cards: Iterable[Card | str] = ()) -> WeightedRange:
        dead = set(parse_cards(dead_cards))
        combos = [
            combo
            for hand_class, weight in self.class_weights
            for combo in _class_combos(hand_class, weight)
            if not dead.intersection(combo.cards)
        ]
        if not combos:
            raise ValueError("every range combination is blocked")
        return WeightedRange(combos)

    def stats(self, dead_cards: Iterable[Card | str] = ()) -> RangeStats:
        raw = sum(
            len(_class_combos(hand_class, weight))
            for hand_class, weight in self.class_weights
        )
        legal = self.to_weighted_range(dead_cards)
        legal_count = len(legal.combos)
        return RangeStats(
            raw,
            legal_count,
            raw - legal_count,
            legal.total_weight,
            raw / 1326,
            legal_count / raw,
        )

    def matrix(self) -> tuple[tuple[float | None, ...], ...]:
        weights = dict(self.class_weights)
        rows: list[tuple[float | None, ...]] = []
        for row, first in enumerate(RANGE_RANKS):
            values: list[float | None] = []
            for column, second in enumerate(RANGE_RANKS):
                if row == column:
                    label = first + second
                elif row < column:
                    label = first + second + "s"
                else:
                    label = second + first + "o"
                values.append(weights.get(label))
            rows.append(tuple(values))
        return tuple(rows)


def _expand_class(expression: str) -> tuple[str, ...]:
    match = _CLASS_PATTERN.fullmatch(expression)
    if match is None:
        raise ValueError(f"invalid range class {expression!r}")
    first, second, suffix, plus = match.groups()
    first_index = RANGE_RANKS.index(first)
    second_index = RANGE_RANKS.index(second)
    if first == second:
        if suffix:
            raise ValueError("pairs cannot have suited/offsuit suffixes")
        if not plus:
            return (first + second,)
        return tuple(rank + rank for rank in RANGE_RANKS[: first_index + 1])
    if first_index >= second_index:
        raise ValueError("range classes must put the higher rank first")
    suffixes = ("s", "o") if suffix == "" else (suffix,)
    seconds = RANGE_RANKS[first_index + 1 : second_index + 1] if plus else (second,)
    return tuple(first + lower + suited for lower in seconds for suited in suffixes)


def _class_combos(hand_class: str, weight: float) -> tuple[WeightedCombo, ...]:
    first, second = hand_class[0], hand_class[1]
    if first == second:
        return tuple(
            WeightedCombo((Card(first, suit_a), Card(second, suit_b)), weight)
            for suit_a, suit_b in combinations("cdhs", 2)
        )
    suffix = hand_class[2]
    if suffix == "s":
        pairs = ((suit, suit) for suit in "cdhs")
    else:
        pairs = (
            (suit_a, suit_b)
            for suit_a in "cdhs"
            for suit_b in "cdhs"
            if suit_a != suit_b
        )
    return tuple(
        WeightedCombo((Card(first, suit_a), Card(second, suit_b)), weight)
        for suit_a, suit_b in pairs
    )


def _class_sort_key(hand_class: str) -> tuple[int, int, int]:
    first = RANGE_RANKS.index(hand_class[0])
    second = RANGE_RANKS.index(hand_class[1])
    suffix = 0 if len(hand_class) == 2 else (1 if hand_class[2] == "s" else 2)
    return first, second, suffix
