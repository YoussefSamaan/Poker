from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from itertools import combinations
from typing import Iterable

from .cards import Card, parse_cards

CATEGORY_NAMES = (
    "high card",
    "one pair",
    "two pair",
    "three of a kind",
    "straight",
    "flush",
    "full house",
    "four of a kind",
    "straight flush",
)


@dataclass(frozen=True, order=True, slots=True)
class HandRank:
    """Comparable hand strength. Larger values are better."""

    category: int
    kickers: tuple[int, ...]
    best_five: tuple[Card, ...] = field(default=(), compare=False)

    @property
    def name(self) -> str:
        return CATEGORY_NAMES[self.category]


def _straight_high(values: Iterable[int]) -> int | None:
    unique = set(values)
    if 14 in unique:
        unique.add(1)
    ordered = sorted(unique)
    run = 1
    best = None
    for previous, current in zip(ordered, ordered[1:]):
        if current == previous + 1:
            run += 1
            if run >= 5:
                best = current
        else:
            run = 1
    return best


def evaluate_five(cards: Iterable[Card | str]) -> HandRank:
    hand = parse_cards(cards)
    if len(hand) != 5:
        raise ValueError(
            f"five-card evaluation requires exactly 5 cards, got {len(hand)}"
        )

    values = [card.rank_value for card in hand]
    counts = Counter(values)
    groups = sorted(((count, value) for value, count in counts.items()), reverse=True)
    flush = len({card.suit for card in hand}) == 1
    straight_high = _straight_high(values)
    key: tuple[int, tuple[int, ...]]

    if flush and straight_high is not None:
        key = (8, (straight_high,))
    elif groups[0][0] == 4:
        quad = groups[0][1]
        key = (7, (quad, max(value for value in values if value != quad)))
    elif sorted(counts.values()) == [2, 3]:
        trip = max(value for value, count in counts.items() if count == 3)
        pair = max(value for value, count in counts.items() if count == 2)
        key = (6, (trip, pair))
    elif flush:
        key = (5, tuple(sorted(values, reverse=True)))
    elif straight_high is not None:
        key = (4, (straight_high,))
    elif groups[0][0] == 3:
        trip = groups[0][1]
        kickers = sorted((value for value in values if value != trip), reverse=True)
        key = (3, (trip, *kickers))
    elif sorted(counts.values()) == [1, 2, 2]:
        pairs = sorted(
            (value for value, count in counts.items() if count == 2), reverse=True
        )
        kicker = max(value for value, count in counts.items() if count == 1)
        key = (2, (pairs[0], pairs[1], kicker))
    elif 2 in counts.values():
        pair = max(value for value, count in counts.items() if count == 2)
        kickers = sorted((value for value in values if value != pair), reverse=True)
        key = (1, (pair, *kickers))
    else:
        key = (0, tuple(sorted(values, reverse=True)))

    return HandRank(key[0], key[1], tuple(hand))


def evaluate_holdem(cards: Iterable[Card | str]) -> HandRank:
    """Return the best five-card hand from 5, 6, or 7 cards."""

    available = parse_cards(cards)
    if not 5 <= len(available) <= 7:
        raise ValueError(
            f"Hold'em evaluation requires 5 to 7 cards, got {len(available)}"
        )
    return max(evaluate_five(combo) for combo in combinations(available, 5))
