from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Iterable, Sequence

RANKS = "23456789TJQKA"
SUITS = "cdhs"
_SUIT_ALIASES = {
    "c": "c",
    "♣": "c",
    "d": "d",
    "♦": "d",
    "h": "h",
    "♥": "h",
    "s": "s",
    "♠": "s",
}


@dataclass(frozen=True, slots=True)
class Card:
    """A canonical playing card, rendered rank first (for example ``As``)."""

    rank: str
    suit: str

    def __post_init__(self) -> None:
        rank = self.rank.upper()
        suit = _SUIT_ALIASES.get(self.suit.lower(), self.suit.lower())
        if rank not in RANKS:
            raise ValueError(f"invalid rank {self.rank!r}; expected one of {RANKS}")
        if suit not in SUITS:
            raise ValueError(f"invalid suit {self.suit!r}; expected c, d, h, or s")
        object.__setattr__(self, "rank", rank)
        object.__setattr__(self, "suit", suit)

    @classmethod
    def parse(cls, text: str) -> Card:
        token = text.strip().replace("10", "T")
        if len(token) != 2:
            raise ValueError(f"invalid card {text!r}; use notation such as As or Q♦")
        return cls(token[0], token[1])

    @property
    def rank_value(self) -> int:
        return RANKS.index(self.rank) + 2

    def __str__(self) -> str:
        return f"{self.rank}{self.suit}"


def parse_cards(cards: str | Iterable[str | Card]) -> tuple[Card, ...]:
    """Parse whitespace/comma-separated text or an iterable of cards."""

    if isinstance(cards, str):
        values: Iterable[str | Card] = cards.replace(",", " ").split()
    else:
        values = cards
    parsed = tuple(card if isinstance(card, Card) else Card.parse(card) for card in values)
    require_unique(parsed)
    return parsed


def require_unique(cards: Sequence[Card]) -> None:
    if len(set(cards)) != len(cards):
        duplicates = sorted(str(card) for card in cards if cards.count(card) > 1)
        raise ValueError(f"duplicate cards are impossible: {', '.join(sorted(set(duplicates)))}")


def full_deck() -> tuple[Card, ...]:
    return tuple(Card(rank, suit) for suit in SUITS for rank in RANKS)


def shuffled_deck(seed: int | None = None) -> list[Card]:
    deck = list(full_deck())
    random.Random(seed).shuffle(deck)
    return deck
