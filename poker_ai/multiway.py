from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
import math
import random
from typing import Iterable

from .cards import Card, full_deck, parse_cards, require_unique
from .evaluation import evaluate_holdem
from .ranges import WeightedRange


@dataclass(frozen=True, slots=True)
class MultiwayEquityResult:
    win_probability: float
    tie_probability: float
    equity: float
    standard_error: float
    confidence_interval_95: tuple[float, float]
    outcomes: int
    method: str
    seed: int | None


class MultiwayEquityCalculator:
    """Hero equity against 1–5 ranges with card-disjoint conditional sampling."""

    def __init__(self, exact_threshold: int = 10_000) -> None:
        if exact_threshold < 1:
            raise ValueError("exact_threshold must be positive")
        self.exact_threshold = exact_threshold

    def calculate(
        self,
        hero: Iterable[Card | str],
        board: Iterable[Card | str],
        opponent_ranges: Iterable[WeightedRange | None],
        *,
        samples: int = 20_000,
        seed: int | None = 0,
        exact: bool | None = None,
    ) -> MultiwayEquityResult:
        hero_cards = parse_cards(hero)
        board_cards = parse_cards(board)
        if len(hero_cards) != 2 or len(board_cards) > 5:
            raise ValueError("hero needs two cards and board at most five")
        require_unique(hero_cards + board_cards)
        ranges = tuple(opponent_ranges)
        if not 1 <= len(ranges) <= 5:
            raise ValueError("multiway equity requires 1 to 5 opponents")
        dead = hero_cards + board_cards
        legal_ranges = tuple(
            (value or WeightedRange.random(dead)).without_blocked(dead)
            for value in ranges
        )
        cards_to_come = 5 - len(board_cards)
        estimate = math.comb(50 - len(board_cards) - 2 * len(ranges), cards_to_come)
        for value in legal_ranges:
            estimate *= len(value.combos)
            if estimate > self.exact_threshold:
                break
        use_exact = exact if exact is not None else estimate <= self.exact_threshold
        if use_exact:
            observations = self._enumerate(hero_cards, board_cards, legal_ranges)
            return _summarize(observations, "exact", seed)
        if samples < 1:
            raise ValueError("samples must be positive")
        rng = random.Random(seed)
        shares: list[tuple[float, float]] = []
        for _ in range(samples):
            hands = self.sample_opponent_hands(legal_ranges, dead, rng)
            used = set(dead)
            for hand in hands:
                used.update(hand)
            runout = tuple(
                rng.sample(
                    [card for card in full_deck() if card not in used], cards_to_come
                )
            )
            shares.append((_hero_share(hero_cards, hands, board_cards + runout), 1.0))
        return _summarize(shares, "monte_carlo", seed)

    @staticmethod
    def sample_opponent_hands(
        ranges: tuple[WeightedRange, ...],
        dead_cards: Iterable[Card],
        rng: random.Random,
    ) -> tuple[tuple[Card, Card], ...]:
        """Sequentially sample from each range conditioned on all used cards."""
        used = set(dead_cards)
        selected: list[tuple[Card, Card]] = []
        for opponent_range in ranges:
            legal = [
                combo
                for combo in opponent_range.combos
                if not used.intersection(combo.cards)
            ]
            if not legal:
                raise ValueError("opponent ranges have no mutually compatible deal")
            combo = rng.choices(legal, weights=[item.weight for item in legal], k=1)[0]
            selected.append(combo.cards)
            used.update(combo.cards)
        return tuple(selected)

    def _enumerate(
        self,
        hero: tuple[Card, ...],
        board: tuple[Card, ...],
        ranges: tuple[WeightedRange, ...],
    ) -> list[tuple[float, float]]:
        weighted_shares: list[tuple[float, float]] = []

        def visit(
            index: int,
            used: set[Card],
            hands: tuple[tuple[Card, Card], ...],
            weight: float,
        ) -> None:
            if index < len(ranges):
                for combo in ranges[index].combos:
                    if not used.intersection(combo.cards):
                        visit(
                            index + 1,
                            used.union(combo.cards),
                            hands + (combo.cards,),
                            weight * combo.weight,
                        )
                return
            available = [card for card in full_deck() if card not in used]
            for runout in combinations(available, 5 - len(board)):
                share = _hero_share(hero, hands, board + runout)
                weighted_shares.append((share, weight))

        visit(0, set(hero + board), (), 1.0)
        if not weighted_shares:
            raise ValueError("opponent ranges have no mutually compatible deal")
        return weighted_shares


def _hero_share(
    hero: tuple[Card, ...],
    opponents: tuple[tuple[Card, Card], ...],
    board: tuple[Card, ...],
) -> float:
    ranks = [evaluate_holdem(hero + board)] + [
        evaluate_holdem(hand + board) for hand in opponents
    ]
    best = max(ranks)
    if ranks[0] != best:
        return 0.0
    return 1.0 / sum(rank == best for rank in ranks)


def _summarize(
    observations: list[tuple[float, float]], method: str, seed: int | None
) -> MultiwayEquityResult:
    outcomes = len(observations)
    total_weight = sum(weight for _, weight in observations)
    equity = sum(share * weight for share, weight in observations) / total_weight
    wins = sum(weight for share, weight in observations if share == 1.0) / total_weight
    ties = (
        sum(weight for share, weight in observations if 0.0 < share < 1.0)
        / total_weight
    )
    if method == "monte_carlo" and outcomes > 1:
        variance = sum((share - equity) ** 2 for share, _ in observations) / (
            outcomes - 1
        )
        standard_error = math.sqrt(variance / outcomes)
    else:
        standard_error = 0.0
    interval = (
        max(0.0, equity - 1.96 * standard_error),
        min(1.0, equity + 1.96 * standard_error),
    )
    return MultiwayEquityResult(
        wins, ties, equity, standard_error, interval, outcomes, method, seed
    )
