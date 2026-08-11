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
class EquityResult:
    win_probability: float
    tie_probability: float
    loss_probability: float
    equity: float
    standard_error: float
    confidence_interval_95: tuple[float, float]
    outcomes: int
    method: str
    seed: int | None


class EquityCalculator:
    """Heads-up Hold'em equity by exact enumeration or seeded Monte Carlo."""

    def __init__(self, exact_threshold: int = 250_000):
        if exact_threshold < 1:
            raise ValueError("exact_threshold must be positive")
        self.exact_threshold = exact_threshold

    def calculate(
        self,
        hero: Iterable[Card | str],
        board: Iterable[Card | str] = (),
        opponent_range: WeightedRange | None = None,
        *,
        samples: int = 20_000,
        seed: int | None = 0,
        exact: bool | None = None,
    ) -> EquityResult:
        hero_cards = parse_cards(hero)
        board_cards = parse_cards(board)
        if len(hero_cards) != 2:
            raise ValueError("hero must have exactly two hole cards")
        if len(board_cards) > 5:
            raise ValueError("the board cannot contain more than five cards")
        require_unique(hero_cards + board_cards)
        if samples < 1:
            raise ValueError("samples must be positive")

        dead = hero_cards + board_cards
        ranges = (opponent_range or WeightedRange.random(dead)).without_blocked(dead)
        cards_to_come = 5 - len(board_cards)
        possible_boards = math.comb(48 - len(board_cards), cards_to_come)
        outcome_count = len(ranges.combos) * possible_boards
        use_exact = exact if exact is not None else outcome_count <= self.exact_threshold

        if use_exact:
            return self._enumerate(hero_cards, board_cards, ranges, seed)
        return self._monte_carlo(hero_cards, board_cards, ranges, samples, seed)

    @staticmethod
    def _score(hero: tuple[Card, Card], opponent: tuple[Card, Card], board: tuple[Card, ...]) -> float:
        hero_rank = evaluate_holdem(hero + board)
        opponent_rank = evaluate_holdem(opponent + board)
        if hero_rank > opponent_rank:
            return 1.0
        if hero_rank == opponent_rank:
            return 0.5
        return 0.0

    def _enumerate(
        self,
        hero: tuple[Card, Card],
        board: tuple[Card, ...],
        ranges: WeightedRange,
        seed: int | None,
    ) -> EquityResult:
        wins = ties = losses = total_weight = 0.0
        outcomes = 0
        known = set(hero + board)
        needed = 5 - len(board)
        for combo in ranges.combos:
            remaining = [card for card in full_deck() if card not in known and card not in combo.cards]
            for runout in combinations(remaining, needed):
                score = self._score(hero, combo.cards, board + runout)
                outcomes += 1
                total_weight += combo.weight
                if score == 1.0:
                    wins += combo.weight
                elif score == 0.5:
                    ties += combo.weight
                else:
                    losses += combo.weight
        return self._result(wins, ties, losses, total_weight, outcomes, "exact", seed, ())

    def _monte_carlo(
        self,
        hero: tuple[Card, Card],
        board: tuple[Card, ...],
        ranges: WeightedRange,
        samples: int,
        seed: int | None,
    ) -> EquityResult:
        rng = random.Random(seed)
        combos = ranges.combos
        weights = [combo.weight for combo in combos]
        known = set(hero + board)
        needed = 5 - len(board)
        wins = ties = losses = 0.0
        scores: list[float] = []
        for _ in range(samples):
            combo = rng.choices(combos, weights=weights, k=1)[0]
            remaining = [card for card in full_deck() if card not in known and card not in combo.cards]
            runout = tuple(rng.sample(remaining, needed))
            score = self._score(hero, combo.cards, board + runout)
            scores.append(score)
            if score == 1.0:
                wins += 1
            elif score == 0.5:
                ties += 1
            else:
                losses += 1
        return self._result(wins, ties, losses, float(samples), samples, "monte_carlo", seed, scores)

    @staticmethod
    def _result(
        wins: float,
        ties: float,
        losses: float,
        total: float,
        outcomes: int,
        method: str,
        seed: int | None,
        scores: Iterable[float],
    ) -> EquityResult:
        win_p, tie_p, loss_p = wins / total, ties / total, losses / total
        equity = win_p + tie_p / 2
        score_values = tuple(scores)
        if len(score_values) > 1:
            variance = sum((score - equity) ** 2 for score in score_values) / (len(score_values) - 1)
            standard_error = math.sqrt(variance / len(score_values))
            interval = (max(0.0, equity - 1.96 * standard_error), min(1.0, equity + 1.96 * standard_error))
        else:
            standard_error = 0.0
            interval = (equity, equity)
        return EquityResult(
            win_p,
            tie_p,
            loss_p,
            equity,
            standard_error,
            interval,
            outcomes,
            method,
            seed,
        )
