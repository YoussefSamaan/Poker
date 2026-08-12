from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from itertools import combinations, product
import math
import random
from typing import Iterable, Iterator

from .cards import Card, full_deck, parse_cards, require_unique
from .evaluation import HandRank, evaluate_holdem
from .ranges import WeightedRange

FULL_DECK = full_deck()


@dataclass(frozen=True, slots=True)
class ShowdownWorld:
    opponent_hands: tuple[tuple[Card, Card], ...]
    board: tuple[Card, ...]
    player_ranks: tuple[HandRank, ...]
    weight: float = 1.0

    @property
    def hero_share(self) -> float:
        best = max(self.player_ranks)
        if self.player_ranks[0] != best:
            return 0.0
        return 1.0 / sum(rank == best for rank in self.player_ranks)


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


class ShowdownSampler:
    """Generate card-consistent worlds from product-weighted opponent ranges.

    Small joint spaces are enumerated and sampled directly by product weight.
    Large spaces use independent weighted proposals and reject the entire tuple
    on collision. Rejection therefore conditions the product distribution on
    compatibility without seat-order renormalization.
    """

    def __init__(
        self,
        hero: Iterable[Card | str],
        board: Iterable[Card | str],
        opponent_ranges: Iterable[WeightedRange | None],
        *,
        joint_enumeration_threshold: int = 100_000,
    ) -> None:
        self.hero = parse_cards(hero)
        self.known_board = parse_cards(board)
        if len(self.hero) != 2 or len(self.known_board) > 5:
            raise ValueError("hero needs two cards and board at most five")
        require_unique(self.hero + self.known_board)
        supplied = tuple(opponent_ranges)
        if not 1 <= len(supplied) <= 5:
            raise ValueError("showdown sampling requires 1 to 5 opponents")
        dead = self.hero + self.known_board
        self.ranges = tuple(
            (value or WeightedRange.random(dead)).without_blocked(dead)
            for value in supplied
        )
        self._dead = frozenset(dead)
        self._joint_enumeration_threshold = joint_enumeration_threshold
        product_size = math.prod(len(value.combos) for value in self.ranges)
        self._valid_joints: (
            tuple[tuple[tuple[tuple[Card, Card], ...], float], ...] | None
        ) = None
        if product_size <= joint_enumeration_threshold:
            self._valid_joints = self._enumerate_valid_joints()
            if not self._valid_joints:
                raise ValueError("opponent ranges have no mutually compatible deal")
        elif not self._has_compatible_joint():
            raise ValueError("opponent ranges have no mutually compatible deal")

    def sample_worlds(
        self,
        samples: int,
        *,
        seed: int | None = 0,
        max_attempts_per_sample: int = 10_000,
    ) -> tuple[ShowdownWorld, ...]:
        if samples < 1:
            raise ValueError("samples must be positive")
        rng = random.Random(seed)
        worlds: list[ShowdownWorld] = []
        for _ in range(samples):
            hands = self.sample_joint_hands(rng, max_attempts=max_attempts_per_sample)
            used = self._dead.union(card for hand in hands for card in hand)
            runout = tuple(
                rng.sample(
                    [card for card in FULL_DECK if card not in used],
                    5 - len(self.known_board),
                )
            )
            worlds.append(self._world(hands, self.known_board + runout))
        return tuple(worlds)

    def sample_joint_hands(
        self, rng: random.Random, *, max_attempts: int = 10_000
    ) -> tuple[tuple[Card, Card], ...]:
        if self._valid_joints is not None:
            choices, weights = zip(*self._valid_joints)
            return rng.choices(choices, weights=weights, k=1)[0]
        for _ in range(max_attempts):
            hands = tuple(
                rng.choices(
                    value.combos,
                    weights=[combo.weight for combo in value.combos],
                    k=1,
                )[0].cards
                for value in self.ranges
            )
            flattened = tuple(card for hand in hands for card in hand)
            if len(flattened) == len(set(flattened)):
                return hands
        raise RuntimeError(
            "could not draw a compatible joint deal within the attempt limit; "
            "use narrower compatible ranges or exact joint enumeration"
        )

    def exact_worlds(self) -> Iterator[ShowdownWorld]:
        joints = self._valid_joints or self._enumerate_valid_joints()
        if not joints:
            raise ValueError("opponent ranges have no mutually compatible deal")
        for hands, joint_weight in joints:
            used = self._dead.union(card for hand in hands for card in hand)
            available = tuple(card for card in FULL_DECK if card not in used)
            for runout in combinations(available, 5 - len(self.known_board)):
                yield self._world(hands, self.known_board + runout, weight=joint_weight)

    def estimated_exact_outcomes(self) -> int:
        joint_count = (
            len(self._valid_joints)
            if self._valid_joints is not None
            else math.prod(len(value.combos) for value in self.ranges)
        )
        remaining = 52 - len(self._dead) - 2 * len(self.ranges)
        return joint_count * math.comb(remaining, 5 - len(self.known_board))

    def _enumerate_valid_joints(
        self,
    ) -> tuple[tuple[tuple[tuple[Card, Card], ...], float], ...]:
        valid = []
        for combos in product(*(value.combos for value in self.ranges)):
            cards = tuple(card for combo in combos for card in combo.cards)
            if len(cards) == len(set(cards)):
                valid.append(
                    (
                        tuple(combo.cards for combo in combos),
                        math.prod(combo.weight for combo in combos),
                    )
                )
        return tuple(valid)

    def _has_compatible_joint(self) -> bool:
        def visit(index: int, used: frozenset[Card]) -> bool:
            if index == len(self.ranges):
                return True
            return any(
                visit(index + 1, used.union(combo.cards))
                for combo in self.ranges[index].combos
                if not used.intersection(combo.cards)
            )

        return visit(0, self._dead)

    def _world(
        self,
        hands: tuple[tuple[Card, Card], ...],
        board: tuple[Card, ...],
        *,
        weight: float = 1.0,
    ) -> ShowdownWorld:
        ranks = (_cached_rank(self.hero + board),) + tuple(
            _cached_rank(hand + board) for hand in hands
        )
        return ShowdownWorld(hands, board, ranks, weight)


class MultiwayEquityCalculator:
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
        sampler = ShowdownSampler(hero, board, opponent_ranges)
        use_exact = (
            exact
            if exact is not None
            else sampler.estimated_exact_outcomes() <= self.exact_threshold
        )
        worlds = (
            tuple(sampler.exact_worlds())
            if use_exact
            else sampler.sample_worlds(samples, seed=seed)
        )
        return summarize_equity(worlds, "exact" if use_exact else "monte_carlo", seed)

    @staticmethod
    def sample_opponent_hands(
        ranges: tuple[WeightedRange, ...],
        dead_cards: Iterable[Card],
        rng: random.Random,
    ) -> tuple[tuple[Card, Card], ...]:
        """Backward-compatible product-conditioned joint-hand sampler."""
        dead = frozenset(dead_cards)
        valid: list[tuple[tuple[tuple[Card, Card], ...], float]] = []
        if math.prod(len(value.combos) for value in ranges) <= 100_000:
            for combos in product(*(value.combos for value in ranges)):
                cards = tuple(card for combo in combos for card in combo.cards)
                if not dead.intersection(cards) and len(cards) == len(set(cards)):
                    valid.append(
                        (
                            tuple(combo.cards for combo in combos),
                            math.prod(combo.weight for combo in combos),
                        )
                    )
            if not valid:
                raise ValueError("opponent ranges have no mutually compatible deal")
            choices, weights = zip(*valid)
            return rng.choices(choices, weights=weights, k=1)[0]
        for _ in range(10_000):
            hands = tuple(
                rng.choices(
                    value.combos,
                    weights=[combo.weight for combo in value.combos],
                    k=1,
                )[0].cards
                for value in ranges
            )
            cards = tuple(card for hand in hands for card in hand)
            if not dead.intersection(cards) and len(cards) == len(set(cards)):
                return hands
        raise RuntimeError("could not sample a compatible product-weighted joint deal")


def summarize_equity(
    worlds: Iterable[ShowdownWorld], method: str, seed: int | None
) -> MultiwayEquityResult:
    materialized = tuple(worlds)
    if not materialized:
        raise ValueError("at least one showdown world is required")
    total_weight = sum(world.weight for world in materialized)
    equity = (
        sum(world.hero_share * world.weight for world in materialized) / total_weight
    )
    wins = (
        sum(world.weight for world in materialized if world.hero_share == 1.0)
        / total_weight
    )
    ties = (
        sum(world.weight for world in materialized if 0 < world.hero_share < 1)
        / total_weight
    )
    if method == "monte_carlo" and len(materialized) > 1:
        variance = sum((world.hero_share - equity) ** 2 for world in materialized) / (
            len(materialized) - 1
        )
        standard_error = math.sqrt(variance / len(materialized))
    else:
        standard_error = 0.0
    interval = (
        max(0.0, equity - 1.96 * standard_error),
        min(1.0, equity + 1.96 * standard_error),
    )
    return MultiwayEquityResult(
        wins, ties, equity, standard_error, interval, len(materialized), method, seed
    )


@lru_cache(maxsize=200_000)
def _cached_rank(cards: tuple[Card, ...]) -> HandRank:
    return evaluate_holdem(cards)
