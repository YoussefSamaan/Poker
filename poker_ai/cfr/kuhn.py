from __future__ import annotations

from dataclasses import dataclass, field
import itertools
import random
from typing import Mapping

CARDS = ("J", "Q", "K")
ACTIONS = ("p", "b")


@dataclass(slots=True)
class InformationSet:
    regret_sum: list[float] = field(default_factory=lambda: [0.0, 0.0])
    strategy_sum: list[float] = field(default_factory=lambda: [0.0, 0.0])

    def strategy(self, reach_probability: float) -> tuple[float, float]:
        positive = [max(regret, 0.0) for regret in self.regret_sum]
        normalizer = sum(positive)
        current: tuple[float, float] = (
            (positive[0] / normalizer, positive[1] / normalizer)
            if normalizer
            else (0.5, 0.5)
        )
        for index, probability in enumerate(current):
            self.strategy_sum[index] += reach_probability * probability
        return current

    def average_strategy(self) -> tuple[float, float]:
        normalizer = sum(self.strategy_sum)
        if not normalizer:
            return (0.5, 0.5)
        return tuple(value / normalizer for value in self.strategy_sum)  # type: ignore[return-value]


@dataclass(frozen=True, slots=True)
class KuhnTrainingResult:
    strategy: Mapping[str, tuple[float, float]]
    training_value: float
    exact_value: float
    nash_conv: float
    iterations: int
    seed: int


def _terminal_utility(
    cards: tuple[str, str], history: str, player: int
) -> float | None:
    opponent = 1 - player
    higher = CARDS.index(cards[player]) > CARDS.index(cards[opponent])
    if history == "pp":
        return 1.0 if higher else -1.0
    if history in ("bp", "pbp"):
        return 1.0
    if history in ("bb", "pbb"):
        return 2.0 if higher else -2.0
    return None


class KuhnCFR:
    """Vanilla full-tree CFR for Kuhn poker, kept deliberately transparent."""

    def __init__(self) -> None:
        self.information_sets: dict[str, InformationSet] = {}

    def _cfr(
        self, cards: tuple[str, str], history: str, reach_0: float, reach_1: float
    ) -> float:
        player = len(history) % 2
        terminal = _terminal_utility(cards, history, player)
        if terminal is not None:
            return terminal

        key = cards[player] + history
        node = self.information_sets.setdefault(key, InformationSet())
        own_reach = reach_0 if player == 0 else reach_1
        opponent_reach = reach_1 if player == 0 else reach_0
        strategy = node.strategy(own_reach)
        action_utilities = [0.0, 0.0]
        node_utility = 0.0
        for action_index, action in enumerate(ACTIONS):
            if player == 0:
                child = self._cfr(
                    cards, history + action, reach_0 * strategy[action_index], reach_1
                )
            else:
                child = self._cfr(
                    cards, history + action, reach_0, reach_1 * strategy[action_index]
                )
            action_utilities[action_index] = -child
            node_utility += strategy[action_index] * action_utilities[action_index]
        for action_index in range(2):
            node.regret_sum[action_index] += opponent_reach * (
                action_utilities[action_index] - node_utility
            )
        return node_utility

    def train(self, iterations: int = 100_000, seed: int = 0) -> KuhnTrainingResult:
        if iterations < 1:
            raise ValueError("iterations must be positive")
        rng = random.Random(seed)
        value = 0.0
        deck = list(CARDS)
        for _ in range(iterations):
            rng.shuffle(deck)
            value += self._cfr((deck[0], deck[1]), "", 1.0, 1.0)
        strategy = {
            key: node.average_strategy()
            for key, node in sorted(self.information_sets.items())
        }
        return KuhnTrainingResult(
            strategy,
            value / iterations,
            expected_value(strategy),
            nash_conv(strategy),
            iterations,
            seed,
        )


def _payoff_player_zero(cards: tuple[str, str], history: str) -> float | None:
    player = len(history) % 2
    utility_for_player = _terminal_utility(cards, history, player)
    if utility_for_player is None:
        return None
    return utility_for_player if player == 0 else -utility_for_player


def expected_value(
    player_zero_strategy: Mapping[str, tuple[float, float]],
    player_one_strategy: Mapping[str, tuple[float, float]] | None = None,
) -> float:
    """Exactly evaluate two strategies as player-zero chips per hand."""

    if player_one_strategy is None:
        player_one_strategy = player_zero_strategy

    def walk(cards: tuple[str, str], history: str) -> float:
        terminal = _payoff_player_zero(cards, history)
        if terminal is not None:
            return terminal
        player = len(history) % 2
        policy = player_zero_strategy if player == 0 else player_one_strategy
        probabilities = policy.get(cards[player] + history, (0.5, 0.5))
        return sum(
            probabilities[index] * walk(cards, history + action)
            for index, action in enumerate(ACTIONS)
        )

    deals = tuple(itertools.permutations(CARDS, 2))
    return sum(walk(deal, "") for deal in deals) / len(deals)


def _pure_strategies(player: int):
    histories = ("", "pb") if player == 0 else ("p", "b")
    keys = tuple(card + history for card in CARDS for history in histories)
    for choices in itertools.product(range(2), repeat=len(keys)):
        yield {
            key: (1.0, 0.0) if action_index == 0 else (0.0, 1.0)
            for key, action_index in zip(keys, choices)
        }


def best_response_value(
    strategy: Mapping[str, tuple[float, float]], player: int
) -> float:
    """Return a player's exact best-response value without observing hidden cards."""

    if player == 0:
        return max(
            expected_value(response, strategy) for response in _pure_strategies(0)
        )
    if player == 1:
        return max(
            -expected_value(strategy, response) for response in _pure_strategies(1)
        )
    raise ValueError("player must be 0 or 1")


def nash_conv(strategy: Mapping[str, tuple[float, float]]) -> float:
    """Sum of both players' incentives to deviate; zero is a Nash equilibrium."""

    return best_response_value(strategy, 0) + best_response_value(strategy, 1)
