from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

from .cards import Card, parse_cards, require_unique
from .equity import EquityCalculator, EquityResult
from .ranges import WeightedRange


@dataclass(frozen=True, slots=True)
class HeadsUpScenario:
    """A heads-up, all-in-or-fold continuation model for offline study."""

    hero: tuple[Card, Card]
    board: tuple[Card, ...]
    pot: float
    to_call: float
    hero_stack: float
    villain_stack: float
    opponent_range: WeightedRange | None = None

    def __post_init__(self) -> None:
        if len(self.hero) != 2:
            raise ValueError("hero must have exactly two cards")
        if len(self.board) > 5:
            raise ValueError("board cannot exceed five cards")
        require_unique(self.hero + self.board)
        if min(self.pot, self.to_call, self.hero_stack, self.villain_stack) < 0:
            raise ValueError("pot, call, and stack values cannot be negative")
        if self.to_call > self.hero_stack:
            raise ValueError("to_call cannot exceed the hero's remaining stack")

    @classmethod
    def from_text(
        cls,
        *,
        hero: str,
        board: str = "",
        pot: float,
        to_call: float,
        hero_stack: float,
        villain_stack: float,
        opponent_range: WeightedRange | None = None,
    ) -> HeadsUpScenario:
        hero_cards = parse_cards(hero)
        if len(hero_cards) != 2:
            raise ValueError("hero must have exactly two cards")
        return cls(
            (hero_cards[0], hero_cards[1]),
            parse_cards(board),
            pot,
            to_call,
            hero_stack,
            villain_stack,
            opponent_range,
        )


@dataclass(frozen=True, slots=True)
class ActionValue:
    action: str
    ev: float
    assumptions: str


@dataclass(frozen=True, slots=True)
class ScenarioAnalysis:
    equity: EquityResult
    pot_odds: float
    actions: tuple[ActionValue, ...]
    recommended_action: str
    model: str

    def ev_for(self, action: str) -> float:
        for value in self.actions:
            if value.action == action:
                return value.ev
        raise KeyError(action)

    def regret(self, action: str) -> float:
        return max(value.ev for value in self.actions) - self.ev_for(action)


class ScenarioAnalyzer:
    def __init__(self, equity_calculator: EquityCalculator | None = None):
        self.equity_calculator = equity_calculator or EquityCalculator()

    def analyze(
        self,
        scenario: HeadsUpScenario,
        *,
        raise_costs: Iterable[float] = (),
        fold_equity: float | Mapping[float, float] = 0.0,
        samples: int = 20_000,
        seed: int | None = 0,
        exact: bool | None = None,
    ) -> ScenarioAnalysis:
        equity = self.equity_calculator.calculate(
            scenario.hero,
            scenario.board,
            scenario.opponent_range,
            samples=samples,
            seed=seed,
            exact=exact,
        )
        denominator = scenario.pot + scenario.to_call
        pot_odds = scenario.to_call / denominator if denominator else 0.0
        actions = [ActionValue("fold", 0.0, "Fold EV is the decision-point baseline.")]
        call_ev = equity.equity * denominator - scenario.to_call
        actions.append(
            ActionValue(
                "check" if scenario.to_call == 0 else "call",
                call_ev,
                "All remaining cards are dealt with no future betting; range is fixed.",
            )
        )

        for cost in raise_costs:
            if cost <= scenario.to_call:
                raise ValueError("each raise cost must exceed the call cost")
            if cost > scenario.hero_stack:
                raise ValueError("raise cost cannot exceed the hero stack")
            if cost > scenario.to_call + scenario.villain_stack:
                raise ValueError("raise cost cannot exceed the effective heads-up amount")
            opponent_call = min(cost - scenario.to_call, scenario.villain_stack)
            fe = fold_equity.get(cost, 0.0) if isinstance(fold_equity, Mapping) else fold_equity
            if not 0 <= fe <= 1:
                raise ValueError("fold equity must be between zero and one")
            final_pot = scenario.pot + cost + opponent_call
            called_ev = equity.equity * final_pot - cost
            raise_ev = fe * scenario.pot + (1 - fe) * called_ev
            actions.append(
                ActionValue(
                    f"raise_cost_{cost:g}",
                    raise_ev,
                    f"Villain folds {fe:.1%}; otherwise calls {opponent_call:g} and no future betting occurs.",
                )
            )

        best = max(actions, key=lambda action: action.ev)
        return ScenarioAnalysis(
            equity,
            pot_odds,
            tuple(actions),
            best.action,
            "heads-up fixed-range showdown EV (not a GTO solve)",
        )
