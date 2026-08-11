from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

from ..cards import Card
from ..equity import EquityResult
from ..holdem import (
    ActionRecord,
    BetTo,
    HoldemGame,
    LegalActions,
    PlayerStatus,
    RaiseTo,
    Street,
)
from ..ranges import WeightedRange
from ..scenario import HeadsUpScenario, ScenarioAnalysis, ScenarioAnalyzer


class HeadsUpAnalysisRequired(ValueError):
    """Raised when the heads-up baseline is requested for a multiway state."""


@dataclass(frozen=True, slots=True)
class DecisionContext:
    hero_id: str
    opponent_ids: tuple[str, ...]
    hero_cards: tuple[Card, Card]
    board: tuple[Card, ...]
    street: Street
    pot: int
    to_call: int
    hero_stack: int
    hero_street_contribution: int
    opponent_stacks: tuple[tuple[str, int], ...]
    legal_actions: LegalActions
    button_player: str
    small_blind_player: str
    big_blind_player: str
    history: tuple[ActionRecord, ...]

    @property
    def pot_after_call(self) -> int:
        return self.pot + self.to_call

    @property
    def required_equity(self) -> float:
        return self.to_call / self.pot_after_call if self.pot_after_call else 0.0


@dataclass(frozen=True, slots=True)
class CandidateSizing:
    label: str
    action: BetTo | RaiseTo
    target_to: int
    decision_cost: int
    pot_fraction: float | None


@dataclass(frozen=True, slots=True)
class CandidateValue:
    sizing: CandidateSizing
    ev: float
    regret: float
    assumptions: str


@dataclass(frozen=True, slots=True)
class BaselineDecisionAnalysis:
    context: DecisionContext
    equity: EquityResult
    scenario_analysis: ScenarioAnalysis
    candidate_values: tuple[CandidateValue, ...]
    equity_edge: float
    explanation: tuple[str, ...]
    model_label: str = "Simplified one-step baseline — not a GTO or solver result"


def decision_context(game: HoldemGame, hero_id: str) -> DecisionContext:
    if game.is_terminal:
        raise ValueError("cannot analyze a terminal hand")
    if hero_id != game.current_player:
        raise ValueError("analysis is available only for the current actor")
    observation = game.observation_for(hero_id)
    legal = observation.legal_actions
    if legal is None:
        raise ValueError("current actor has no legal action information")
    players = {player.player_id: player for player in observation.players}
    hero = players[hero_id]
    opponents = tuple(
        player.player_id
        for player in observation.players
        if player.player_id != hero_id
        and player.status not in (PlayerStatus.FOLDED, PlayerStatus.OUT)
    )
    if len(observation.hole_cards) != 2:
        raise ValueError("hero does not have a complete Hold'em hand")
    return DecisionContext(
        hero_id=hero_id,
        opponent_ids=opponents,
        hero_cards=(observation.hole_cards[0], observation.hole_cards[1]),
        board=observation.board,
        street=observation.street,
        pot=observation.pot,
        to_call=legal.call_amount or 0,
        hero_stack=hero.stack,
        hero_street_contribution=hero.street_contribution,
        opponent_stacks=tuple(
            (opponent, players[opponent].stack) for opponent in opponents
        ),
        legal_actions=legal,
        button_player=observation.button_player,
        small_blind_player=observation.small_blind_player,
        big_blind_player=observation.big_blind_player,
        history=observation.history,
    )


def candidate_sizings(
    context: DecisionContext,
    fractions: Iterable[float] = (0.33, 0.5, 0.75, 1.0, 1.5),
) -> tuple[CandidateSizing, ...]:
    legal = context.legal_actions
    is_bet = legal.can_bet
    minimum = legal.min_bet_to if is_bet else legal.min_raise_to
    maximum = legal.max_bet_to if is_bet else legal.max_raise_to
    if minimum is None or maximum is None:
        return ()

    candidates: list[tuple[str, int, float | None]] = [("minimum", minimum, None)]
    for fraction in fractions:
        if fraction <= 0:
            raise ValueError("candidate pot fractions must be positive")
        if is_bet:
            additional = max(1, math.floor(context.pot * fraction + 0.5))
            target = context.hero_street_contribution + additional
        else:
            raise_increment = max(
                1, math.floor(context.pot_after_call * fraction + 0.5)
            )
            target = (
                context.hero_street_contribution + context.to_call + raise_increment
            )
        candidates.append((f"{fraction:.0%} pot", target, fraction))
    effective_all_in = context.hero_street_contribution + context.hero_stack
    maximum_label = "all-in" if maximum == effective_all_in else "maximum effective"
    candidates.append((maximum_label, maximum, None))

    unique: dict[int, CandidateSizing] = {}
    for label, raw_target, candidate_fraction in candidates:
        target = min(max(raw_target, minimum), maximum)
        if target in unique:
            continue
        action = BetTo(target) if is_bet else RaiseTo(target)
        unique[target] = CandidateSizing(
            label,
            action,
            target,
            target - context.hero_street_contribution,
            candidate_fraction,
        )
    return tuple(unique.values())


def analyze_current_decision(
    game: HoldemGame,
    hero_id: str,
    *,
    opponent_range: WeightedRange | None = None,
    fold_equity: float = 0.0,
    samples: int = 20_000,
    seed: int | None = 0,
    sizings: Iterable[CandidateSizing] | None = None,
) -> BaselineDecisionAnalysis:
    context = decision_context(game, hero_id)
    if len(context.opponent_ids) != 1:
        raise HeadsUpAnalysisRequired(
            "the existing equity and simplified EV models require exactly one "
            f"non-folded opponent; this state has {len(context.opponent_ids)}"
        )
    opponent_id = context.opponent_ids[0]
    opponent_stack = dict(context.opponent_stacks)[opponent_id]
    scenario = HeadsUpScenario(
        hero=context.hero_cards,
        board=context.board,
        pot=context.pot,
        to_call=context.to_call,
        hero_stack=context.hero_stack,
        villain_stack=opponent_stack,
        opponent_range=opponent_range,
    )
    selected_sizings = tuple(sizings or candidate_sizings(context))
    raise_costs = tuple(
        sizing.decision_cost
        for sizing in selected_sizings
        if sizing.decision_cost > context.to_call
    )
    scenario_analysis = ScenarioAnalyzer().analyze(
        scenario,
        raise_costs=raise_costs,
        fold_equity=fold_equity,
        samples=samples,
        seed=seed,
    )
    best_ev = max(action.ev for action in scenario_analysis.actions)
    candidate_values: list[CandidateValue] = []
    for sizing in selected_sizings:
        key = f"raise_cost_{sizing.decision_cost:g}"
        try:
            value = next(
                action for action in scenario_analysis.actions if action.action == key
            )
        except StopIteration:
            continue
        candidate_values.append(
            CandidateValue(sizing, value.ev, best_ev - value.ev, value.assumptions)
        )

    equity = scenario_analysis.equity
    interval = equity.confidence_interval_95
    explanation = (
        f"Calling costs {context.to_call} to contest a pot of "
        f"{context.pot_after_call} after calling.",
        f"Required raw showdown equity is {context.required_equity:.1%}.",
        f"Estimated showdown equity is {equity.equity:.1%}; the 95% sampling "
        f"interval is {interval[0]:.1%} to {interval[1]:.1%}.",
        "EV values assume a fixed opponent range, no future betting, and the "
        f"supplied fold frequency of {fold_equity:.1%} for aggressive actions.",
    )
    return BaselineDecisionAnalysis(
        context=context,
        equity=equity,
        scenario_analysis=scenario_analysis,
        candidate_values=tuple(candidate_values),
        equity_edge=equity.equity - context.required_equity,
        explanation=explanation,
    )
