from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Mapping

from ..cards import Card
from ..evaluation import HandRank, evaluate_holdem
from ..holdem import Action, BetTo, Call, Check, Fold, PlayerStatus, RaiseTo
from ..holdem.pots import PotLayer, build_side_pots
from ..multiway import MultiwayEquityCalculator, MultiwayEquityResult
from ..ranges import WeightedRange
from .analysis import DecisionContext, decision_context


RANK_NAMES = {
    14: "aces",
    13: "kings",
    12: "queens",
    11: "jacks",
    10: "tens",
    9: "nines",
    8: "eights",
    7: "sevens",
    6: "sixes",
    5: "fives",
    4: "fours",
    3: "threes",
    2: "twos",
}


@dataclass(frozen=True, slots=True)
class BoardFeatures:
    card_count: int
    paired: bool
    double_paired: bool
    suit_texture: str
    highest_rank: str | None
    maximum_adjacent_rank_gap: int | None


@dataclass(frozen=True, slots=True)
class BaselineAction:
    action: str
    ev: float | None
    assumptions: str


@dataclass(frozen=True, slots=True)
class MultiwayDecisionAnalysis:
    context: DecisionContext
    equity: MultiwayEquityResult
    actions: tuple[BaselineAction, ...]
    relevant_pots: tuple[PotLayer, ...]
    explanation: tuple[str, ...]
    model_label: str = "Simplified showdown baseline — not a GTO or solver result"

    @property
    def best_action(self) -> BaselineAction:
        supported = [action for action in self.actions if action.ev is not None]
        return max(
            supported,
            key=lambda action: action.ev if action.ev is not None else float("-inf"),
        )


@dataclass(frozen=True, slots=True)
class SensitivityResult:
    name: str
    combo_count: int
    equity: float
    required_equity: float
    equity_edge: float
    call_ev: float | None


@dataclass(frozen=True, slots=True)
class DecisionReview:
    timeline_position: int
    hero_id: str
    street: str
    board: tuple[str, ...]
    pot: int
    to_call: int
    legal_actions: tuple[str, ...]
    chosen_action: str
    selected_ranges: tuple[str, ...]
    best_baseline_action: str | None
    estimated_baseline_regret: float | None
    note: str


def board_features(board: tuple[Card, ...]) -> BoardFeatures:
    ranks = Counter(card.rank for card in board)
    suits = Counter(card.suit for card in board)
    if len(board) < 3:
        texture = "not applicable"
    elif max(suits.values()) == len(board):
        texture = "monotone"
    elif max(suits.values()) >= 2:
        texture = "two-tone"
    else:
        texture = "rainbow"
    values = sorted({card.rank_value for card in board}, reverse=True)
    gaps = [higher - lower for higher, lower in zip(values, values[1:])]
    return BoardFeatures(
        len(board),
        any(count >= 2 for count in ranks.values()),
        sum(count >= 2 for count in ranks.values()) >= 2,
        texture,
        max(board, key=lambda card: card.rank_value).rank if board else None,
        max(gaps) if gaps else None,
    )


def describe_current_hand(hero: tuple[Card, Card], board: tuple[Card, ...]) -> str:
    if len(board) < 3:
        return "preflop starting hand"
    rank = evaluate_holdem(hero + board)
    return _describe_rank(rank)


def _describe_rank(rank: HandRank) -> str:
    if rank.category == 0:
        return f"{RANK_NAMES[rank.kickers[0]][:-1]}-high"
    if rank.category == 1:
        return f"one pair — {RANK_NAMES[rank.kickers[0]]}"
    if rank.category == 2:
        return f"two pair — {RANK_NAMES[rank.kickers[0]]} and {RANK_NAMES[rank.kickers[1]]}"
    if rank.category == 3:
        return f"three of a kind — {RANK_NAMES[rank.kickers[0]]}"
    if rank.category == 6:
        return f"full house — {RANK_NAMES[rank.kickers[0]]} over {RANK_NAMES[rank.kickers[1]]}"
    if rank.category == 7:
        return f"four of a kind — {RANK_NAMES[rank.kickers[0]]}"
    return rank.name


def analyze_showdown_baseline(
    game,
    hero_id: str,
    opponent_ranges: Mapping[str, WeightedRange | None],
    *,
    samples: int = 20_000,
    seed: int | None = 0,
) -> MultiwayDecisionAnalysis:
    context = decision_context(game, hero_id)
    ranges = tuple(opponent_ranges.get(opponent) for opponent in context.opponent_ids)
    equity = MultiwayEquityCalculator().calculate(
        context.hero_cards, context.board, ranges, samples=samples, seed=seed
    )
    observation = game.observation_for(hero_id)
    contributions = {
        player.player_id: player.total_contribution for player in observation.players
    }
    hero_after = contributions[hero_id] + context.to_call
    called_contributions = dict(contributions)
    called_contributions[hero_id] = hero_after
    eligible = {
        player.player_id
        for player in observation.players
        if player.status not in (PlayerStatus.FOLDED, PlayerStatus.OUT)
    }
    pots = build_side_pots(called_contributions, eligible)
    relevant = tuple(layer for layer in pots if hero_id in layer.eligible_players)
    expected_payout = 0.0
    for layer in relevant:
        layer_opponents = tuple(
            opponent
            for opponent in context.opponent_ids
            if opponent in layer.eligible_players
        )
        if not layer_opponents:
            expected_payout += layer.amount
            continue
        if layer_opponents == context.opponent_ids:
            layer_equity = equity.equity
        else:
            layer_equity = (
                MultiwayEquityCalculator()
                .calculate(
                    context.hero_cards,
                    context.board,
                    tuple(
                        opponent_ranges.get(opponent) for opponent in layer_opponents
                    ),
                    samples=samples,
                    seed=seed,
                )
                .equity
            )
        expected_payout += layer.amount * layer_equity
    passive_ev = expected_payout - context.to_call
    passive_name = "check" if context.to_call == 0 else "call"
    actions = [BaselineAction("fold", 0.0, "Previously invested chips are sunk costs.")]
    actions.append(
        BaselineAction(
            passive_name,
            passive_ev,
            "All eligible hands run to showdown with fixed ranges and no future betting.",
        )
    )
    if context.legal_actions.can_bet or context.legal_actions.can_raise:
        actions.append(
            BaselineAction(
                "bet / raise",
                None,
                "Aggressive multiway EV is not modeled in Poker Coach v1.",
            )
        )
    interval = equity.confidence_interval_95
    explanation = (
        f"You are facing a call of {context.to_call} chips into a current pot of {context.pot}.",
        f"Required raw equity is {context.required_equity:.1%}.",
        f"Estimated expected pot share against {len(ranges)} opponent range(s) is {equity.equity:.1%}.",
        f"The Monte Carlo 95% sampling interval is {interval[0]:.1%}–{interval[1]:.1%}.",
        f"Under the simplified showdown baseline, {passive_name} EV is {passive_ev:+.2f} chips.",
    )
    return MultiwayDecisionAnalysis(
        context, equity, tuple(actions), relevant, explanation
    )


def compare_ranges(
    game,
    hero_id: str,
    named_ranges: Mapping[str, WeightedRange],
    *,
    samples: int = 10_000,
    seed: int = 0,
) -> tuple[SensitivityResult, ...]:
    context = decision_context(game, hero_id)
    if len(context.opponent_ids) != 1:
        raise ValueError("range sensitivity v1 compares heads-up ranges")
    results = []
    for offset, (name, opponent_range) in enumerate(named_ranges.items()):
        analysis = analyze_showdown_baseline(
            game,
            hero_id,
            {context.opponent_ids[0]: opponent_range},
            samples=samples,
            seed=seed + offset,
        )
        passive = next(
            action for action in analysis.actions if action.action in {"call", "check"}
        )
        results.append(
            SensitivityResult(
                name,
                len(opponent_range.combos),
                analysis.equity.equity,
                context.required_equity,
                analysis.equity.equity - context.required_equity,
                passive.ev,
            )
        )
    return tuple(results)


def capture_decision_review(
    timeline_position: int,
    analysis: MultiwayDecisionAnalysis,
    chosen_action: Action,
    selected_ranges: tuple[str, ...] = (),
) -> DecisionReview:
    labels = {
        Fold: "fold",
        Check: "check",
        Call: "call",
        BetTo: "bet / raise",
        RaiseTo: "bet / raise",
    }
    chosen = labels[type(chosen_action)]
    evaluated = next((item for item in analysis.actions if item.action == chosen), None)
    best = analysis.best_action
    if evaluated is None or evaluated.ev is None or best.ev is None:
        regret = None
    else:
        regret = best.ev - evaluated.ev
    note = "Estimated baseline regret within the simplified model."
    if regret is None:
        note = "Chosen action is not evaluated by this baseline."
    legal = analysis.context.legal_actions
    legal_names = tuple(
        name
        for name, enabled in (
            ("fold", legal.can_fold),
            ("check", legal.can_check),
            ("call", legal.can_call),
            ("bet", legal.can_bet),
            ("raise", legal.can_raise),
        )
        if enabled
    )
    return DecisionReview(
        timeline_position,
        analysis.context.hero_id,
        analysis.context.street.value,
        tuple(map(str, analysis.context.board)),
        analysis.context.pot,
        analysis.context.to_call,
        legal_names,
        chosen,
        selected_ranges,
        best.action,
        regret,
        note,
    )
