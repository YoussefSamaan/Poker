from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import math
from typing import Mapping

from ..cards import Card
from ..evaluation import HandRank, evaluate_holdem
from ..holdem import Action, BetTo, Call, Check, Fold, PlayerStatus, RaiseTo
from ..holdem.pots import PotLayer, build_side_pots
from ..multiway import (
    MultiwayEquityResult,
    ShowdownSampler,
    ShowdownWorld,
    summarize_equity,
)
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
SINGULAR_RANK_NAMES = {
    14: "ace",
    13: "king",
    12: "queen",
    11: "jack",
    10: "ten",
    9: "nine",
    8: "eight",
    7: "seven",
    6: "six",
    5: "five",
    4: "four",
    3: "three",
    2: "two",
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
    standard_error: float | None = None
    confidence_interval_95: tuple[float, float] | None = None


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
    standard_error: float | None = None


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
        return f"{SINGULAR_RANK_NAMES[rank.kickers[0]]}-high"
    if rank.category == 1:
        return f"one pair — {RANK_NAMES[rank.kickers[0]]}"
    if rank.category == 2:
        return f"two pair — {RANK_NAMES[rank.kickers[0]]} and {RANK_NAMES[rank.kickers[1]]}"
    if rank.category == 3:
        return f"three of a kind — {RANK_NAMES[rank.kickers[0]]}"
    if rank.category == 4:
        return f"straight — {SINGULAR_RANK_NAMES[rank.kickers[0]]}-high"
    if rank.category == 5:
        return f"flush — {SINGULAR_RANK_NAMES[rank.kickers[0]]}-high"
    if rank.category == 6:
        return f"full house — {RANK_NAMES[rank.kickers[0]]} over {RANK_NAMES[rank.kickers[1]]}"
    if rank.category == 7:
        return f"four of a kind — {RANK_NAMES[rank.kickers[0]]}"
    if rank.category == 8:
        return f"straight flush — {SINGULAR_RANK_NAMES[rank.kickers[0]]}-high"
    raise AssertionError(f"unknown hand category {rank.category}")


def analyze_showdown_baseline(
    game,
    hero_id: str,
    opponent_ranges: Mapping[str, WeightedRange | None],
    *,
    samples: int = 20_000,
    seed: int | None = 0,
    exact: bool | None = None,
) -> MultiwayDecisionAnalysis:
    context = decision_context(game, hero_id)
    ranges = tuple(opponent_ranges.get(opponent) for opponent in context.opponent_ids)
    sampler = ShowdownSampler(context.hero_cards, context.board, ranges)
    use_exact = (
        exact if exact is not None else sampler.estimated_exact_outcomes() <= 10_000
    )
    worlds = (
        tuple(sampler.exact_worlds())
        if use_exact
        else sampler.sample_worlds(samples, seed=seed)
    )
    method = "exact" if use_exact else "monte_carlo"
    equity = summarize_equity(worlds, method, seed)
    observation = game.observation_for(hero_id)
    contributions = {
        player.player_id: player.total_contribution for player in observation.players
    }
    called_contributions = dict(contributions)
    called_contributions[hero_id] += context.to_call
    for player in observation.players:
        if player.player_id == hero_id or player.status != PlayerStatus.ACTIVE:
            continue
        owed = max(0, observation.current_bet - player.street_contribution)
        called_contributions[player.player_id] += min(owed, player.stack)
    eligible = {
        player.player_id
        for player in observation.players
        if player.status not in (PlayerStatus.FOLDED, PlayerStatus.OUT)
    }
    pots = build_side_pots(called_contributions, eligible)
    relevant = tuple(layer for layer in pots if hero_id in layer.eligible_players)
    payouts = tuple(
        hero_payout_for_world(world, hero_id, context.opponent_ids, relevant)
        for world in worlds
    )
    expected_payout, payout_se, payout_interval = summarize_payout_samples(
        payouts, worlds, method
    )
    passive_ev = expected_payout - context.to_call
    ev_interval = (
        payout_interval[0] - context.to_call,
        payout_interval[1] - context.to_call,
    )
    passive_name = "check" if context.to_call == 0 else "call"
    actions = [BaselineAction("fold", 0.0, "Previously invested chips are sunk costs.")]
    actions.append(
        BaselineAction(
            passive_name,
            passive_ev,
            "After Hero's passive action, every remaining active player checks or "
            "calls the current wager up to its effective stack; nobody raises, "
            "and fixed ranges use one shared showdown world.",
            payout_se,
            ev_interval,
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
        f"Under the check/call-to-showdown baseline, {passive_name} EV is "
        f"{passive_ev:+.2f} chips (simulation SE {payout_se:.2f}; 95% interval "
        f"{ev_interval[0]:+.2f} to {ev_interval[1]:+.2f}).",
    )
    return MultiwayDecisionAnalysis(
        context, equity, tuple(actions), relevant, explanation
    )


def hero_payout_for_world(
    world: ShowdownWorld,
    hero_id: str,
    opponent_ids: tuple[str, ...],
    pots: tuple[PotLayer, ...],
) -> float:
    ranks = {hero_id: world.player_ranks[0]}
    ranks.update(zip(opponent_ids, world.player_ranks[1:]))
    payout = 0.0
    for layer in pots:
        eligible_ranks = {
            player_id: ranks[player_id]
            for player_id in layer.eligible_players
            if player_id in ranks
        }
        best = max(eligible_ranks.values())
        winners = tuple(
            player_id for player_id, rank in eligible_ranks.items() if rank == best
        )
        if hero_id in winners:
            payout += layer.amount / len(winners)
    return payout


def summarize_payout_samples(
    values: tuple[float, ...],
    worlds: tuple[ShowdownWorld, ...],
    method: str,
) -> tuple[float, float, tuple[float, float]]:
    total_weight = sum(world.weight for world in worlds)
    mean = (
        sum(value * world.weight for value, world in zip(values, worlds)) / total_weight
    )
    if method == "monte_carlo" and len(values) > 1:
        variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
        standard_error = math.sqrt(variance / len(values))
    else:
        standard_error = 0.0
    return (
        mean,
        standard_error,
        (
            mean - 1.96 * standard_error,
            mean + 1.96 * standard_error,
        ),
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
                passive.standard_error,
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
