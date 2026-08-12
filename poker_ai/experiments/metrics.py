from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

from .records import SeatHandStats


@dataclass(frozen=True, slots=True)
class StrategyMetrics:
    profile: str
    hands: int
    total_net_chips: int
    total_net_bb: float
    mean_bb_per_hand: float
    bb_per_100: float
    standard_deviation_bb: float
    standard_error_bb_per_100: float
    confidence_interval_95_bb_per_100: tuple[float, float]
    vpip: float
    pfr: float
    three_bet_frequency: float
    fold_frequency: float
    call_frequency: float
    bet_raise_frequency: float
    postflop_aggression_frequency: float
    showdown_rate: float


def summarize_metrics(profile: str, seats: Iterable[SeatHandStats]) -> StrategyMetrics:
    values = tuple(seats)
    if not values:
        raise ValueError("metrics require at least one hand")
    nets = [value.net_bb for value in values]
    hands = len(values)
    mean = sum(nets) / hands
    variance = (
        sum((value - mean) ** 2 for value in nets) / (hands - 1) if hands > 1 else 0.0
    )
    sd = math.sqrt(variance)
    se100 = sd / math.sqrt(hands) * 100
    actions = sum(value.folds + value.calls + value.bets_raises for value in values)
    opportunities = sum(value.three_bet_opportunities for value in values)
    postflop = sum(value.postflop_actions for value in values)
    bb100 = mean * 100
    return StrategyMetrics(
        profile,
        hands,
        sum(value.net_chips for value in values),
        sum(nets),
        mean,
        bb100,
        sd,
        se100,
        (bb100 - 1.96 * se100, bb100 + 1.96 * se100),
        sum(value.vpip for value in values) / hands,
        sum(value.pfr for value in values) / hands,
        sum(value.three_bets for value in values) / opportunities
        if opportunities
        else 0.0,
        sum(value.folds for value in values) / actions if actions else 0.0,
        sum(value.calls for value in values) / actions if actions else 0.0,
        sum(value.bets_raises for value in values) / actions if actions else 0.0,
        sum(value.postflop_bets_raises for value in values) / postflop
        if postflop
        else 0.0,
        sum(value.showdown for value in values) / hands,
    )
