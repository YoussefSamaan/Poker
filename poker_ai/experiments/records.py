from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class SeatHandStats:
    player_id: str
    profile: str
    seat: int
    position: str
    net_chips: int
    net_bb: float
    vpip: bool
    pfr: bool
    three_bet_opportunities: int
    three_bets: int
    folds: int
    calls: int
    bets_raises: int
    postflop_actions: int
    postflop_bets_raises: int
    showdown: bool


@dataclass(frozen=True, slots=True)
class HandExperimentRecord:
    hand_index: int
    deal_seed: int
    button: int
    assignments: tuple[tuple[str, str], ...]
    seats: tuple[SeatHandStats, ...]
    winners: tuple[str, ...]
    went_to_showdown: bool
    action_count: int
    history: tuple[dict[str, Any], ...] | None = None
