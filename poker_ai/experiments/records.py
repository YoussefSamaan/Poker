from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class SeatHandStats:
    player_id: str
    profile: str
    participant_id: str
    seat: int
    position: str
    net_chips: int
    net_bb: float
    vpip: bool
    pfr: bool
    three_bet_opportunities: int
    three_bets: int
    folds: int
    checks: int
    calls: int
    bets_raises: int
    postflop_actions: int
    postflop_bets_raises: int
    player_reached_showdown: bool


@dataclass(frozen=True, slots=True)
class HandExperimentRecord:
    hand_index: int
    deal_seed: int
    button: int
    assignments: tuple[tuple[str, str], ...]
    participant_assignments: tuple[tuple[str, str], ...]
    duplicate_block_id: int | None
    duplicate_leg: int | None
    seats: tuple[SeatHandStats, ...]
    winners: tuple[str, ...]
    went_to_showdown: bool
    action_count: int
    history: tuple[dict[str, Any], ...] | None = None
