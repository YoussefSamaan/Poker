from __future__ import annotations

from dataclasses import dataclass
import math

from ..agents import StrategyProfile
from .schedule import Participant
from .simulator import SimulationConfig, SimulationRunner


@dataclass(frozen=True, slots=True)
class MatchupResult:
    participant_a: str
    participant_b: str
    physical_hands: int
    duplicate_blocks: int
    a_bb_per_100: float
    b_bb_per_100: float
    a_paired_bb_per_100: float
    paired_standard_error_bb_per_100: float
    a_paired_ci_95: tuple[float, float]
    b_paired_ci_95: tuple[float, float]


@dataclass(frozen=True, slots=True)
class CrossPlayResult:
    profiles: tuple[str, ...]
    matrix: tuple[tuple[float, ...], ...]
    confidence_intervals: tuple[tuple[tuple[float, float], ...], ...]
    matchups: tuple[MatchupResult, ...]


def run_balanced_matchup(
    first: Participant, second: Participant, *, hands: int, seed: int
) -> MatchupResult:
    if hands % 2:
        raise ValueError(
            "heads-up duplicate matchup requires an even physical hand count"
        )
    result = SimulationRunner(
        SimulationConfig(
            (first.profile, second.profile),
            hands=hands,
            master_seed=seed,
            duplicate_deals=True,
            participants=(first, second),
        )
    ).run()
    if any(
        sum(seat.net_chips for seat in record.seats) != 0
        for record in result.records
    ):
        raise AssertionError("heads-up matchup must be zero-sum on every hand")
    block_values = []
    for block in range(hands // 2):
        legs = tuple(
            record for record in result.records if record.duplicate_block_id == block
        )
        values = [
            seat.net_bb
            for record in legs
            for seat in record.seats
            if seat.participant_id == first.participant_id
        ]
        block_values.append(sum(values) / len(values))
    mean = sum(block_values) / len(block_values)
    variance = (
        sum((value - mean) ** 2 for value in block_values) / (len(block_values) - 1)
        if len(block_values) > 1
        else 0.0
    )
    se100 = math.sqrt(variance / len(block_values)) * 100
    a100 = mean * 100
    a_ci = (a100 - 1.96 * se100, a100 + 1.96 * se100)
    return MatchupResult(
        first.participant_id,
        second.participant_id,
        hands,
        len(block_values),
        a100,
        -a100,
        a100,
        se100,
        a_ci,
        (-a_ci[1], -a_ci[0]),
    )


def run_crossplay(
    profiles: tuple[StrategyProfile, ...],
    *,
    hands_per_matchup: int = 1_000,
    seed: int = 0,
) -> CrossPlayResult:
    participants = tuple(
        Participant(f"p{index}", profile.name, profile)
        for index, profile in enumerate(profiles)
    )
    count = len(profiles)
    matrix = [[0.0] * count for _ in range(count)]
    cis = [[(0.0, 0.0)] * count for _ in range(count)]
    matchups = []
    for row in range(count):
        for column in range(row + 1, count):
            matchup = run_balanced_matchup(
                participants[row],
                participants[column],
                hands=hands_per_matchup,
                seed=seed + row * count + column,
            )
            matchups.append(matchup)
            matrix[row][column], matrix[column][row] = (
                matchup.a_bb_per_100,
                matchup.b_bb_per_100,
            )
            cis[row][column], cis[column][row] = (
                matchup.a_paired_ci_95,
                matchup.b_paired_ci_95,
            )
    return CrossPlayResult(
        tuple(profile.name for profile in profiles),
        tuple(map(tuple, matrix)),
        tuple(tuple(row) for row in cis),
        tuple(matchups),
    )
