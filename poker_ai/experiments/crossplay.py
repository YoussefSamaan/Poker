from __future__ import annotations

from dataclasses import dataclass

from ..agents import StrategyProfile
from .simulator import SimulationConfig, SimulationRunner


@dataclass(frozen=True, slots=True)
class CrossPlayResult:
    profiles: tuple[str, ...]
    matrix: tuple[tuple[float, ...], ...]
    confidence_intervals: tuple[tuple[tuple[float, float], ...], ...]


def run_crossplay(
    profiles: tuple[StrategyProfile, ...],
    *,
    hands_per_matchup: int = 1_000,
    seed: int = 0,
) -> CrossPlayResult:
    matrix = []
    intervals = []
    for row, first in enumerate(profiles):
        values = []
        cis = []
        for column, second in enumerate(profiles):
            result = SimulationRunner(
                SimulationConfig(
                    (first, second),
                    hands_per_matchup,
                    master_seed=seed + row * len(profiles) + column,
                    duplicate_deals=True,
                )
            ).run()
            metric = next(item for item in result.metrics if item.profile == first.name)
            values.append(metric.bb_per_100)
            cis.append(metric.confidence_interval_95_bb_per_100)
        matrix.append(tuple(values))
        intervals.append(tuple(cis))
    return CrossPlayResult(
        tuple(profile.name for profile in profiles), tuple(matrix), tuple(intervals)
    )
