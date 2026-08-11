from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True, slots=True)
class PotLayer:
    amount: int
    contributors: tuple[str, ...]
    eligible_players: tuple[str, ...]


def build_side_pots(
    contributions: Mapping[str, int], eligible_players: set[str]
) -> tuple[PotLayer, ...]:
    """Derive main/side pots from total contributions, including folded chips."""

    levels = sorted({amount for amount in contributions.values() if amount > 0})
    previous = 0
    pots: list[PotLayer] = []
    for level in levels:
        contributors = tuple(
            player for player, amount in contributions.items() if amount >= level
        )
        amount = (level - previous) * len(contributors)
        eligible = tuple(
            player for player in contributors if player in eligible_players
        )
        if amount:
            pots.append(PotLayer(amount, contributors, eligible))
        previous = level
    return tuple(pots)
