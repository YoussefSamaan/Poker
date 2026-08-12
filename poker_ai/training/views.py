from __future__ import annotations

from dataclasses import dataclass

from ..cards import Card
from ..holdem import PlayerObservation, PlayerStatus
from ..ranges import RANGE_RANKS, PreflopRange
from .coach import SensitivityResult


@dataclass(frozen=True, slots=True)
class TableSeatView:
    player_id: str
    seat: int
    stack: int
    status: PlayerStatus
    street_contribution: int
    cards: tuple[Card, ...] | None


@dataclass(frozen=True, slots=True)
class PlayerTableView:
    hero_id: str
    board: tuple[Card, ...]
    street: str
    pot: int
    current_player: str | None
    button_player: str
    small_blind_player: str
    big_blind_player: str
    seats: tuple[TableSeatView, ...]


def player_table_view(observation: PlayerObservation) -> PlayerTableView:
    """Build the normal trainer view solely from leak-free observation state."""
    return PlayerTableView(
        observation.player_id,
        observation.board,
        observation.street.value,
        observation.pot,
        observation.current_player,
        observation.button_player,
        observation.small_blind_player,
        observation.big_blind_player,
        tuple(
            TableSeatView(
                player.player_id,
                player.seat,
                player.stack,
                player.status,
                player.street_contribution,
                observation.hole_cards
                if player.player_id == observation.player_id
                else None,
            )
            for player in observation.players
        ),
    )


def range_matrix_rows(value: PreflopRange) -> tuple[dict[str, str | float | None], ...]:
    matrix = value.matrix()
    return tuple(
        {
            "row": RANGE_RANKS[row],
            **{rank: matrix[row][column] for column, rank in enumerate(RANGE_RANKS)},
        }
        for row in range(13)
    )


def sensitivity_rows(
    values: tuple[SensitivityResult, ...],
) -> tuple[dict[str, str | int | float | None], ...]:
    return tuple(
        {
            "name": value.name,
            "legal combos": value.combo_count,
            "equity": value.equity,
            "required equity": value.required_equity,
            "equity edge": value.equity_edge,
            "passive EV": value.call_ev,
            "EV standard error": value.standard_error,
        }
        for value in values
    )
