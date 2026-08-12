from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from enum import Enum

from ..cards import Card
from ..evaluation import evaluate_holdem
from ..holdem import ActionType, LegalActions, PlayerObservation, Street
from ..training.coach import board_features


class HandBucket(Enum):
    AIR = "air"
    DRAW = "draw"
    WEAK_MADE = "weak_made"
    MEDIUM_MADE = "medium_made"
    STRONG_MADE = "strong_made"
    MONSTER = "monster"


@dataclass(frozen=True, slots=True)
class DecisionFeatures:
    player_id: str
    street: Street
    position: str
    active_players: int
    pot: int
    to_call: int
    pot_odds: float
    stack: int
    effective_stack: int
    street_contribution: int
    current_bet: int
    big_blind: int
    prior_raises: int
    previous_aggressor: bool
    spr: float | None
    hand_class: str
    pair: bool
    suited: bool
    rank_values: tuple[int, int]
    made_category: str | None
    bucket: HandBucket
    flush_draw: bool
    open_ended_straight_draw: bool
    board_paired: bool
    board_suit_texture: str


def position_name(observation: PlayerObservation, player_id: str) -> str:
    players = tuple(
        player.player_id
        for player in observation.players
        if player.stack > 0 or player.status.value != "out"
    )
    count = len(players)
    button = players.index(observation.button_player)
    seat = players.index(player_id)
    relative = (seat - button) % count
    if count == 2:
        return ("BTN/SB", "BB")[relative]
    tables = {
        3: ("BTN", "SB", "BB"),
        4: ("BTN", "SB", "BB", "CO"),
        5: ("BTN", "SB", "BB", "HJ", "CO"),
        6: ("BTN", "SB", "BB", "UTG", "HJ", "CO"),
    }
    return tables[count][relative]


def canonical_hand_class(cards: tuple[Card, ...]) -> str:
    if len(cards) != 2:
        return "unknown"
    first, second = sorted(cards, key=lambda card: card.rank_value, reverse=True)
    if first.rank == second.rank:
        return first.rank + second.rank
    return first.rank + second.rank + ("s" if first.suit == second.suit else "o")


def extract_features(
    observation: PlayerObservation, legal: LegalActions
) -> DecisionFeatures:
    player = next(
        item for item in observation.players if item.player_id == observation.player_id
    )
    opponents = tuple(
        item
        for item in observation.players
        if item.player_id != observation.player_id
        and item.status.value not in {"folded", "out"}
    )
    effective = min((item.stack for item in opponents), default=player.stack)
    to_call = legal.call_amount or 0
    street_history = tuple(
        record for record in observation.history if record.street == observation.street
    )
    aggressive = {ActionType.BET, ActionType.RAISE}
    prior_raises = sum(record.action_type in aggressive for record in street_history)
    big_blind = max(
        (
            record.amount_paid
            for record in observation.history
            if record.action_type == ActionType.BIG_BLIND
        ),
        default=max(1, observation.current_bet),
    )
    previous_aggressor = bool(
        street_history
        and street_history[-1].player_id == observation.player_id
        and street_history[-1].action_type in aggressive
    )
    hand_class = canonical_hand_class(observation.hole_cards)
    flush_draw = _flush_draw(observation.hole_cards + observation.board)
    straight_draw = _open_ended_draw(observation.hole_cards + observation.board)
    category = None
    if len(observation.board) >= 3:
        category = evaluate_holdem(observation.hole_cards + observation.board).name
    bucket = _bucket(category, flush_draw or straight_draw)
    texture = board_features(observation.board)
    ranks = tuple(
        sorted((card.rank_value for card in observation.hole_cards), reverse=True)
    )
    return DecisionFeatures(
        observation.player_id,
        observation.street,
        position_name(observation, observation.player_id),
        1 + len(opponents),
        observation.pot,
        to_call,
        to_call / (observation.pot + to_call) if observation.pot + to_call else 0.0,
        player.stack,
        min(player.stack, effective),
        player.street_contribution,
        observation.current_bet,
        big_blind,
        prior_raises,
        previous_aggressor,
        effective / observation.pot
        if observation.street != Street.PREFLOP and observation.pot
        else None,
        hand_class,
        len(ranks) == 2 and ranks[0] == ranks[1],
        len(observation.hole_cards) == 2
        and observation.hole_cards[0].suit == observation.hole_cards[1].suit,
        ranks if len(ranks) == 2 else (0, 0),
        category,
        bucket,
        flush_draw,
        straight_draw,
        texture.paired,
        texture.suit_texture,
    )


def _bucket(category: str | None, draw: bool) -> HandBucket:
    if category in {
        "straight",
        "flush",
        "full house",
        "four of a kind",
        "straight flush",
    }:
        return HandBucket.MONSTER
    if category in {"three of a kind", "two pair"}:
        return HandBucket.STRONG_MADE
    if category == "one pair":
        return HandBucket.MEDIUM_MADE
    if draw:
        return HandBucket.DRAW
    return HandBucket.AIR if category == "high card" else HandBucket.WEAK_MADE


def _flush_draw(cards: tuple[Card, ...]) -> bool:
    if len(cards) >= 7:
        return False
    return max(Counter(card.suit for card in cards).values(), default=0) == 4


def _open_ended_draw(cards: tuple[Card, ...]) -> bool:
    values = {card.rank_value for card in cards}
    if 14 in values:
        values.add(1)
    return any(
        len(values.intersection(range(start, start + 4))) == 4 for start in range(1, 11)
    )
