from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Iterable

from ..agents import PersonalityAgent, StrategyProfile, position_name
from ..holdem import ActionType, HoldemGame, Street, TableConfig
from .metrics import StrategyMetrics, summarize_metrics
from .records import HandExperimentRecord, SeatHandStats


@dataclass(frozen=True, slots=True)
class SimulationConfig:
    profiles: tuple[StrategyProfile, ...]
    hands: int = 1_000
    stack_bb: int = 100
    small_blind: int = 1
    big_blind: int = 2
    master_seed: int = 0
    rotate_profiles: bool = True
    duplicate_deals: bool = False
    record_full_history: bool = False


@dataclass(frozen=True, slots=True)
class ExperimentResult:
    config: SimulationConfig
    records: tuple[HandExperimentRecord, ...]
    metrics: tuple[StrategyMetrics, ...]
    position_metrics: tuple[tuple[str, str, StrategyMetrics], ...]

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(asdict(self), indent=indent, sort_keys=True)

    def metrics_csv(self) -> str:
        fields = tuple(asdict(self.metrics[0])) if self.metrics else ()
        lines = [",".join(fields)]
        for metric in self.metrics:
            row = asdict(metric)
            lines.append(
                ",".join(
                    json.dumps(row[field], separators=(",", ":")) for field in fields
                )
            )
        return "\n".join(lines)

    def cumulative_net_bb(self) -> tuple[dict[str, float | int], ...]:
        totals = {metric.profile: 0.0 for metric in self.metrics}
        rows = []
        for record in self.records:
            for seat in record.seats:
                totals[seat.profile] += seat.net_bb
            rows.append({"hand": record.hand_index + 1, **totals})
        return tuple(rows)


class SimulationRunner:
    """Independent cash-hand evaluator with deterministic seed hierarchy."""

    def __init__(self, config: SimulationConfig) -> None:
        if not 2 <= len(config.profiles) <= 6 or config.hands < 1:
            raise ValueError("simulation requires 2–6 profiles and positive hands")
        self.config = config

    def run(self) -> ExperimentResult:
        records = tuple(self._run_hand(index) for index in range(self.config.hands))
        profiles = tuple(
            dict.fromkeys(profile.name for profile in self.config.profiles)
        )
        metrics = tuple(
            summarize_metrics(name, self._seats(records, name)) for name in profiles
        )
        position = []
        for name in profiles:
            positions = sorted({seat.position for seat in self._seats(records, name)})
            for label in positions:
                position.append(
                    (
                        name,
                        label,
                        summarize_metrics(
                            name,
                            (
                                seat
                                for seat in self._seats(records, name)
                                if seat.position == label
                            ),
                        ),
                    )
                )
        return ExperimentResult(self.config, records, metrics, tuple(position))

    def _run_hand(self, index: int) -> HandExperimentRecord:
        count = len(self.config.profiles)
        rotation = index % count if self.config.rotate_profiles else 0
        profiles = self.config.profiles[rotation:] + self.config.profiles[:rotation]
        button = index % count
        deal_index = index // 2 if self.config.duplicate_deals else index
        deal_seed = _seed(self.config.master_seed, "deck", deal_index)
        players = tuple(f"P{seat + 1}" for seat in range(count))
        stack = self.config.stack_bb * self.config.big_blind
        game = HoldemGame(
            TableConfig(
                players,
                (stack,) * count,
                self.config.small_blind,
                self.config.big_blind,
                button,
            ),
            seed=deal_seed,
        )
        agents = {
            player: PersonalityAgent(
                profiles[seat], _seed(self.config.master_seed, "policy", index, seat)
            )
            for seat, player in enumerate(players)
        }
        game.start_hand()
        while not game.is_terminal:
            actor = game.current_player
            if actor is None:
                raise AssertionError("non-terminal hand must have a current actor")
            observation = game.observation_for(actor)
            action = agents[actor].decide(observation, game.legal_actions(actor))
            game.step(action, actor)
        result = game.result
        if result is None:
            raise AssertionError("terminal hand must have a result")
        seats = tuple(
            _seat_stats(
                game, player, profiles[seat].name, seat, stack, self.config.big_blind
            )
            for seat, player in enumerate(players)
        )
        history = (
            tuple(
                {
                    **asdict(record),
                    "street": record.street.value,
                    "action_type": record.action_type.value,
                }
                for record in game.history
            )
            if self.config.record_full_history
            else None
        )
        return HandExperimentRecord(
            index,
            deal_seed,
            button,
            tuple((player, profiles[seat].name) for seat, player in enumerate(players)),
            seats,
            result.winners,
            result.showdown,
            len(game.history),
            history,
        )

    @staticmethod
    def _seats(
        records: tuple[HandExperimentRecord, ...], profile: str
    ) -> tuple[SeatHandStats, ...]:
        return tuple(
            seat
            for record in records
            for seat in record.seats
            if seat.profile == profile
        )


def sweep_parameter(
    base_profile: StrategyProfile,
    parameter: str,
    values: Iterable[float],
    opponents: tuple[StrategyProfile, ...],
    *,
    hands: int = 1_000,
    seed: int = 0,
) -> tuple[tuple[float, ExperimentResult], ...]:
    return tuple(
        (
            value,
            SimulationRunner(
                SimulationConfig(
                    (base_profile.with_parameter(parameter, value), *opponents),
                    hands=hands,
                    master_seed=seed,
                )
            ).run(),
        )
        for value in values
    )


def _seed(master: int, *parts: object) -> int:
    payload = ":".join(map(str, (master, *parts))).encode()
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big")


def _seat_stats(
    game: HoldemGame,
    player: str,
    profile: str,
    seat: int,
    starting: int,
    big_blind: int,
) -> SeatHandStats:
    observation = game.observation_for(player)
    records = tuple(record for record in game.history if record.player_id == player)
    preflop = tuple(record for record in records if record.street == Street.PREFLOP)
    voluntary = tuple(
        record
        for record in preflop
        if record.action_type
        not in {
            ActionType.SMALL_BLIND,
            ActionType.BIG_BLIND,
            ActionType.CHECK,
            ActionType.FOLD,
        }
    )
    raises_before = 0
    opportunities = three_bets = 0
    for record in game.history:
        if record.street != Street.PREFLOP:
            continue
        if record.player_id == player and raises_before == 1:
            opportunities = 1
            if record.action_type == ActionType.RAISE:
                three_bets += 1
        if record.action_type == ActionType.RAISE:
            raises_before += 1
    actions = tuple(
        record
        for record in records
        if record.action_type not in {ActionType.SMALL_BLIND, ActionType.BIG_BLIND}
    )
    post = tuple(record for record in actions if record.street != Street.PREFLOP)
    if game.result is None:
        raise AssertionError("metrics require a terminal hand")
    final_stack = game.result.final_stacks[player]
    return SeatHandStats(
        player,
        profile,
        seat,
        position_name(observation, player),
        final_stack - starting,
        (final_stack - starting) / big_blind,
        bool(voluntary),
        any(record.action_type == ActionType.RAISE for record in preflop),
        opportunities,
        three_bets,
        sum(record.action_type == ActionType.FOLD for record in actions),
        sum(record.action_type == ActionType.CALL for record in actions),
        sum(
            record.action_type in {ActionType.BET, ActionType.RAISE}
            for record in actions
        ),
        len(post),
        sum(
            record.action_type in {ActionType.BET, ActionType.RAISE} for record in post
        ),
        game.result.showdown,
    )
