from __future__ import annotations

import copy
from dataclasses import dataclass
from enum import Enum
import json
from typing import Any, Mapping
import uuid

from ..cards import Card, parse_cards
from ..holdem import (
    Action,
    BetTo,
    Call,
    Check,
    CheckCallPolicy,
    Fold,
    HoldemGame,
    LegalActions,
    Policy,
    RaiseTo,
    RandomLegalPolicy,
    ScenarioBuilder,
    TableConfig,
    Transition,
)

SCHEMA_VERSION = 1


class SeatControl(Enum):
    HUMAN = "human"
    POLICY = "policy"


class PolicyKind(Enum):
    CHECK_CALL = "check_call"
    RANDOM_LEGAL = "random_legal"
    NIT = "nit"
    TAG = "tag"
    LAG = "lag"
    CALLING_STATION = "calling_station"
    MANIAC = "maniac"
    BLUFF_HEAVY = "bluff_heavy"


@dataclass(frozen=True, slots=True)
class PolicyConfig:
    kind: PolicyKind
    seed: int | None = None

    def build(self) -> Policy:
        if self.kind == PolicyKind.CHECK_CALL:
            return CheckCallPolicy()
        if self.kind == PolicyKind.RANDOM_LEGAL:
            return RandomLegalPolicy(self.seed)
        if self.kind.value in {
            "nit",
            "tag",
            "lag",
            "calling_station",
            "maniac",
            "bluff_heavy",
        }:
            from ..agents import PRESETS, PersonalityAgent

            return PersonalityAgent(PRESETS[self.kind.value], self.seed)
        raise ValueError(f"unsupported policy kind {self.kind}")


@dataclass(frozen=True, slots=True)
class TimelineAction:
    player_id: str
    action: Action


@dataclass(frozen=True, slots=True)
class PolicyStep:
    player_id: str
    action: Action
    transition: Transition


def action_to_dict(action: Action) -> dict[str, Any]:
    if isinstance(action, Fold):
        return {"type": "fold"}
    if isinstance(action, Check):
        return {"type": "check"}
    if isinstance(action, Call):
        return {"type": "call"}
    if isinstance(action, BetTo):
        return {"type": "bet_to", "amount": action.amount}
    if isinstance(action, RaiseTo):
        return {"type": "raise_to", "amount": action.amount}
    raise TypeError(f"cannot serialize action {type(action).__name__}")


def action_from_dict(data: Mapping[str, Any]) -> Action:
    action_type = data.get("type")
    if action_type in {"fold", "check", "call"}:
        if set(data) != {"type"}:
            raise ValueError(f"{action_type} action cannot contain extra fields")
        return {"fold": Fold, "check": Check, "call": Call}[action_type]()
    if action_type in {"bet_to", "raise_to"}:
        if set(data) != {"type", "amount"}:
            raise ValueError(f"{action_type} requires only type and amount")
        amount = data.get("amount")
        if isinstance(amount, bool) or not isinstance(amount, int):
            raise ValueError("action amount must be an integer")
        return BetTo(amount) if action_type == "bet_to" else RaiseTo(amount)
    raise ValueError(f"unknown action type {action_type!r}")


class TrainingSession:
    """Replay-based application layer over the authoritative Hold'em engine."""

    def __init__(
        self,
        config: TableConfig,
        *,
        seed: int | None = None,
        preset_deck: tuple[Card, ...] | None = None,
        controls: Mapping[str, SeatControl] | None = None,
        policy_configs: Mapping[str, PolicyConfig] | None = None,
        actions: tuple[TimelineAction, ...] = (),
        position: int | None = None,
        scenario_metadata: Mapping[str, Any] | None = None,
        session_id: str | None = None,
    ) -> None:
        if seed is not None and preset_deck is not None:
            raise ValueError("use either seed or preset_deck, not both")
        self.config = config
        self.seed = seed
        self.preset_deck = tuple(preset_deck) if preset_deck is not None else None
        self.controls = {
            player_id: (controls or {}).get(player_id, SeatControl.HUMAN)
            for player_id in config.player_ids
        }
        self.policy_configs = dict(policy_configs or {})
        self.scenario_metadata = copy.deepcopy(dict(scenario_metadata or {}))
        self.session_id = session_id or str(uuid.uuid4())
        self.actions: list[TimelineAction] = list(actions)
        self.position = len(self.actions) if position is None else position
        if not 0 <= self.position <= len(self.actions):
            raise ValueError("timeline position is outside the action history")
        self._custom_policy_templates: dict[str, Policy] = {}
        self._policies: dict[str, Policy] = {}
        self.last_policy_trace: Any | None = None
        for player_id, policy_config in self.policy_configs.items():
            self._require_player(player_id)
            self.controls[player_id] = SeatControl.POLICY
            if not isinstance(policy_config, PolicyConfig):
                raise TypeError("policy_configs values must be PolicyConfig objects")
        self.game: HoldemGame
        self._rebuild()

    @classmethod
    def new_hand(
        cls,
        config: TableConfig,
        *,
        seed: int = 0,
        human_players: set[str] | None = None,
        policy_configs: Mapping[str, PolicyConfig] | None = None,
    ) -> TrainingSession:
        humans = set(config.player_ids) if human_players is None else set(human_players)
        unknown = humans.difference(config.player_ids)
        if unknown:
            raise KeyError(f"unknown human players: {sorted(unknown)}")
        controls = {
            player_id: SeatControl.HUMAN if player_id in humans else SeatControl.POLICY
            for player_id in config.player_ids
        }
        configs = dict(policy_configs or {})
        for player_id in config.player_ids:
            if controls[player_id] == SeatControl.POLICY and player_id not in configs:
                configs[player_id] = PolicyConfig(PolicyKind.CHECK_CALL)
        return cls(config, seed=seed, controls=controls, policy_configs=configs)

    @classmethod
    def from_scenario(
        cls,
        builder: ScenarioBuilder,
        *,
        human_players: set[str] | None = None,
        policy_configs: Mapping[str, PolicyConfig] | None = None,
    ) -> TrainingSession:
        built = builder.build()
        humans = (
            set(builder.config.player_ids)
            if human_players is None
            else set(human_players)
        )
        unknown = humans.difference(builder.config.player_ids)
        if unknown:
            raise KeyError(f"unknown human players: {sorted(unknown)}")
        controls = {
            player_id: SeatControl.HUMAN if player_id in humans else SeatControl.POLICY
            for player_id in builder.config.player_ids
        }
        configs = dict(policy_configs or {})
        for player_id in builder.config.player_ids:
            if controls[player_id] == SeatControl.POLICY and player_id not in configs:
                configs[player_id] = PolicyConfig(PolicyKind.CHECK_CALL)
        actions = tuple(
            TimelineAction(player_id, action)
            for player_id, action in builder.scripted_actions
        )
        metadata = {
            "known_hole_cards": {
                player_id: [str(card) for card in cards]
                for player_id, cards in builder.known_hole_cards.items()
            },
            "board_runout": [str(card) for card in builder.board_runout],
        }
        return cls(
            builder.config,
            preset_deck=built.privileged_replay_deck,
            controls=controls,
            policy_configs=configs,
            actions=actions,
            scenario_metadata=metadata,
        )

    @property
    def current_actor(self) -> str | None:
        return self.game.current_player

    @property
    def timeline(self) -> tuple[TimelineAction, ...]:
        """The immutable recorded action line, including any redo future."""
        return tuple(self.actions)

    @property
    def needs_human_action(self) -> bool:
        actor = self.current_actor
        return (
            actor is not None
            and not self.game.is_terminal
            and self.controls[actor] == SeatControl.HUMAN
        )

    @property
    def can_undo(self) -> bool:
        return self.position > 0

    @property
    def can_redo(self) -> bool:
        return self.position < len(self.actions)

    def available_actions(self) -> LegalActions | None:
        if self.game.is_terminal or self.current_actor is None:
            return None
        return self.game.legal_actions(self.current_actor)

    def act(self, action: Action) -> Transition:
        if self.game.is_terminal or self.current_actor is None:
            raise RuntimeError("the training hand is terminal")
        if self.position < len(self.actions):
            del self.actions[self.position :]
        player_id = self.current_actor
        transition = self.game.step(action, player_id)
        self.actions.append(TimelineAction(player_id, action))
        self.position += 1
        return transition

    def undo(self) -> None:
        if not self.can_undo:
            raise IndexError("already at the beginning of the hand")
        self.goto_action(self.position - 1)

    def redo(self) -> None:
        if not self.can_redo:
            raise IndexError("already at the end of the recorded timeline")
        self.goto_action(self.position + 1)

    def goto_action(self, position: int) -> None:
        if not 0 <= position <= len(self.actions):
            raise IndexError("timeline position is outside the recorded actions")
        self.position = position
        self._rebuild()

    def branch(self, at_action: int | None = None) -> TrainingSession:
        position = self.position if at_action is None else at_action
        if not 0 <= position <= len(self.actions):
            raise IndexError("branch point is outside the recorded actions")
        branch = TrainingSession(
            self.config,
            seed=self.seed,
            preset_deck=self.preset_deck,
            controls=self.controls,
            policy_configs=self.policy_configs,
            actions=tuple(self.actions[:position]),
            position=position,
            scenario_metadata=self.scenario_metadata,
            session_id=self.session_id,
        )
        for player_id, policy in self._custom_policy_templates.items():
            branch.set_policy(player_id, copy.deepcopy(policy))
        branch.goto_action(position)
        return branch

    def set_human(self, player_id: str) -> None:
        self._require_player(player_id)
        self.controls[player_id] = SeatControl.HUMAN
        self._custom_policy_templates.pop(player_id, None)
        self.policy_configs.pop(player_id, None)
        self._policies.pop(player_id, None)

    def set_policy(
        self,
        player_id: str,
        policy: Policy,
        *,
        config: PolicyConfig | None = None,
    ) -> None:
        self._require_player(player_id)
        self.controls[player_id] = SeatControl.POLICY
        if config is not None:
            self.policy_configs[player_id] = config
            self._custom_policy_templates.pop(player_id, None)
        else:
            self.policy_configs.pop(player_id, None)
            self._custom_policy_templates[player_id] = copy.deepcopy(policy)
        self._policies[player_id] = copy.deepcopy(policy)

    def next_policy_action(self) -> PolicyStep:
        actor = self.current_actor
        if actor is None or self.game.is_terminal:
            raise RuntimeError("the training hand is terminal")
        if self.controls[actor] != SeatControl.POLICY:
            raise RuntimeError(f"{actor} is human-controlled")
        policy = self._policies.get(actor)
        if policy is None:
            raise RuntimeError(f"no policy assigned to {actor}")
        observation = self.game.observation_for(actor)
        legal = self.game.legal_actions(actor)
        action = policy.decide(observation, legal)
        self.last_policy_trace = copy.deepcopy(getattr(policy, "last_trace", None))
        transition = self.act(action)
        return PolicyStep(actor, action, transition)

    def auto_play_until_human(self, max_actions: int = 1_000) -> tuple[PolicyStep, ...]:
        if max_actions < 1:
            raise ValueError("max_actions must be positive")
        steps: list[PolicyStep] = []
        while not self.game.is_terminal and not self.needs_human_action:
            if len(steps) >= max_actions:
                raise RuntimeError("auto-play action limit exceeded")
            steps.append(self.next_policy_action())
        return tuple(steps)

    def to_dict(self) -> dict[str, Any]:
        custom = set(self._custom_policy_templates)
        if custom:
            raise ValueError(
                f"custom policies are runtime-only and cannot be serialized: {sorted(custom)}"
            )
        return {
            "schema_version": SCHEMA_VERSION,
            "session_id": self.session_id,
            "config": {
                "player_ids": list(self.config.player_ids),
                "starting_stacks": list(self.config.starting_stacks),
                "small_blind": self.config.small_blind,
                "big_blind": self.config.big_blind,
                "button": self.config.button,
            },
            "seed": self.seed,
            "preset_deck": (
                [str(card) for card in self.preset_deck]
                if self.preset_deck is not None
                else None
            ),
            "controls": {
                player_id: control.value for player_id, control in self.controls.items()
            },
            "policies": {
                player_id: {"kind": config.kind.value, "seed": config.seed}
                for player_id, config in self.policy_configs.items()
            },
            "actions": [
                {"player_id": item.player_id, "action": action_to_dict(item.action)}
                for item in self.actions
            ],
            "position": self.position,
            "scenario": copy.deepcopy(self.scenario_metadata),
        }

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> TrainingSession:
        if data.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(
                f"unsupported schema_version {data.get('schema_version')!r}; "
                f"expected {SCHEMA_VERSION}"
            )
        config_data = data.get("config")
        if not isinstance(config_data, Mapping):
            raise ValueError("config must be an object")
        config = TableConfig(
            tuple(config_data["player_ids"]),
            tuple(config_data["starting_stacks"]),
            config_data["small_blind"],
            config_data["big_blind"],
            config_data["button"],
        )
        controls_data = data.get("controls", {})
        controls = {
            player_id: SeatControl(value) for player_id, value in controls_data.items()
        }
        policies_data = data.get("policies", {})
        policies = {
            player_id: PolicyConfig(PolicyKind(value["kind"]), value.get("seed"))
            for player_id, value in policies_data.items()
        }
        action_data = data.get("actions", [])
        actions = tuple(
            TimelineAction(item["player_id"], action_from_dict(item["action"]))
            for item in action_data
        )
        deck_data = data.get("preset_deck")
        preset_deck = parse_cards(deck_data) if deck_data is not None else None
        return cls(
            config,
            seed=data.get("seed"),
            preset_deck=preset_deck,
            controls=controls,
            policy_configs=policies,
            actions=actions,
            position=data.get("position"),
            scenario_metadata=data.get("scenario", {}),
            session_id=data.get("session_id"),
        )

    @classmethod
    def from_json(cls, text: str) -> TrainingSession:
        data = json.loads(text)
        if not isinstance(data, Mapping):
            raise ValueError("training session JSON must contain an object")
        return cls.from_dict(data)

    def _rebuild(self) -> None:
        self.game = HoldemGame(
            self.config,
            seed=self.seed,
            preset_deck=self.preset_deck,
        )
        self.game.start_hand()
        self._policies = {
            player_id: config.build()
            for player_id, config in self.policy_configs.items()
        }
        self._policies.update(
            {
                player_id: copy.deepcopy(policy)
                for player_id, policy in self._custom_policy_templates.items()
            }
        )
        for item in self.actions[: self.position]:
            policy = self._policies.get(item.player_id)
            if (
                policy is not None
                and self.controls[item.player_id] == SeatControl.POLICY
            ):
                observation = self.game.observation_for(item.player_id)
                legal = self.game.legal_actions(item.player_id)
                policy.decide(observation, legal)
            self.game.step(item.action, item.player_id)

    def _require_player(self, player_id: str) -> None:
        if player_id not in self.config.player_ids:
            raise KeyError(f"unknown player {player_id!r}")
