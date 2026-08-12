from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Any, Mapping


@dataclass(frozen=True, slots=True)
class WeightedSize:
    pot_fraction: float
    weight: float = 1.0


@dataclass(frozen=True, slots=True)
class StrategyProfile:
    name: str
    open_ranges: tuple[tuple[str, str], ...]
    call_open_range: str
    three_bet_range: str
    continue_vs_reraise_range: str
    open_raise_frequency: float
    limp_frequency: float
    call_open_frequency: float
    three_bet_frequency: float
    fold_weights: tuple[tuple[str, float], ...]
    call_weights: tuple[tuple[str, float], ...]
    aggression_weights: tuple[tuple[str, float], ...]
    bluff_frequency: float
    semi_bluff_multiplier: float
    open_sizes_bb: tuple[tuple[float, float], ...]
    postflop_sizes: tuple[WeightedSize, ...]

    def open_range(self, position: str) -> str:
        mapping = dict(self.open_ranges)
        return mapping.get(position, mapping.get("default", "22+,A2s+,KTs+,QJs,AJo+"))

    def table(self, name: str) -> dict[str, float]:
        return dict(getattr(self, name))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> StrategyProfile:
        values = dict(data)
        values["open_ranges"] = tuple(tuple(item) for item in values["open_ranges"])
        for field in (
            "fold_weights",
            "call_weights",
            "aggression_weights",
            "open_sizes_bb",
        ):
            values[field] = tuple(tuple(item) for item in values[field])
        values["postflop_sizes"] = tuple(
            WeightedSize(**item) if isinstance(item, Mapping) else WeightedSize(*item)
            for item in values["postflop_sizes"]
        )
        return cls(**values)

    def with_parameter(self, parameter: str, value: float) -> StrategyProfile:
        allowed = {
            "bluff_frequency",
            "semi_bluff_multiplier",
            "open_raise_frequency",
            "call_open_frequency",
            "three_bet_frequency",
        }
        if parameter not in allowed:
            raise ValueError(f"unsupported sweep parameter {parameter!r}")
        if parameter == "bluff_frequency":
            return replace(self, bluff_frequency=value)
        if parameter == "semi_bluff_multiplier":
            return replace(self, semi_bluff_multiplier=value)
        if parameter == "open_raise_frequency":
            return replace(self, open_raise_frequency=value)
        if parameter == "call_open_frequency":
            return replace(self, call_open_frequency=value)
        return replace(self, three_bet_frequency=value)
