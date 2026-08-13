from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import math
from typing import Any, Mapping
from ..ranges import PreflopRange


@dataclass(frozen=True, slots=True)
class WeightedSize:
    pot_fraction: float
    weight: float = 1.0

    def __post_init__(self) -> None:
        if not math.isfinite(self.pot_fraction) or self.pot_fraction <= 0:
            raise ValueError("pot fraction must be finite and positive")
        if not math.isfinite(self.weight) or self.weight <= 0:
            raise ValueError("size weight must be finite and positive")


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

    def __post_init__(self) -> None:
        if not self.open_ranges or "default" not in dict(self.open_ranges):
            raise ValueError("open_ranges must include a default range")
        probabilities = (
            self.open_raise_frequency,
            self.limp_frequency,
            self.call_open_frequency,
            self.three_bet_frequency,
            self.bluff_frequency,
        )
        if any(
            not math.isfinite(value) or not 0 <= value <= 1 for value in probabilities
        ):
            raise ValueError("frequency fields must be finite probabilities in [0,1]")
        if (
            not math.isfinite(self.semi_bluff_multiplier)
            or self.semi_bluff_multiplier < 0
        ):
            raise ValueError("semi_bluff_multiplier must be finite and non-negative")
        for _, expression in self.open_ranges:
            PreflopRange.parse(expression)
        for expression in (
            self.call_open_range,
            self.three_bet_range,
            self.continue_vs_reraise_range,
        ):
            PreflopRange.parse(expression)
        for table_name in ("fold_weights", "call_weights", "aggression_weights"):
            table = getattr(self, table_name)
            if (
                not table
                or not any(weight > 0 for _, weight in table)
                or any(not math.isfinite(weight) or weight < 0 for _, weight in table)
            ):
                raise ValueError(
                    f"{table_name} must contain finite non-negative usable mass"
                )
        if not self.open_sizes_bb or any(
            not math.isfinite(size)
            or size <= 0
            or not math.isfinite(weight)
            or weight <= 0
            for size, weight in self.open_sizes_bb
        ):
            raise ValueError("open sizes and weights must be finite and positive")
        if not self.postflop_sizes:
            raise ValueError("postflop sizes cannot be empty")

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
