from __future__ import annotations

from dataclasses import dataclass

from ..holdem import Action
from .features import DecisionFeatures


@dataclass(frozen=True, slots=True)
class DecisionTrace:
    profile: str
    features: DecisionFeatures
    probabilities: tuple[tuple[str, float], ...]
    chosen_family: str
    chosen_action: str
    chosen_target: int | None
    random_draw: float
    rationale: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class PolicyDecision:
    action: Action
    trace: DecisionTrace
