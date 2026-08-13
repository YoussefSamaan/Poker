from __future__ import annotations

from dataclasses import dataclass

from ..agents import PersonalityAgent, StrategyProfile, extract_features
from ..holdem import Action, LegalActions, PlayerObservation
from .model import OpponentModel


@dataclass(frozen=True, slots=True)
class AdaptationConfig:
    maximum_probability_shift: float = 0.15
    evidence_half_saturation: int = 40


class AdaptiveExploitPolicy:
    """Small confidence-aware adjustment around an existing synthetic policy."""

    def __init__(
        self,
        base_profile: StrategyProfile,
        opponent_model: OpponentModel,
        *,
        seed: int = 0,
        config: AdaptationConfig = AdaptationConfig(),
    ) -> None:
        self.base = PersonalityAgent(base_profile, seed)
        self.model = opponent_model
        self.config = config

    def action_distribution(
        self, observation: PlayerObservation, legal: LegalActions
    ) -> dict[str, float]:
        distribution = self.base.action_distribution(observation, legal)
        features = extract_features(observation, legal)
        if features.street.value == "preflop" or features.bucket.value != "air":
            return distribution
        fold = self.model.stats.estimate("fold_vs_bet")
        call = self.model.stats.estimate("call_vs_bet")
        confidence = fold.opportunities / (
            fold.opportunities + self.config.evidence_half_saturation
        )
        tendency = fold.mean - call.mean
        shift = max(
            -self.config.maximum_probability_shift,
            min(
                self.config.maximum_probability_shift,
                self.config.maximum_probability_shift * confidence * tendency,
            ),
        )
        aggressive = "bet" if "bet" in distribution else "raise"
        if aggressive not in distribution or not shift:
            return distribution
        donors = [name for name in distribution if name != aggressive]
        adjusted = dict(distribution)
        if shift > 0:
            available = sum(adjusted[name] for name in donors)
            actual = min(shift, available)
            adjusted[aggressive] += actual
            if available:
                for name in donors:
                    adjusted[name] -= actual * adjusted[name] / available
        else:
            actual = min(-shift, adjusted[aggressive])
            adjusted[aggressive] -= actual
            target = "check" if "check" in adjusted else (
                "call" if "call" in adjusted else donors[0]
            )
            adjusted[target] += actual
        total = sum(adjusted.values())
        return {name: value / total for name, value in adjusted.items() if value > 0}

    def decide(
        self, observation: PlayerObservation, legal: LegalActions
    ) -> Action:
        distribution = self.action_distribution(observation, legal)
        draw = self.base._rng.random()  # noqa: SLF001 - preserve base policy RNG stream
        cumulative = 0.0
        family = next(iter(distribution))
        for candidate, probability in distribution.items():
            cumulative += probability
            if draw <= cumulative:
                family = candidate
                break
        return self.base._action_for_family(  # noqa: SLF001 - intentional composition
            family, extract_features(observation, legal), legal
        )
