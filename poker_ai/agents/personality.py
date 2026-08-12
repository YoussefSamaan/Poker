from __future__ import annotations

import random

from ..holdem import (
    Action,
    BetTo,
    Call,
    Check,
    Fold,
    LegalActions,
    PlayerObservation,
    RaiseTo,
    Street,
)
from ..ranges import PreflopRange
from .features import DecisionFeatures, extract_features
from .profile import StrategyProfile
from .trace import DecisionTrace, PolicyDecision


class PersonalityAgent:
    """Fixed synthetic policy configuration; not an inferred opponent model."""

    def __init__(self, profile: StrategyProfile, seed: int | None = 0) -> None:
        self.profile = profile
        self._rng = random.Random(seed)
        self.last_trace: DecisionTrace | None = None

    def action_distribution(
        self, observation: PlayerObservation, legal: LegalActions
    ) -> dict[str, float]:
        features = extract_features(observation, legal)
        if features.street == Street.PREFLOP:
            distribution = self._preflop_distribution(features, legal)
        else:
            distribution = self._postflop_distribution(features, legal)
        legal_families = _legal_families(legal)
        filtered = {
            family: max(0.0, probability)
            for family, probability in distribution.items()
            if family in legal_families
        }
        if not filtered or sum(filtered.values()) <= 0:
            fallback = (
                "check" if legal.can_check else ("call" if legal.can_call else "fold")
            )
            return {fallback: 1.0}
        total = sum(filtered.values())
        return {family: value / total for family, value in filtered.items()}

    def decide(
        self, observation: PlayerObservation, legal_actions: LegalActions
    ) -> Action:
        return self.decide_with_trace(observation, legal_actions).action

    def decide_with_trace(
        self, observation: PlayerObservation, legal: LegalActions
    ) -> PolicyDecision:
        features = extract_features(observation, legal)
        probabilities = self.action_distribution(observation, legal)
        draw = self._rng.random()
        cumulative = 0.0
        family = next(iter(probabilities))
        for candidate, probability in probabilities.items():
            cumulative += probability
            if draw <= cumulative:
                family = candidate
                break
        action = self._action_for_family(family, features, legal)
        target = action.amount if isinstance(action, (BetTo, RaiseTo)) else None
        rationale = (
            f"Profile: {self.profile.name} (synthetic configuration).",
            f"{features.street.value.title()} {features.position}; hand class {features.hand_class}; bucket {features.bucket.value}.",
            f"Pot {features.pot}; call {features.to_call}; pot odds {features.pot_odds:.1%}.",
            "Configured legal-family probabilities: "
            + ", ".join(f"{key} {value:.1%}" for key, value in probabilities.items())
            + ".",
            f"Sample {draw:.6f} selected {family}"
            + (f" to {target}." if target is not None else "."),
        )
        trace = DecisionTrace(
            self.profile.name,
            features,
            tuple(probabilities.items()),
            family,
            type(action).__name__,
            target,
            draw,
            rationale,
        )
        self.last_trace = trace
        return PolicyDecision(action, trace)

    def _preflop_distribution(
        self, features: DecisionFeatures, legal: LegalActions
    ) -> dict[str, float]:
        hand = features.hand_class
        if features.prior_raises == 0:
            inside = _contains(self.profile.open_range(features.position), hand)
            if inside:
                aggressive = self.profile.open_raise_frequency
                call = self.profile.limp_frequency
                return {
                    "raise": aggressive,
                    "call": call,
                    "check": call,
                    "fold": 1 - aggressive - call,
                }
            return {"check": 0.12, "fold": 0.88}
        if features.prior_raises == 1:
            three_bet = (
                self.profile.three_bet_frequency
                if _contains(self.profile.three_bet_range, hand)
                else 0.02
            )
            call = (
                self.profile.call_open_frequency
                if _contains(self.profile.call_open_range, hand)
                else 0.03
            )
            return {"raise": three_bet, "call": call, "fold": 1 - three_bet - call}
        continue_weight = (
            self.profile.call_open_frequency
            if _contains(self.profile.continue_vs_reraise_range, hand)
            else 0.02
        )
        return {
            "raise": self.profile.three_bet_frequency * 0.35,
            "call": continue_weight,
            "fold": 1 - continue_weight - self.profile.three_bet_frequency * 0.35,
        }

    def _postflop_distribution(
        self, features: DecisionFeatures, legal: LegalActions
    ) -> dict[str, float]:
        bucket = features.bucket.value
        aggression = self.profile.table("aggression_weights").get(bucket, 0.2)
        call = self.profile.table("call_weights").get(bucket, 0.2)
        if bucket == "air":
            aggression = self.profile.bluff_frequency
        elif bucket == "draw":
            aggression = min(1.0, aggression * self.profile.semi_bluff_multiplier)
        if legal.can_check:
            return {"bet": aggression, "check": 1 - aggression}
        fold = max(0.0, 1 - aggression - call)
        return {"raise": aggression, "call": call, "fold": fold}

    def _action_for_family(
        self, family: str, features: DecisionFeatures, legal: LegalActions
    ) -> Action:
        if family == "fold":
            return Fold()
        if family == "check":
            return Check()
        if family == "call":
            return Call()
        minimum = legal.min_bet_to if family == "bet" else legal.min_raise_to
        maximum = legal.max_bet_to if family == "bet" else legal.max_raise_to
        if minimum is None or maximum is None:
            return (
                Check() if legal.can_check else (Call() if legal.can_call else Fold())
            )
        if features.street == Street.PREFLOP:
            sizes, weights = zip(*self.profile.open_sizes_bb)
            requested = round(
                self._rng.choices(sizes, weights=weights, k=1)[0] * features.big_blind
            )
            target = (
                requested
                if features.prior_raises == 0
                else max(requested, features.current_bet * 3)
            )
        else:
            size = self._rng.choices(
                self.profile.postflop_sizes,
                weights=[item.weight for item in self.profile.postflop_sizes],
                k=1,
            )[0]
            additional = max(1, round(features.pot * size.pot_fraction))
            target = features.street_contribution + features.to_call + additional
        target = min(max(target, minimum), maximum)
        return BetTo(target) if family == "bet" else RaiseTo(target)


def _contains(expression: str, hand_class: str) -> bool:
    return hand_class in dict(PreflopRange.parse(expression).class_weights)


def _legal_families(legal: LegalActions) -> set[str]:
    families = set()
    if legal.can_fold:
        families.add("fold")
    if legal.can_check:
        families.add("check")
    if legal.can_call:
        families.add("call")
    if legal.can_bet:
        families.add("bet")
    if legal.can_raise:
        families.add("raise")
    return families
