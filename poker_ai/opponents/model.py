from __future__ import annotations

from dataclasses import dataclass
import json
import math
from typing import Iterable, Mapping

from ..agents import PRESETS, PersonalityAgent, StrategyProfile
from ..cards import Card
from ..ranges import RANGE_RANKS, WeightedCombo, WeightedRange
from .bayes import BetaEstimate
from .observation import ObservedDecision


@dataclass(frozen=True, slots=True)
class TendencyEstimate:
    mean: float
    credible_interval_95: tuple[float, float]
    opportunities: int
    successes: int


@dataclass(frozen=True, slots=True)
class RangeSummary:
    legal_combo_count: int
    entropy: float
    effective_combo_count: float
    top_hand_classes: tuple[tuple[str, float], ...]
    matrix: tuple[tuple[float, ...], ...]


@dataclass(frozen=True, slots=True)
class OpponentSnapshot:
    opponent_id: str
    hands_observed: int
    tendencies: Mapping[str, TendencyEstimate]
    position_tendencies: Mapping[str, Mapping[str, TendencyEstimate]]
    street_tendencies: Mapping[str, Mapping[str, TendencyEstimate]]
    archetype_posterior: Mapping[str, float]
    range_summary: RangeSummary


class OpponentStats:
    """Incremental public-action sufficient statistics with real denominators."""

    def __init__(self, alpha_prior: float = 1, beta_prior: float = 1) -> None:
        self.alpha_prior = alpha_prior
        self.beta_prior = beta_prior
        self.counts: dict[str, list[int]] = {}
        self.position_counts: dict[str, dict[str, list[int]]] = {}
        self.street_counts: dict[str, dict[str, list[int]]] = {}
        self.hands: set[int] = set()
        self._preflop_seen: set[int] = set()
        self._vpip_hands: set[int] = set()
        self._pfr_hands: set[int] = set()

    def observe(self, decision: ObservedDecision) -> None:
        self.hands.add(decision.hand_index)
        action = decision.action_family
        if decision.street.value == "preflop":
            first = decision.hand_index not in self._preflop_seen
            if first:
                self._preflop_seen.add(decision.hand_index)
                self._add("vpip", False)
                self._add("pfr", False)
                self._position(decision.position, "vpip", False)
                self._position(decision.position, "pfr", False)
            if action in {"call", "raise"} and decision.hand_index not in self._vpip_hands:
                self._vpip_hands.add(decision.hand_index)
                self.counts["vpip"][0] += 1
                self.position_counts[decision.position]["vpip"][0] += 1
            if action == "raise" and decision.hand_index not in self._pfr_hands:
                self._pfr_hands.add(decision.hand_index)
                self.counts["pfr"][0] += 1
                self.position_counts[decision.position]["pfr"][0] += 1
            raises = decision.prior_voluntary_raises
            if raises == 0:
                if decision.can_raise:
                    self._add("open_raise", action == "raise")
                    self._position(decision.position, "open_raise", action == "raise")
                if decision.can_call or decision.can_check:
                    self._add("limp", action == "call")
            elif raises == 1:
                self._add("fold_vs_open", action == "fold")
                self._add("call_vs_open", action == "call")
                if decision.can_raise:
                    self._add("three_bet", action == "raise")
                    self._position(decision.position, "three_bet", action == "raise")
            else:
                self._add("fold_vs_three_bet", action == "fold")
                self._add("call_vs_three_bet", action == "call")
                if decision.can_raise:
                    self._add("raise_vs_three_bet", action == "raise")
            return

        street = decision.street.value
        aggressive = action in {"bet", "raise"}
        if decision.can_bet:
            self._add("bet_when_checked_to", action == "bet")
            self._street(street, "bet_when_checked_to", action == "bet")
        if decision.to_call > 0 and decision.can_fold:
            for name, target in (
                ("fold_vs_bet", "fold"),
                ("call_vs_bet", "call"),
                ("raise_vs_bet", "raise"),
            ):
                self._add(name, action == target)
                self._street(street, name, action == target)
        if decision.can_bet or decision.can_raise:
            self._add("aggression", aggressive)
            self._street(street, "aggression", aggressive)

    def estimate(self, name: str) -> TendencyEstimate:
        successes, opportunities = self.counts.get(name, [0, 0])
        posterior = BetaEstimate(
            successes,
            opportunities - successes,
            self.alpha_prior,
            self.beta_prior,
        )
        return TendencyEstimate(
            posterior.mean,
            posterior.credible_interval(),
            opportunities,
            successes,
        )

    def _add(self, name: str, success: bool) -> None:
        value = self.counts.setdefault(name, [0, 0])
        value[0] += int(success)
        value[1] += 1

    def _position(self, position: str, name: str, success: bool) -> None:
        value = self.position_counts.setdefault(position, {}).setdefault(name, [0, 0])
        value[0] += int(success)
        value[1] += 1

    def _street(self, street: str, name: str, success: bool) -> None:
        value = self.street_counts.setdefault(street, {}).setdefault(name, [0, 0])
        value[0] += int(success)
        value[1] += 1

    def grouped_estimates(
        self, groups: Mapping[str, Mapping[str, list[int]]]
    ) -> dict[str, dict[str, TendencyEstimate]]:
        result = {}
        for group, values in groups.items():
            result[group] = {}
            for name, (successes, opportunities) in values.items():
                posterior = BetaEstimate(
                    successes,
                    opportunities - successes,
                    self.alpha_prior,
                    self.beta_prior,
                )
                result[group][name] = TendencyEstimate(
                    posterior.mean,
                    posterior.credible_interval(),
                    opportunities,
                    successes,
                )
        return result


class RangeBelief:
    """Sequential blocker-aware posterior over concrete two-card combinations."""

    def __init__(
        self,
        weights: Mapping[tuple[Card, Card], float] | None = None,
        known_cards: Iterable[Card] = (),
    ) -> None:
        if weights is None:
            initial = WeightedRange.random(known_cards)
            weights = {combo.cards: combo.weight for combo in initial.combos}
        self.weights = _normalized(weights, known_cards)

    def update(
        self,
        decision: ObservedDecision,
        profile: StrategyProfile,
        known_cards: Iterable[Card] = (),
        epsilon: float = 1e-9,
    ) -> float:
        blockers = (*known_cards, *decision.board)
        legal_prior = _normalized(self.weights, blockers)
        likelihoods = {}
        agent = PersonalityAgent(profile)
        for cards, prior in legal_prior.items():
            observation = decision.observation_with_hole_cards(cards)
            distribution = agent.action_distribution(observation, decision.legal_actions())
            likelihoods[cards] = prior * max(
                epsilon, distribution.get(decision.action_family, 0.0)
            )
        evidence = sum(likelihoods.values())
        self.weights = _normalized(likelihoods, blockers)
        return evidence

    def weighted_range(self, known_cards: Iterable[Card] = ()) -> WeightedRange:
        weights = _normalized(self.weights, known_cards)
        return WeightedRange(WeightedCombo(cards, weight) for cards, weight in weights.items())

    def summary(self, known_cards: Iterable[Card] = ()) -> RangeSummary:
        weights = _normalized(self.weights, known_cards)
        entropy = -sum(value * math.log(value) for value in weights.values())
        classes: dict[str, float] = {}
        for cards, value in sorted(weights.items(), key=lambda item: tuple(map(str, item[0]))):
            label = _hand_class(cards)
            classes[label] = classes.get(label, 0) + value
        matrix = []
        for row, first in enumerate(RANGE_RANKS):
            values = []
            for column, second in enumerate(RANGE_RANKS):
                label = (
                    first + second
                    if row == column
                    else (first + second + "s" if row < column else second + first + "o")
                )
                values.append(classes.get(label, 0.0))
            matrix.append(tuple(values))
        return RangeSummary(
            len(weights),
            entropy,
            math.exp(entropy),
            tuple(sorted(classes.items(), key=lambda item: item[1], reverse=True)[:10]),
            tuple(matrix),
        )


class OpponentModel:
    SCHEMA_VERSION = 1

    def __init__(
        self,
        observer_id: str,
        opponent_id: str,
        *,
        archetypes: Mapping[str, StrategyProfile] = PRESETS,
        archetype_priors: Mapping[str, float] | None = None,
        known_cards: Iterable[Card] = (),
        alpha_prior: float = 1,
        beta_prior: float = 1,
    ) -> None:
        self.observer_id = observer_id
        self.opponent_id = opponent_id
        self.archetypes = dict(archetypes)
        priors = (
            {name: 1 for name in self.archetypes}
            if archetype_priors is None
            else archetype_priors
        )
        total = sum(priors.values())
        if (
            not self.archetypes
            or set(priors) != set(self.archetypes)
            or any(value <= 0 or not math.isfinite(value) for value in priors.values())
        ):
            raise ValueError("archetype priors must cover every archetype with positive mass")
        self.log_posterior = {name: math.log(priors[name] / total) for name in priors}
        self.stats = OpponentStats(alpha_prior, beta_prior)
        self.known_cards = tuple(known_cards)
        self.beliefs = {
            name: RangeBelief(known_cards=self.known_cards) for name in self.archetypes
        }
        self.current_hand: int | None = None

    def observe(self, decision: ObservedDecision) -> None:
        if not isinstance(decision, ObservedDecision):
            raise TypeError("OpponentModel accepts public ObservedDecision records only")
        if (
            decision.player_id != self.opponent_id
            and decision.participant_id != self.opponent_id
        ):
            return
        if self.current_hand != decision.hand_index:
            self.current_hand = decision.hand_index
            self.beliefs = {
                name: RangeBelief(known_cards=(*self.known_cards, *decision.board))
                for name in self.archetypes
            }
        self.stats.observe(decision)
        for name, profile in self.archetypes.items():
            evidence = self.beliefs[name].update(
                decision, profile, (*self.known_cards, *decision.board)
            )
            self.log_posterior[name] += math.log(max(evidence, 1e-12))
        self._normalize_logs()

    def action_probability(self, decision: ObservedDecision) -> float:
        """Predict a public action without mutating model state."""
        probability = 0.0
        for name, profile in self.archetypes.items():
            belief = (
                self.beliefs[name]
                if self.current_hand == decision.hand_index
                else RangeBelief(known_cards=(*self.known_cards, *decision.board))
            )
            evidence = 0.0
            agent = PersonalityAgent(profile)
            legal_prior = _normalized(
                belief.weights, (*self.known_cards, *decision.board)
            )
            for cards, prior in legal_prior.items():
                distribution = agent.action_distribution(
                    decision.observation_with_hole_cards(cards), decision.legal_actions()
                )
                evidence += prior * distribution.get(decision.action_family, 0.0)
            probability += self.archetype_posterior[name] * evidence
        return max(1e-12, probability)

    @property
    def archetype_posterior(self) -> dict[str, float]:
        return {name: math.exp(value) for name, value in self.log_posterior.items()}

    def inferred_range(self) -> WeightedRange:
        mixture: dict[tuple[Card, Card], float] = {}
        for name, probability in sorted(self.archetype_posterior.items()):
            for cards, weight in self.beliefs[name].weights.items():
                mixture[cards] = mixture.get(cards, 0) + probability * weight
        return RangeBelief(mixture, self.known_cards).weighted_range(self.known_cards)

    def snapshot(self) -> OpponentSnapshot:
        names = (
            "vpip", "pfr", "open_raise", "limp", "call_vs_open",
            "fold_vs_open", "three_bet", "fold_vs_three_bet",
            "call_vs_three_bet", "raise_vs_three_bet", "bet_when_checked_to",
            "fold_vs_bet", "call_vs_bet", "raise_vs_bet", "aggression",
        )
        mixture = {
            combo.cards: combo.weight for combo in self.inferred_range().combos
        }
        return OpponentSnapshot(
            self.opponent_id,
            len(self.stats.hands),
            {name: self.stats.estimate(name) for name in names},
            self.stats.grouped_estimates(self.stats.position_counts),
            self.stats.grouped_estimates(self.stats.street_counts),
            self.archetype_posterior,
            RangeBelief(mixture, self.known_cards).summary(self.known_cards),
        )

    def explanation(self) -> tuple[str, ...]:
        snapshot = self.snapshot()
        vpip = snapshot.tendencies["vpip"]
        posterior = sorted(
            snapshot.archetype_posterior.items(), key=lambda item: item[1], reverse=True
        )
        return (
            f"Observed {snapshot.hands_observed} hands.",
            f"{self.opponent_id} voluntarily entered {vpip.successes} of "
            f"{vpip.opportunities} observed preflop opportunities.",
            f"Posterior VPIP mean: {vpip.mean:.1%}; 95% credible interval "
            f"{vpip.credible_interval_95[0]:.1%}–{vpip.credible_interval_95[1]:.1%}.",
            "Archetype posterior: "
            + ", ".join(f"{name} {value:.1%}" for name, value in posterior)
            + ".",
            f"Current inferred range: {snapshot.range_summary.legal_combo_count} legal "
            f"combos; entropy {snapshot.range_summary.entropy:.2f} nats.",
        )

    def to_json(self) -> str:
        payload = {
            "schema_version": self.SCHEMA_VERSION,
            "observer_id": self.observer_id,
            "opponent_id": self.opponent_id,
            "alpha_prior": self.stats.alpha_prior,
            "beta_prior": self.stats.beta_prior,
            "counts": self.stats.counts,
            "position_counts": self.stats.position_counts,
            "street_counts": self.stats.street_counts,
            "hands": sorted(self.stats.hands),
            "preflop_seen": sorted(self.stats._preflop_seen),
            "vpip_hands": sorted(self.stats._vpip_hands),
            "pfr_hands": sorted(self.stats._pfr_hands),
            "known_cards": [str(card) for card in self.known_cards],
            "log_posterior": self.log_posterior,
            "current_hand": self.current_hand,
            "beliefs": {
                name: {
                    "".join(map(str, cards)): weight
                    for cards, weight in belief.weights.items()
                }
                for name, belief in self.beliefs.items()
            },
        }
        return json.dumps(payload, sort_keys=True)

    @classmethod
    def from_json(
        cls, text: str, *, archetypes: Mapping[str, StrategyProfile] = PRESETS
    ) -> OpponentModel:
        from ..cards import parse_cards

        payload = json.loads(text)
        if payload["schema_version"] != cls.SCHEMA_VERSION:
            raise ValueError("unsupported opponent model schema")
        model = cls(
            payload["observer_id"], payload["opponent_id"], archetypes=archetypes,
            known_cards=parse_cards(payload["known_cards"]),
            alpha_prior=payload["alpha_prior"], beta_prior=payload["beta_prior"],
        )
        model.stats.counts = payload["counts"]
        model.stats.position_counts = payload["position_counts"]
        model.stats.street_counts = payload["street_counts"]
        model.stats.hands = set(payload["hands"])
        model.stats._preflop_seen = set(payload["preflop_seen"])
        model.stats._vpip_hands = set(payload["vpip_hands"])
        model.stats._pfr_hands = set(payload["pfr_hands"])
        model.log_posterior = payload["log_posterior"]
        model.current_hand = payload["current_hand"]
        model.beliefs = {}
        for name, values in payload["beliefs"].items():
            belief = object.__new__(RangeBelief)
            belief.weights = {
                parse_cards((combo[:2], combo[2:])): weight
                for combo, weight in values.items()
            }
            model.beliefs[name] = belief
        return model

    def _normalize_logs(self) -> None:
        maximum = max(self.log_posterior.values())
        denominator = sum(math.exp(value - maximum) for value in self.log_posterior.values())
        normalizer = maximum + math.log(denominator)
        self.log_posterior = {
            name: value - normalizer for name, value in self.log_posterior.items()
        }


class OpponentModelTable:
    """One independent model per opponent from an explicit observer perspective."""

    def __init__(
        self,
        observer_id: str,
        opponents: Iterable[str],
        known_cards=(),
        *,
        reset_each_hand: bool = False,
    ) -> None:
        self.observer_id = observer_id
        self.opponents = tuple(opponent for opponent in opponents if opponent != observer_id)
        self.known_cards = tuple(known_cards)
        self.reset_each_hand = reset_each_hand
        self.current_hand: int | None = None
        self.models = {
            opponent: OpponentModel(observer_id, opponent, known_cards=known_cards)
            for opponent in self.opponents
        }

    def observe(self, decision: ObservedDecision) -> None:
        if self.reset_each_hand and self.current_hand != decision.hand_index:
            self.models = {
                opponent: OpponentModel(
                    self.observer_id, opponent, known_cards=self.known_cards
                )
                for opponent in self.opponents
            }
        self.current_hand = decision.hand_index
        model = self.models.get(decision.participant_id) or self.models.get(
            decision.player_id
        )
        if model is not None:
            model.observe(decision)


def _normalized(
    weights: Mapping[tuple[Card, Card], float], dead_cards: Iterable[Card]
) -> dict[tuple[Card, Card], float]:
    dead = set(dead_cards)
    legal = dict(
        sorted(
            (
                (tuple(sorted(cards, key=str)), value)
                for cards, value in weights.items()
                if value > 0 and not dead.intersection(cards)
            ),
            key=lambda item: tuple(map(str, item[0])),
        )
    )
    total = sum(legal.values())
    if total <= 0:
        raise ValueError("range belief has no legal positive mass")
    return {cards: value / total for cards, value in legal.items()}


def _hand_class(cards: tuple[Card, Card]) -> str:
    first, second = sorted(cards, key=lambda card: card.rank_value, reverse=True)
    if first.rank == second.rank:
        return first.rank + second.rank
    return first.rank + second.rank + ("s" if first.suit == second.suit else "o")
