from __future__ import annotations

from dataclasses import dataclass
import json
import math
from typing import Iterable, Mapping

from ..agents import PRESETS, PersonalityAgent, StrategyProfile
from ..cards import Card
from ..ranges import RANGE_RANKS, WeightedCombo, WeightedRange
from .bayes import BetaEstimate
from .observation import HandKey, ObservedDecision, ObserverContext


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


@dataclass(frozen=True, slots=True)
class HandRangeInference:
    hand_key: HandKey
    observer_known_cards: tuple[Card, ...]
    conditioned_actions: tuple[str, ...]
    weighted_range: WeightedRange
    summary: RangeSummary


class OpponentStats:
    """Incremental public-action sufficient statistics with real denominators."""

    def __init__(self, alpha_prior: float = 1, beta_prior: float = 1) -> None:
        self.alpha_prior = alpha_prior
        self.beta_prior = beta_prior
        self.counts: dict[str, list[int]] = {}
        self.position_counts: dict[str, dict[str, list[int]]] = {}
        self.street_counts: dict[str, dict[str, list[int]]] = {}
        self.hands: set[HandKey] = set()
        self._preflop_seen: set[HandKey] = set()
        self._vpip_hands: set[HandKey] = set()
        self._pfr_hands: set[HandKey] = set()

    def observe(self, decision: ObservedDecision) -> None:
        hand_key = decision.hand_key
        self.hands.add(hand_key)
        action = decision.action_family
        if decision.street.value == "preflop":
            first = hand_key not in self._preflop_seen
            if first:
                self._preflop_seen.add(hand_key)
                self._add("vpip", False)
                self._add("pfr", False)
                self._position(decision.position, "vpip", False)
                self._position(decision.position, "pfr", False)
            if action in {"call", "raise"} and hand_key not in self._vpip_hands:
                self._vpip_hands.add(hand_key)
                self.counts["vpip"][0] += 1
                self.position_counts[decision.position]["vpip"][0] += 1
            if action == "raise" and hand_key not in self._pfr_hands:
                self._pfr_hands.add(hand_key)
                self.counts["pfr"][0] += 1
                self.position_counts[decision.position]["pfr"][0] += 1
            raises = decision.prior_voluntary_raises
            if raises == 0:
                if decision.can_raise:
                    self._add("open_raise", action == "raise")
                    self._position(decision.position, "open_raise", action == "raise")
                if decision.can_call and decision.to_call > 0:
                    self._add("limp", action == "call")
            elif raises == 1:
                if decision.can_fold:
                    self._add("fold_vs_open", action == "fold")
                if decision.can_call:
                    self._add("call_vs_open", action == "call")
                if decision.can_raise:
                    self._add("three_bet", action == "raise")
                    self._position(decision.position, "three_bet", action == "raise")
            else:
                if decision.can_fold:
                    self._add("fold_vs_three_bet", action == "fold")
                if decision.can_call:
                    self._add("call_vs_three_bet", action == "call")
                if decision.can_raise:
                    self._add("raise_vs_three_bet", action == "raise")
            return

        street = decision.street.value
        aggressive = action in {"bet", "raise"}
        if decision.can_bet:
            self._add("bet_when_checked_to", action == "bet")
            self._street(street, "bet_when_checked_to", action == "bet")
        if decision.to_call > 0:
            for name, target, available in (
                ("fold_vs_bet", "fold", decision.can_fold),
                ("call_vs_bet", "call", decision.can_call),
                ("raise_vs_bet", "raise", decision.can_raise),
            ):
                if not available:
                    continue
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


class OpponentHandBelief:
    """Ephemeral hidden-card belief for exactly one globally identified hand."""

    def __init__(
        self,
        hand_key: HandKey,
        observer_known_cards: Iterable[Card],
        archetypes: Mapping[str, StrategyProfile],
        archetype_posterior: Mapping[str, float],
        *,
        update_archetype_weights: bool = True,
    ) -> None:
        self.hand_key = hand_key
        self.observer_known_cards = tuple(observer_known_cards)
        self.archetypes = dict(archetypes)
        self.archetype_weights = dict(archetype_posterior)
        self.beliefs = {
            name: RangeBelief(known_cards=self.observer_known_cards)
            for name in self.archetypes
        }
        self.conditioned_actions: list[str] = []
        self.update_archetype_weights = update_archetype_weights

    def action_likelihoods(self, decision: ObservedDecision) -> dict[str, float]:
        self._require_hand(decision)
        return {
            name: self.beliefs[name].update(
                decision,
                profile,
                (*self.observer_known_cards, *decision.board),
            )
            for name, profile in self.archetypes.items()
        }

    def observe(self, decision: ObservedDecision) -> dict[str, float]:
        likelihoods = self.action_likelihoods(decision)
        if self.update_archetype_weights:
            updated = {
                name: self.archetype_weights[name] * likelihood
                for name, likelihood in likelihoods.items()
            }
            total = sum(updated.values())
            self.archetype_weights = {
                name: value / total for name, value in updated.items()
            }
        self.conditioned_actions.append(decision.action_family)
        return likelihoods

    def weighted_range(self) -> WeightedRange:
        mixture: dict[tuple[Card, Card], float] = {}
        total = sum(self.archetype_weights.values())
        for name, probability in sorted(self.archetype_weights.items()):
            for cards, weight in self.beliefs[name].weights.items():
                mixture[cards] = mixture.get(cards, 0) + probability / total * weight
        return RangeBelief(mixture, self.observer_known_cards).weighted_range(
            self.observer_known_cards
        )

    def inference(self) -> HandRangeInference:
        weighted = self.weighted_range()
        return HandRangeInference(
            self.hand_key,
            self.observer_known_cards,
            tuple(self.conditioned_actions),
            weighted,
            RangeBelief(
                {combo.cards: combo.weight for combo in weighted.combos},
                self.observer_known_cards,
            ).summary(self.observer_known_cards),
        )

    def _require_hand(self, decision: ObservedDecision) -> None:
        if decision.hand_key != self.hand_key:
            raise ValueError("decision belongs to a different hand belief")


class OpponentModel:
    SCHEMA_VERSION = 2

    def __init__(
        self,
        observer_id: str,
        opponent_id: str,
        *,
        archetypes: Mapping[str, StrategyProfile] = PRESETS,
        archetype_priors: Mapping[str, float] | None = None,
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
        self.observation_count = 0
        self.model_version = 0
        self._learning_hands: dict[HandKey, OpponentHandBelief] = {}
        self._seen_decisions: set[tuple[HandKey, str, int]] = set()

    def observe(
        self, decision: ObservedDecision, *, observer_context: ObserverContext
    ) -> None:
        if not isinstance(decision, ObservedDecision):
            raise TypeError("OpponentModel accepts public ObservedDecision records only")
        if (
            decision.player_id != self.opponent_id
            and decision.participant_id != self.opponent_id
        ):
            return
        self._require_context(decision, observer_context)
        decision_key = (decision.hand_key, decision.player_id, len(decision.history))
        if decision_key in self._seen_decisions:
            return
        self._seen_decisions.add(decision_key)
        belief = self._learning_hands.get(decision.hand_key)
        if belief is None:
            belief = self.start_hand(observer_context)
            self._learning_hands[decision.hand_key] = belief
        self.stats.observe(decision)
        likelihoods = belief.observe(decision)
        for name, evidence in likelihoods.items():
            self.log_posterior[name] += math.log(max(evidence, 1e-12))
        self._normalize_logs()
        self.observation_count += 1
        self.model_version += 1

    def action_probability(
        self,
        decision: ObservedDecision,
        *,
        observer_context: ObserverContext,
        hand_belief: OpponentHandBelief | None = None,
    ) -> float:
        """Predict a public action without mutating model state."""
        self._require_context(decision, observer_context)
        belief = hand_belief or self.start_hand(observer_context)
        probability = 0.0
        for name, profile in self.archetypes.items():
            evidence = 0.0
            agent = PersonalityAgent(profile)
            legal_prior = _normalized(
                belief.beliefs[name].weights,
                (*observer_context.observer_known_cards, *decision.board),
            )
            for cards, prior in legal_prior.items():
                distribution = agent.action_distribution(
                    decision.observation_with_hole_cards(cards), decision.legal_actions()
                )
                evidence += prior * distribution.get(decision.action_family, 0.0)
            probability += belief.archetype_weights[name] * evidence
        return max(1e-12, probability)

    def start_hand(
        self,
        context: ObserverContext,
        *,
        update_archetype_weights: bool = True,
    ) -> OpponentHandBelief:
        return OpponentHandBelief(
            context.hand_key,
            context.observer_known_cards,
            self.archetypes,
            self.archetype_posterior,
            update_archetype_weights=update_archetype_weights,
        )

    def infer_range_for_hand(
        self,
        decisions: Iterable[ObservedDecision],
        *,
        observer_context: ObserverContext,
    ) -> HandRangeInference:
        belief = self.start_hand(observer_context)
        for decision in decisions:
            if (
                decision.player_id == self.opponent_id
                or decision.participant_id == self.opponent_id
            ):
                belief.observe(decision)
        return belief.inference()

    @property
    def archetype_posterior(self) -> dict[str, float]:
        return {name: math.exp(value) for name, value in self.log_posterior.items()}

    def snapshot(self) -> OpponentSnapshot:
        names = (
            "vpip", "pfr", "open_raise", "limp", "call_vs_open",
            "fold_vs_open", "three_bet", "fold_vs_three_bet",
            "call_vs_three_bet", "raise_vs_three_bet", "bet_when_checked_to",
            "fold_vs_bet", "call_vs_bet", "raise_vs_bet", "aggression",
        )
        return OpponentSnapshot(
            self.opponent_id,
            len(self.stats.hands),
            {name: self.stats.estimate(name) for name in names},
            self.stats.grouped_estimates(self.stats.position_counts),
            self.stats.grouped_estimates(self.stats.street_counts),
            self.archetype_posterior,
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
            "No current-hand range is implied by historical statistics alone.",
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
            "hands": [_hand_key_dict(key) for key in sorted(self.stats.hands)],
            "preflop_seen": [
                _hand_key_dict(key) for key in sorted(self.stats._preflop_seen)
            ],
            "vpip_hands": [
                _hand_key_dict(key) for key in sorted(self.stats._vpip_hands)
            ],
            "pfr_hands": [
                _hand_key_dict(key) for key in sorted(self.stats._pfr_hands)
            ],
            "log_posterior": self.log_posterior,
            "observation_count": self.observation_count,
            "model_version": self.model_version,
            "seen_decisions": [
                {
                    "hand_key": _hand_key_dict(hand_key),
                    "player_id": player_id,
                    "sequence": sequence,
                }
                for hand_key, player_id, sequence in sorted(self._seen_decisions)
            ],
            "learning_hands": [
                {
                    "hand_key": _hand_key_dict(hand_key),
                    "observer_known_cards": [
                        str(card) for card in belief.observer_known_cards
                    ],
                    "archetype_weights": belief.archetype_weights,
                    "conditioned_actions": belief.conditioned_actions,
                    "beliefs": {
                        name: {
                            "".join(map(str, cards)): weight
                            for cards, weight in range_belief.weights.items()
                        }
                        for name, range_belief in belief.beliefs.items()
                    },
                }
                for hand_key, belief in sorted(self._learning_hands.items())
            ],
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
            alpha_prior=payload["alpha_prior"], beta_prior=payload["beta_prior"],
        )
        model.stats.counts = payload["counts"]
        model.stats.position_counts = payload["position_counts"]
        model.stats.street_counts = payload["street_counts"]
        model.stats.hands = {_hand_key(value) for value in payload["hands"]}
        model.stats._preflop_seen = {
            _hand_key(value) for value in payload["preflop_seen"]
        }
        model.stats._vpip_hands = {
            _hand_key(value) for value in payload["vpip_hands"]
        }
        model.stats._pfr_hands = {
            _hand_key(value) for value in payload["pfr_hands"]
        }
        model.log_posterior = payload["log_posterior"]
        model.observation_count = payload["observation_count"]
        model.model_version = payload["model_version"]
        model._seen_decisions = {
            (
                _hand_key(value["hand_key"]),
                value["player_id"],
                value["sequence"],
            )
            for value in payload.get("seen_decisions", [])
        }
        model._learning_hands = {}
        for value in payload.get("learning_hands", []):
            hand_key = _hand_key(value["hand_key"])
            belief = OpponentHandBelief(
                hand_key,
                parse_cards(value["observer_known_cards"]),
                model.archetypes,
                value["archetype_weights"],
            )
            belief.conditioned_actions = list(value["conditioned_actions"])
            for name, weights in value["beliefs"].items():
                belief.beliefs[name] = RangeBelief(
                    {
                        parse_cards((combo[:2], combo[2:])): weight
                        for combo, weight in weights.items()
                    }
                )
            model._learning_hands[hand_key] = belief
        return model

    def _require_context(
        self, decision: ObservedDecision, context: ObserverContext
    ) -> None:
        if context.hand_key != decision.hand_key:
            raise ValueError("observer context belongs to a different hand")
        if context.observer_id != self.observer_id:
            raise ValueError("observer context belongs to a different observer")

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
        *,
        reset_each_hand: bool = False,
    ) -> None:
        self.observer_id = observer_id
        self.opponents = tuple(opponent for opponent in opponents if opponent != observer_id)
        self.reset_each_hand = reset_each_hand
        self.current_hand: HandKey | None = None
        self.models = {
            opponent: OpponentModel(observer_id, opponent)
            for opponent in self.opponents
        }

    def observe(
        self, decision: ObservedDecision, *, observer_context: ObserverContext
    ) -> None:
        if self.reset_each_hand and self.current_hand != decision.hand_key:
            self.models = {
                opponent: OpponentModel(self.observer_id, opponent)
                for opponent in self.opponents
            }
        self.current_hand = decision.hand_key
        model = self.models.get(decision.participant_id) or self.models.get(
            decision.player_id
        )
        if model is not None:
            model.observe(decision, observer_context=observer_context)


def _hand_key_dict(key: HandKey) -> dict[str, object]:
    return {"session_id": key.session_id, "hand_index": key.hand_index}


def _hand_key(value: Mapping[str, object]) -> HandKey:
    return HandKey(str(value["session_id"]), int(value["hand_index"]))


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
