from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping

from ..dataset import OpponentFeatureVector, PublicDecisionExample

ACTION_CLASSES = ("fold", "check", "call", "bet", "raise")
FEATURE_SCHEMA_VERSION = 1


HISTORY_TENDENCIES = (
    "vpip",
    "pfr",
    "open_raise",
    "three_bet",
    "fold_vs_bet",
    "call_vs_bet",
    "raise_vs_bet",
    "bet_when_checked_to",
    "aggression",
)


@dataclass(frozen=True, slots=True)
class OpponentHistoryFeatures:
    values: tuple[tuple[str, float], ...]

    def as_dict(self) -> dict[str, float]:
        return dict(self.values)


@dataclass(frozen=True, slots=True)
class HistoryAwareExample:
    public: PublicDecisionExample
    history: OpponentHistoryFeatures


@dataclass(frozen=True, slots=True)
class CandidateHandFeatures:
    high_rank: int
    low_rank: int
    pair: int
    suited: int
    rank_gap: int
    hand_class: str
    made_category: str
    flush_draw: int
    open_ended_straight_draw: int
    gutshot_straight_draw: int
    board_interaction_count: int


@dataclass(frozen=True, slots=True)
class ResearchHandConditionedExample:
    """RESEARCH / SYNTHETIC GROUND TRUTH training row."""

    public: PublicDecisionExample
    history: OpponentHistoryFeatures
    candidate: CandidateHandFeatures
    true_hole_cards: tuple[str, str]


def context_feature_mapping(features: OpponentFeatureVector) -> dict[str, object]:
    """Model inputs only; target, IDs, grouping keys, and action sizing are absent."""
    return asdict(features)


def history_feature_mapping(
    features: OpponentHistoryFeatures,
) -> Mapping[str, float]:
    return features.as_dict()
