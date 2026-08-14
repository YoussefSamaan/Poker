from __future__ import annotations

import math
from typing import Iterable

from ...experiments.simulator import ExperimentResult
from ..dataset import PublicObservationDataset
from ..model import OpponentStats
from ..observation import ObservedDecision
from .schema import HISTORY_TENDENCIES, HistoryAwareExample, OpponentHistoryFeatures


def history_features_from_stats(stats: OpponentStats) -> OpponentHistoryFeatures:
    values = []
    for tendency in HISTORY_TENDENCIES:
        estimate = stats.estimate(tendency)
        values.extend(
            (
                (f"history_{tendency}_mean", estimate.mean),
                (
                    f"history_{tendency}_log_opportunities",
                    math.log1p(estimate.opportunities),
                ),
            )
        )
    return OpponentHistoryFeatures(tuple(values))


def causal_history_examples(
    results: Iterable[ExperimentResult],
) -> tuple[HistoryAwareExample, ...]:
    """Emit each row from statistics through t-1, then observe decision t."""
    histories: dict[tuple[str, str], OpponentStats] = {}
    rows = []
    for result in results:
        dataset = PublicObservationDataset.from_experiment(result)
        decisions = tuple(
            decision
            for record in result.records
            for decision in record.observed_decisions
        )
        if len(dataset.examples) != len(decisions):
            raise AssertionError("public dataset and decisions lost alignment")
        for public, decision in zip(dataset.examples, decisions):
            identity = (public.dataset_session_id, public.public_subject_id)
            stats = histories.setdefault(identity, OpponentStats())
            rows.append(HistoryAwareExample(public, history_features_from_stats(stats)))
            stats.observe(decision)
    return tuple(rows)


def causal_history_for_decisions(
    decisions: Iterable[ObservedDecision],
) -> tuple[OpponentHistoryFeatures, ...]:
    histories: dict[tuple[str, str], OpponentStats] = {}
    rows = []
    for decision in decisions:
        identity = (decision.hand_key.session_id, decision.public_subject_id)
        stats = histories.setdefault(identity, OpponentStats())
        rows.append(history_features_from_stats(stats))
        stats.observe(decision)
    return tuple(rows)
