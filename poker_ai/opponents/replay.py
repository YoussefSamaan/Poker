from __future__ import annotations

from ..training.session import TrainingSession
from .observation import (
    HandKey,
    ObservedDecision,
    ObserverContext,
    observe_decision,
)


def observed_decisions_from_session(
    session: TrainingSession, *, hand_index: int = 0
) -> tuple[ObservedDecision, ...]:
    """Replay the recorded prefix and capture only information public at each step."""
    replay = TrainingSession.from_dict(session.to_dict())
    timeline = replay.timeline[: session.position]
    replay.goto_action(0)
    hand_key = HandKey(session.session_id, hand_index)
    decisions = []
    for item in timeline:
        actor = replay.current_actor
        if actor != item.player_id:
            raise ValueError("recorded timeline actor does not match reconstructed game")
        observation = replay.game.observation_for(actor)
        legal = replay.game.legal_actions(actor)
        decisions.append(
            observe_decision(hand_key, actor, observation, legal, item.action)
        )
        replay.act(item.action)
    return tuple(decisions)


def observer_context_from_session(
    session: TrainingSession, observer_id: str, *, hand_index: int = 0
) -> ObserverContext:
    observation = session.game.observation_for(observer_id)
    return ObserverContext(
        observer_id,
        HandKey(session.session_id, hand_index),
        observation.hole_cards,
    )
