import json
import unittest

from poker_ai.holdem import (
    BetTo,
    Call,
    Check,
    PlayerObservation,
    RaiseTo,
    ScenarioBuilder,
    TableConfig,
)
from poker_ai.training import (
    PolicyConfig,
    PolicyKind,
    TrainingSession,
    action_from_dict,
    action_to_dict,
)


def example_session() -> TrainingSession:
    config = TableConfig(("BTN", "SB", "BB"), (200, 200, 200), 1, 2, 0)
    builder = (
        ScenarioBuilder(config)
        .set_hole_cards("BTN", "As Qs")
        .set_board_runout("Qd 8c 4s")
        .action("BTN", RaiseTo(6))
        .action("SB", Call())
        .action("BB", Call())
        .action("SB", Check())
    )
    return TrainingSession.from_scenario(builder, human_players={"BTN", "BB"})


class RecordingPolicy:
    observations = []

    def decide(self, observation, legal_actions):
        type(self).observations.append(observation)
        return Check() if legal_actions.can_check else Call()


class TrainingSessionTests(unittest.TestCase):
    def test_undo_redo_and_goto_replay_exact_state(self):
        session = example_session()
        states = []
        for position in range(len(session.timeline) + 1):
            session.goto_action(position)
            states.append(session.game.internal_state)

        for position in reversed(range(len(states))):
            session.goto_action(position)
            self.assertEqual(session.game.internal_state, states[position])
        session.goto_action(3)
        session.undo()
        self.assertEqual(session.game.internal_state, states[2])
        session.redo()
        self.assertEqual(session.game.internal_state, states[3])

    def test_branch_is_independent_and_discards_no_parent_future(self):
        session = example_session()
        parent_timeline = session.timeline
        branch = session.branch(3)
        branch.act(BetTo(6))

        self.assertEqual(session.timeline, parent_timeline)
        self.assertEqual(session.position, 4)
        self.assertEqual(len(branch.timeline), 4)
        self.assertNotEqual(branch.timeline, session.timeline)

    def test_policy_receives_only_player_observation_and_one_step_is_exact(self):
        RecordingPolicy.observations.clear()
        config = TableConfig(("Human", "Bot"), (100, 100), 1, 2, 0)
        session = TrainingSession.new_hand(config, human_players={"Human"})
        session.act(Call())
        session.set_policy("Bot", RecordingPolicy())

        before = session.position
        step = session.next_policy_action()
        self.assertEqual(session.position, before + 1)
        self.assertEqual(step.player_id, "Bot")
        self.assertEqual(len(RecordingPolicy.observations), 1)
        self.assertIsInstance(RecordingPolicy.observations[0], PlayerObservation)

    def test_auto_play_stops_at_human_or_terminal(self):
        config = TableConfig(("Human", "Bot1", "Bot2"), (40, 40, 40), 1, 2, 0)
        session = TrainingSession.new_hand(
            config,
            human_players={"Human"},
            policy_configs={
                "Bot1": PolicyConfig(PolicyKind.CHECK_CALL),
                "Bot2": PolicyConfig(PolicyKind.RANDOM_LEGAL, 7),
            },
        )
        self.assertTrue(session.needs_human_action)
        session.act(Call())
        steps = session.auto_play_until_human()
        self.assertGreaterEqual(len(steps), 1)
        self.assertTrue(session.needs_human_action or session.game.is_terminal)

    def test_versioned_json_round_trip_preserves_replay(self):
        session = example_session()
        session.goto_action(2)
        encoded = session.to_json()
        restored = TrainingSession.from_json(encoded)

        self.assertEqual(restored.to_dict(), session.to_dict())
        self.assertEqual(restored.game.internal_state, session.game.internal_state)
        self.assertEqual(json.loads(encoded)["schema_version"], 1)
        bad = json.loads(encoded)
        bad["schema_version"] = 999
        with self.assertRaisesRegex(ValueError, "unsupported schema_version"):
            TrainingSession.from_dict(bad)

    def test_explicit_action_schema_round_trip_and_validation(self):
        action = RaiseTo(17)
        self.assertEqual(action_from_dict(action_to_dict(action)), action)
        with self.assertRaises(ValueError):
            action_from_dict({"type": "raise_to", "amount": True})
        with self.assertRaises(ValueError):
            action_from_dict({"type": "call", "amount": 1})


if __name__ == "__main__":
    unittest.main()
