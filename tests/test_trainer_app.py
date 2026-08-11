import importlib
import unittest

from poker_ai.holdem import Call, RaiseTo
from poker_ai.trainer_app import (
    create_example_session,
    create_new_hand_session,
    parse_action_script,
    parse_weighted_range,
)


class TrainerAppSmokeTests(unittest.TestCase):
    def test_import_has_no_streamlit_runtime_side_effect(self):
        module = importlib.import_module("poker_ai.trainer_app")
        self.assertTrue(callable(module.main))

    def test_session_factories_and_example_integration(self):
        self.assertEqual(create_new_hand_session(4, 100, 2, 9).current_actor, "P4")
        session = create_example_session()
        self.assertEqual(session.current_actor, "BTN")
        self.assertEqual(tuple(map(str, session.game.board)), ("Qd", "8c", "4s"))
        self.assertEqual(session.game.pot, 30)
        self.assertEqual(session.available_actions().call_amount, 12)

        original = session.game.internal_state
        session.undo()
        self.assertEqual(session.current_actor, "BB")
        session.redo()
        self.assertEqual(session.game.internal_state, original)
        branch = session.branch()
        branch.act(Call())
        self.assertEqual(session.position, 5)
        self.assertEqual(branch.position, 6)

    def test_non_json_builder_parsers(self):
        script = parse_action_script("BTN raise_to 6\nSB call")
        self.assertEqual(script, (("BTN", RaiseTo(6)), ("SB", Call())))
        weighted = parse_weighted_range("AhKh:1\n8s8d:0.5")
        self.assertEqual(len(weighted.combos), 2)
        self.assertIsNone(parse_weighted_range(""))


if __name__ == "__main__":
    unittest.main()
