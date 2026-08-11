import unittest

from poker_ai.holdem import BetTo, Call, Check, RaiseTo, ScenarioBuilder, TableConfig
from poker_ai.ranges import WeightedRange
from poker_ai.training import (
    HeadsUpAnalysisRequired,
    TrainingSession,
    analyze_current_decision,
    candidate_sizings,
    decision_context,
)


def heads_up_decision() -> TrainingSession:
    config = TableConfig(("Hero", "Villain"), (100, 100), 1, 2, 0)
    builder = (
        ScenarioBuilder(config)
        .set_hole_cards("Hero", "As Qs")
        .set_hole_cards("Villain", "Kh Kd")
        .action("Hero", RaiseTo(6))
        .action("Villain", RaiseTo(10))
    )
    return TrainingSession.from_scenario(builder)


class TrainingAnalysisTests(unittest.TestCase):
    def test_context_and_raise_cost_adapter_use_engine_semantics(self):
        session = heads_up_decision()
        context = decision_context(session.game, "Hero")
        self.assertEqual(context.pot, 16)
        self.assertEqual(context.to_call, 4)
        self.assertEqual(context.required_equity, 0.2)

        candidates = candidate_sizings(context)
        targets = [candidate.target_to for candidate in candidates]
        self.assertEqual(targets, [14, 17, 20, 25, 30, 40, 100])
        self.assertEqual(
            [candidate.decision_cost for candidate in candidates],
            [8, 11, 14, 19, 24, 34, 94],
        )

    def test_analysis_uses_explicit_weighted_range_and_is_not_labeled_gto(self):
        session = heads_up_decision()
        opponent_range = WeightedRange.from_mapping({"KhKd": 2.0})
        result = analyze_current_decision(
            session.game,
            "Hero",
            opponent_range=opponent_range,
            fold_equity=0.25,
            samples=250,
            seed=4,
        )
        self.assertIn("not a GTO", result.model_label)
        self.assertAlmostEqual(result.context.required_equity, 0.2)
        self.assertTrue(result.candidate_values)
        self.assertTrue(all(value.regret >= 0 for value in result.candidate_values))

    def test_multiway_context_is_valid_but_baseline_fails_honestly(self):
        config = TableConfig(("BTN", "SB", "BB"), (200, 200, 200), 1, 2, 0)
        builder = (
            ScenarioBuilder(config)
            .set_hole_cards("BTN", "As Qs")
            .set_board_runout("Qd 8c 4s")
            .action("BTN", RaiseTo(6))
            .action("SB", Call())
            .action("BB", Call())
            .action("SB", Check())
            .action("BB", BetTo(12))
        )
        session = TrainingSession.from_scenario(builder)
        context = decision_context(session.game, "BTN")
        self.assertEqual(context.opponent_ids, ("SB", "BB"))
        with self.assertRaises(HeadsUpAnalysisRequired):
            analyze_current_decision(session.game, "BTN", samples=100)


if __name__ == "__main__":
    unittest.main()
