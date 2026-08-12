import unittest

from poker_ai.holdem import (
    BetTo,
    Call,
    Check,
    RaiseTo,
    ScenarioBuilder,
    TableConfig,
)
from poker_ai.ranges import PreflopRange
from poker_ai.training import (
    TrainingSession,
    analyze_showdown_baseline,
    board_features,
    capture_decision_review,
    compare_ranges,
    describe_current_hand,
    player_table_view,
)


def three_way_session() -> TrainingSession:
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
    return TrainingSession.from_scenario(builder)


class CoachTests(unittest.TestCase):
    def test_public_view_contains_no_opponent_cards_or_deck(self):
        session = three_way_session()
        view = player_table_view(session.game.observation_for("BTN"))
        self.assertEqual(tuple(map(str, view.seats[0].cards)), ("As", "Qs"))
        self.assertTrue(all(seat.cards is None for seat in view.seats[1:]))
        self.assertFalse(hasattr(view, "remaining_deck"))
        self.assertEqual(tuple(map(str, view.board)), ("Qd", "8c", "4s"))

    def test_board_features_and_current_hand(self):
        session = three_way_session()
        context = session.game.observation_for("BTN")
        features = board_features(context.board)
        self.assertEqual(features.suit_texture, "rainbow")
        self.assertFalse(features.paired)
        self.assertEqual(
            describe_current_hand(context.hole_cards, context.board),
            "one pair — queens",
        )
        self.assertEqual(
            describe_current_hand(context.hole_cards, ()), "preflop starting hand"
        )
        self.assertTrue(
            board_features(
                tuple(__import__("poker_ai").parse_cards("Qs Qh 8c 8d"))
            ).double_paired
        )
        self.assertEqual(
            board_features(
                tuple(__import__("poker_ai").parse_cards("As 8s 2s"))
            ).suit_texture,
            "monotone",
        )
        self.assertEqual(
            board_features(
                tuple(__import__("poker_ai").parse_cards("As 8s 2d"))
            ).suit_texture,
            "two-tone",
        )

    def test_multiway_call_fold_and_unsupported_raise(self):
        session = three_way_session()
        dead = (
            session.game.observation_for("BTN").hole_cards
            + session.game.observation_for("BTN").board
        )
        range_value = PreflopRange.parse("22+,A2s+,KTs+").to_weighted_range(dead)
        result = analyze_showdown_baseline(
            session.game,
            "BTN",
            {"SB": range_value, "BB": range_value},
            samples=300,
            seed=2,
        )
        self.assertEqual(
            next(action.ev for action in result.actions if action.action == "fold"), 0
        )
        self.assertIsInstance(
            next(action.ev for action in result.actions if action.action == "call"),
            float,
        )
        self.assertIsNone(
            next(
                action.ev for action in result.actions if action.action == "bet / raise"
            )
        )
        self.assertTrue(result.relevant_pots)

    def test_check_sensitivity_and_decision_review(self):
        config = TableConfig(("Hero", "Villain"), (100, 100), 1, 2, 0)
        builder = (
            ScenarioBuilder(config)
            .set_hole_cards("Hero", "As Qs")
            .set_board_runout("Qd 8c 4s")
            .action("Hero", Call())
            .action("Villain", Check())
        )
        session = TrainingSession.from_scenario(builder)
        dead = session.game.observation_for("Villain").board
        ranges = {
            "pairs": PreflopRange.parse("22+").to_weighted_range(dead),
            "broadway": PreflopRange.parse("ATs+,AJo+").to_weighted_range(dead),
        }
        compared = compare_ranges(session.game, "Villain", ranges, samples=200, seed=5)
        self.assertEqual([item.name for item in compared], ["pairs", "broadway"])
        analysis = analyze_showdown_baseline(
            session.game, "Villain", {"Hero": ranges["pairs"]}, samples=200, seed=1
        )
        self.assertIn("check", {action.action for action in analysis.actions})
        call_review = capture_decision_review(2, analysis, Check(), ("pairs",))
        self.assertIsNotNone(call_review.estimated_baseline_regret)
        raise_review = capture_decision_review(2, analysis, BetTo(4), ("pairs",))
        self.assertIsNone(raise_review.estimated_baseline_regret)
        self.assertIn("not evaluated", raise_review.note)


if __name__ == "__main__":
    unittest.main()
