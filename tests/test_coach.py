import unittest

from poker_ai.cards import parse_cards
from poker_ai.evaluation import HandRank
from poker_ai.holdem import (
    BetTo,
    Call,
    Check,
    RaiseTo,
    ScenarioBuilder,
    TableConfig,
)
from poker_ai.ranges import PreflopRange, WeightedRange
from poker_ai.training import (
    TrainingSession,
    analyze_showdown_baseline,
    board_features,
    capture_decision_review,
    compare_ranges,
    describe_current_hand,
    hero_payout_for_world,
    player_table_view,
    range_matrix_rows,
    sensitivity_rows,
)
from poker_ai.holdem.pots import build_side_pots
from poker_ai.multiway import ShowdownSampler
from poker_ai.training.coach import _describe_rank


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

    def test_informative_rank_descriptions(self):
        cases = (
            ("As Kd", "2c 4h 7s", "ace-high"),
            ("Ks Qd", "2c 4h 7s", "king-high"),
            ("Ts 9d", "6c 7h 8s", "straight — ten-high"),
            ("As 9s", "2s 4s 7s", "flush — ace-high"),
            ("9s 8s", "5s 6s 7s", "straight flush — nine-high"),
        )
        for hero, board, expected in cases:
            with self.subTest(expected=expected):
                cards = parse_cards(hero)
                self.assertEqual(
                    describe_current_hand((cards[0], cards[1]), parse_cards(board)),
                    expected,
                )
        # Six-high and five-high cannot occur as real five-card high-card hands
        # (the five distinct ranks would form a straight), but the formatter is
        # still total for evaluator-domain HandRank values.
        self.assertEqual(_describe_rank(HandRank(0, (6, 5, 3, 2, 1))), "six-high")
        self.assertEqual(_describe_rank(HandRank(0, (5, 4, 3, 2, 1))), "five-high")

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

    def test_side_pot_uses_one_world_and_exact_zero_uncertainty(self):
        config = TableConfig(("Hero", "Short", "Deep"), (100, 30, 100), 1, 2, 0)
        builder = (
            ScenarioBuilder(config)
            .set_hole_cards("Hero", "Kh Kd")
            .set_hole_cards("Short", "Ah Ad")
            .set_hole_cards("Deep", "Qh Qd")
            .set_board_runout("2c 3d 4h 8s 9c")
            .action("Hero", RaiseTo(30))
            .action("Short", Call())
            .action("Deep", Call())
            .action("Deep", Check())
            .action("Hero", Check())
            .action("Deep", Check())
            .action("Hero", Check())
            .action("Deep", BetTo(30))
        )
        session = TrainingSession.from_scenario(builder)
        result = analyze_showdown_baseline(
            session.game,
            "Hero",
            {
                "Short": WeightedRange.from_mapping({"AhAd": 1}),
                "Deep": WeightedRange.from_mapping({"QhQd": 1}),
            },
            exact=True,
        )
        call = next(action for action in result.actions if action.action == "call")
        self.assertEqual([pot.amount for pot in result.relevant_pots], [90, 60])
        self.assertEqual(call.ev, 30)
        self.assertEqual(call.standard_error, 0)
        self.assertEqual(call.confidence_interval_95, (30, 30))

    def test_multiple_side_pots_are_paid_from_one_complete_world(self):
        sampler = ShowdownSampler(
            "Kh Kd",
            "2c 3d 4h 8s 9c",
            [
                WeightedRange.from_mapping({"AhAd": 1}),
                WeightedRange.from_mapping({"QhQd": 1}),
                WeightedRange.from_mapping({"JhJd": 1}),
            ],
        )
        world = next(sampler.exact_worlds())
        pots = build_side_pots(
            {"Hero": 100, "Short": 20, "Middle": 50, "Deep": 100},
            {"Hero", "Short", "Middle", "Deep"},
        )
        self.assertEqual([pot.amount for pot in pots], [80, 90, 100])
        self.assertEqual(
            hero_payout_for_world(world, "Hero", ("Short", "Middle", "Deep"), pots),
            190,
        )
        self.assertEqual(tuple(map(str, world.opponent_hands[0])), ("Ad", "Ah"))

        blocker_sampler = ShowdownSampler(
            "Kh Kd",
            "2c 3d 4h 8s 9c",
            [
                WeightedRange.from_mapping({"AcAd": 1}),
                WeightedRange.from_mapping({"AcQh": 1, "JhTs": 1}),
            ],
        )
        blocker_worlds = tuple(blocker_sampler.exact_worlds())
        self.assertTrue(
            all(
                tuple(map(str, world.opponent_hands[1])) == ("Jh", "Ts")
                for world in blocker_worlds
            )
        )

    def test_range_matrix_and_sensitivity_view_rows(self):
        matrix_rows = range_matrix_rows(PreflopRange.parse("AA,AKs"))
        self.assertEqual(len(matrix_rows), 13)
        self.assertEqual(matrix_rows[0]["row"], "A")
        self.assertEqual(matrix_rows[0]["A"], 1)

        session = TrainingSession.from_scenario(
            ScenarioBuilder(TableConfig(("Hero", "V"), (20, 20), 1, 2, 0))
            .set_hole_cards("Hero", "As Qs")
            .action("Hero", Call())
            .action("V", Check())
        )
        comparison = compare_ranges(
            session.game,
            "V",
            {"Named by user": PreflopRange.parse("22+").to_weighted_range()},
            samples=100,
        )
        rows = sensitivity_rows(comparison)
        self.assertEqual(rows[0]["name"], "Named by user")
        self.assertIn("EV standard error", rows[0])


if __name__ == "__main__":
    unittest.main()
