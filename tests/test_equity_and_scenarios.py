import unittest

from poker_ai.equity import EquityCalculator
from poker_ai.ranges import WeightedRange
from poker_ai.scenario import HeadsUpScenario, ScenarioAnalyzer


class EquityTests(unittest.TestCase):
    def test_river_nuts_have_full_equity(self):
        result = EquityCalculator().calculate(
            ("As", "Ks"),
            ("Qs", "Js", "Ts", "2d", "3c"),
            exact=True,
        )
        self.assertEqual(result.method, "exact")
        self.assertEqual(result.equity, 1.0)
        self.assertEqual(result.outcomes, 990)

    def test_community_royal_flush_always_ties(self):
        result = EquityCalculator().calculate(
            ("2c", "3d"),
            ("As", "Ks", "Qs", "Js", "Ts"),
            exact=True,
        )
        self.assertEqual(result.tie_probability, 1.0)
        self.assertEqual(result.equity, 0.5)

    def test_seeded_monte_carlo_is_reproducible(self):
        calculator = EquityCalculator(exact_threshold=1)
        first = calculator.calculate(("As", "Qh"), ("8c", "7d", "2s"), samples=400, seed=9)
        second = calculator.calculate(("As", "Qh"), ("8c", "7d", "2s"), samples=400, seed=9)
        self.assertEqual(first, second)
        self.assertGreater(first.standard_error, 0)

    def test_weighted_range_respects_blockers(self):
        opponent = WeightedRange.from_mapping({"AsKh": 3, "QcQd": 1})
        result = EquityCalculator().calculate(
            ("As", "2c"),
            ("3d", "4h", "5s", "9c", "Td"),
            opponent,
            exact=True,
        )
        self.assertEqual(result.outcomes, 1)


class ScenarioTests(unittest.TestCase):
    def test_call_ev_and_pot_odds_use_decision_point_units(self):
        scenario = HeadsUpScenario.from_text(
            hero="As Ks",
            board="Qs Js Ts 2d 3c",
            pot=18,
            to_call=12,
            hero_stack=100,
            villain_stack=100,
        )
        result = ScenarioAnalyzer().analyze(scenario, exact=True)
        self.assertAlmostEqual(result.pot_odds, 0.4)
        self.assertAlmostEqual(result.ev_for("call"), 18.0)
        self.assertEqual(result.recommended_action, "call")
        self.assertEqual(result.regret("fold"), 18.0)
        self.assertIn("not a GTO", result.model)

    def test_raise_ev_includes_explicit_fold_equity_assumption(self):
        scenario = HeadsUpScenario.from_text(
            hero="As Ks",
            board="Qs Js Ts 2d 3c",
            pot=18,
            to_call=12,
            hero_stack=100,
            villain_stack=100,
        )
        result = ScenarioAnalyzer().analyze(
            scenario,
            raise_costs=(30,),
            fold_equity=0.25,
            exact=True,
        )
        self.assertAlmostEqual(result.ev_for("raise_cost_30"), 31.5)

    def test_unmatched_raise_is_rejected(self):
        scenario = HeadsUpScenario.from_text(
            hero="As Ks",
            board="Qs Js Ts",
            pot=18,
            to_call=12,
            hero_stack=100,
            villain_stack=10,
        )
        with self.assertRaisesRegex(ValueError, "effective"):
            ScenarioAnalyzer().analyze(scenario, raise_costs=(30,), samples=10)


if __name__ == "__main__":
    unittest.main()
