import random
import unittest

from poker_ai.cards import parse_cards
from poker_ai.multiway import MultiwayEquityCalculator
from poker_ai.ranges import WeightedRange


class MultiwayEquityTests(unittest.TestCase):
    def test_unbeatable_hand_and_board_ties(self):
        calculator = MultiwayEquityCalculator()
        nuts = calculator.calculate(
            "As Ks",
            "Qs Js Ts 2d 3c",
            [
                WeightedRange.from_mapping({"2c2h": 1}),
                WeightedRange.from_mapping({"4c4h": 1}),
            ],
            exact=True,
        )
        self.assertEqual(nuts.equity, 1.0)
        three_way = calculator.calculate(
            "2c 3d",
            "As Ks Qs Js Ts",
            [
                WeightedRange.from_mapping({"4c5d": 1}),
                WeightedRange.from_mapping({"6c7d": 1}),
            ],
            exact=True,
        )
        self.assertAlmostEqual(three_way.equity, 1 / 3)
        four_way = calculator.calculate(
            "2c 3d",
            "As Ks Qs Js Ts",
            [
                WeightedRange.from_mapping({"4c5d": 1}),
                WeightedRange.from_mapping({"6c7d": 1}),
                WeightedRange.from_mapping({"8c9d": 1}),
            ],
            exact=True,
        )
        self.assertAlmostEqual(four_way.equity, 0.25)

    def test_seeded_monte_carlo_is_deterministic(self):
        ranges = [None, None]
        first = MultiwayEquityCalculator(1).calculate(
            "As Qs", "Qd 8c 4s", ranges, samples=300, seed=9
        )
        second = MultiwayEquityCalculator(1).calculate(
            "As Qs", "Qd 8c 4s", ranges, samples=300, seed=9
        )
        self.assertEqual(first, second)
        self.assertGreater(first.standard_error, 0)

    def test_conditional_sampling_never_overlaps(self):
        dead = parse_cards("As Qs Qd 8c 4s")
        ranges = tuple(WeightedRange.random(dead) for _ in range(5))
        hands = MultiwayEquityCalculator.sample_opponent_hands(
            ranges, dead, random.Random(4)
        )
        all_cards = dead + tuple(card for hand in hands for card in hand)
        self.assertEqual(len(all_cards), len(set(all_cards)))

    def test_blocked_and_mutually_impossible_ranges_fail(self):
        shared = WeightedRange.from_mapping({"AhKh": 1})
        with self.assertRaisesRegex(ValueError, "compatible"):
            MultiwayEquityCalculator().calculate(
                "As Qs", "Qd 8c 4s", [shared, shared], samples=2, exact=False
            )

    def test_weights_affect_sampling(self):
        board = "2c 3d 4h 8s 9c"
        heavy_aces = WeightedRange.from_mapping({"AhAd": 100, "QhJd": 1})
        heavy_weak = WeightedRange.from_mapping({"AhAd": 1, "QhJd": 100})
        calculator = MultiwayEquityCalculator(1)
        first = calculator.calculate("Kh Kd", board, [heavy_aces], samples=2000, seed=3)
        second = calculator.calculate(
            "Kh Kd", board, [heavy_weak], samples=2000, seed=3
        )
        self.assertLess(first.equity, second.equity)

    def test_exact_and_monte_carlo_agree_on_small_river_space(self):
        opponent = WeightedRange.from_mapping({"AhAd": 1, "QhJd": 1})
        calculator = MultiwayEquityCalculator()
        exact = calculator.calculate("Kh Kd", "2c 3d 4h 8s 9c", [opponent], exact=True)
        sampled = calculator.calculate(
            "Kh Kd",
            "2c 3d 4h 8s 9c",
            [opponent],
            exact=False,
            samples=5_000,
            seed=11,
        )
        self.assertAlmostEqual(exact.equity, sampled.equity, delta=0.03)


if __name__ == "__main__":
    unittest.main()
