import random
import unittest

from poker_ai.cards import parse_cards
from poker_ai.multiway import MultiwayEquityCalculator, ShowdownSampler
from poker_ai.ranges import WeightedRange


class MultiwayEquityTests(unittest.TestCase):
    def test_asymmetric_overlap_uses_product_conditioned_distribution(self):
        first = WeightedRange.from_mapping({"AcAd": 1, "KcKd": 1})
        second = WeightedRange.from_mapping({"AcQh": 1, "JhTs": 1})
        sampler = ShowdownSampler("2c 3d", "4h 5s 9c", [first, second])
        rng = random.Random(71)
        samples = [sampler.sample_joint_hands(rng) for _ in range(9_000)]
        first_is_aces = sum(
            set(hands[0]) == set(parse_cards("Ac Ad")) for hands in samples
        )
        self.assertAlmostEqual(first_is_aces / len(samples), 1 / 3, delta=0.025)

    def test_joint_distribution_is_order_invariant(self):
        first = WeightedRange.from_mapping({"AcAd": 2, "KcKd": 1})
        second = WeightedRange.from_mapping({"AcQh": 3, "JhTs": 1})
        forward = MultiwayEquityCalculator(1).calculate(
            "2c 3d", "4h 5s 9c", [first, second], exact=False, samples=8_000, seed=8
        )
        reverse = MultiwayEquityCalculator(1).calculate(
            "2c 3d", "4h 5s 9c", [second, first], exact=False, samples=8_000, seed=9
        )
        self.assertAlmostEqual(forward.equity, reverse.equity, delta=0.025)

    def test_weighted_joint_marginal_uses_products_after_conditioning(self):
        first = WeightedRange.from_mapping({"AcAd": 4, "KcKd": 1})
        second = WeightedRange.from_mapping({"AcQh": 2, "JhTs": 1})
        # Legal product weights are A-D=4, B-C=2, B-D=1, so P(A)=4/7.
        sampler = ShowdownSampler("2c 3d", "4h 5s 9c", [first, second])
        rng = random.Random(19)
        samples = [sampler.sample_joint_hands(rng) for _ in range(10_000)]
        first_is_aces = sum(
            set(hands[0]) == set(parse_cards("Ac Ad")) for hands in samples
        )
        self.assertAlmostEqual(first_is_aces / len(samples), 4 / 7, delta=0.025)

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

    def test_exact_and_monte_carlo_agree_with_two_and_three_opponents(self):
        ranges = (
            WeightedRange.from_mapping({"AcAd": 2, "KcKd": 1}),
            WeightedRange.from_mapping({"AcQh": 3, "JhTs": 1}),
            WeightedRange.from_mapping({"KsQd": 1, "7h7d": 2}),
        )
        calculator = MultiwayEquityCalculator()
        for opponent_count in (2, 3):
            with self.subTest(opponents=opponent_count):
                exact = calculator.calculate(
                    "2c 3d", "4h 5s 9c", ranges[:opponent_count], exact=True
                )
                sampled = calculator.calculate(
                    "2c 3d",
                    "4h 5s 9c",
                    ranges[:opponent_count],
                    exact=False,
                    samples=12_000,
                    seed=27,
                )
                self.assertAlmostEqual(exact.equity, sampled.equity, delta=0.025)


if __name__ == "__main__":
    unittest.main()
