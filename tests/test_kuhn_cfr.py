import unittest

from poker_ai.cfr import KuhnCFR


class KuhnCFRTests(unittest.TestCase):
    def test_seeded_training_converges_to_known_game_value(self):
        result = KuhnCFR().train(iterations=60_000, seed=7)
        self.assertEqual(len(result.strategy), 12)
        self.assertAlmostEqual(result.exact_value, -1 / 18, delta=0.015)
        self.assertLess(result.nash_conv, 0.03)

    def test_training_is_reproducible(self):
        first = KuhnCFR().train(iterations=2_000, seed=3)
        second = KuhnCFR().train(iterations=2_000, seed=3)
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
