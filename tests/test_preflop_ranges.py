import unittest

from poker_ai.ranges import PreflopRange, RANGE_RANKS


class PreflopRangeTests(unittest.TestCase):
    def test_standard_class_combo_counts(self):
        expected = {"AA": 6, "AKs": 4, "AKo": 12, "AK": 16, "TT+": 30, "22+": 78}
        for expression, count in expected.items():
            with self.subTest(expression=expression):
                self.assertEqual(
                    PreflopRange.parse(expression).stats().raw_combo_count, count
                )

    def test_plus_semantics_keep_first_rank_fixed(self):
        self.assertEqual(len(PreflopRange.parse("A2s+").class_weights), 12)
        self.assertEqual(len(PreflopRange.parse("ATs+").class_weights), 4)
        self.assertEqual(len(PreflopRange.parse("AJo+").class_weights), 3)
        self.assertEqual(len(PreflopRange.parse("KTs+").class_weights), 3)

    def test_mixed_weighted_expansion_and_blockers(self):
        value = PreflopRange.parse("QQ+:1, AKs:0.5, AKo:0.25")
        stats = value.stats(("As", "Qs", "Qd", "8c", "4s"))
        self.assertEqual(stats.raw_combo_count, 34)
        self.assertLess(stats.legal_combo_count, stats.raw_combo_count)
        self.assertAlmostEqual(value.stats().coverage, 34 / 1326)
        self.assertEqual(
            stats.blocked_combo_count, stats.raw_combo_count - stats.legal_combo_count
        )
        self.assertEqual(stats.raw_preflop_coverage, 34 / 1326)
        self.assertAlmostEqual(
            stats.legal_fraction_of_original_range,
            stats.legal_combo_count / stats.raw_combo_count,
        )
        self.assertEqual(stats.legal_total_weight, stats.total_weight)
        weights = {combo.weight for combo in value.to_weighted_range().combos}
        self.assertEqual(weights, {1.0, 0.5, 0.25})

    def test_invalid_and_overlapping_syntax_is_rejected(self):
        for expression in ("KA", "AKx", "AAs", "ATs++", "AKs:0", "QQ+,AA"):
            with self.subTest(expression=expression):
                with self.assertRaises(ValueError):
                    PreflopRange.parse(expression)

    def test_matrix_uses_conventional_pair_suited_offsuit_cells(self):
        value = PreflopRange.parse("AA:1, AKs:0.5, AKo:0.25")
        matrix = value.matrix()
        ace, king = RANGE_RANKS.index("A"), RANGE_RANKS.index("K")
        self.assertEqual(matrix[ace][ace], 1.0)
        self.assertEqual(matrix[ace][king], 0.5)
        self.assertEqual(matrix[king][ace], 0.25)
        self.assertIsNone(matrix[king][king])


if __name__ == "__main__":
    unittest.main()
