import unittest

from poker_ai.cards import Card, full_deck, parse_cards, shuffled_deck
from poker_ai.evaluation import evaluate_five, evaluate_holdem


class CardTests(unittest.TestCase):
    def test_standard_deck_is_complete_and_unique(self):
        deck = full_deck()
        self.assertEqual(len(deck), 52)
        self.assertEqual(len(set(deck)), 52)
        self.assertEqual(sum(card.rank == "J" for card in deck), 4)

    def test_unicode_and_ten_notation(self):
        self.assertEqual(Card.parse("A♠"), Card("A", "s"))
        self.assertEqual(Card.parse("10h"), Card("T", "h"))

    def test_seeded_shuffle_is_reproducible(self):
        self.assertEqual(shuffled_deck(17), shuffled_deck(17))

    def test_duplicate_cards_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "duplicate"):
            parse_cards("As As")


class EvaluationTests(unittest.TestCase):
    def test_category_order(self):
        straight = evaluate_five("2c 3d 4h 5s 6c")
        flush = evaluate_five("2h 5h 8h Jh Kh")
        full_house = evaluate_five("Qc Qd Qh 2s 2c")
        self.assertLess(straight, flush)
        self.assertLess(flush, full_house)

    def test_wheel_is_five_high(self):
        wheel = evaluate_five("As 2d 3c 4h 5s")
        six_high = evaluate_five("2s 3d 4c 5h 6s")
        self.assertEqual(wheel.kickers, (5,))
        self.assertLess(wheel, six_high)

    def test_best_five_of_seven(self):
        result = evaluate_holdem("As Ah Ad Kc Kd 2s 3c")
        self.assertEqual(result.name, "full house")
        self.assertEqual(result.kickers, (14, 13))


if __name__ == "__main__":
    unittest.main()
