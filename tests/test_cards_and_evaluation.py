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
    def test_every_hand_category(self):
        examples = {
            "high card": "As Jd 9c 6h 3s",
            "one pair": "As Ad 9c 6h 3s",
            "two pair": "As Ad 9c 9h 3s",
            "three of a kind": "As Ad Ac 6h 3s",
            "straight": "2c 3d 4h 5s 6c",
            "flush": "2h 5h 8h Jh Kh",
            "full house": "Qc Qd Qh 2s 2c",
            "four of a kind": "Qc Qd Qh Qs 2c",
            "straight flush": "9s Ts Js Qs Ks",
        }
        for expected, cards in examples.items():
            with self.subTest(expected):
                self.assertEqual(evaluate_five(cards).name, expected)

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

    def test_board_plays_and_exact_tie(self):
        first = evaluate_holdem("2c 3d As Ks Qs Js Ts")
        second = evaluate_holdem("4c 5d As Ks Qs Js Ts")
        self.assertEqual(first, second)
        self.assertEqual(first.name, "straight flush")

    def test_one_hole_card_plays(self):
        result = evaluate_holdem("As 2d Ah Kc Qd Js 3c")
        self.assertEqual(result.name, "one pair")
        self.assertEqual(result.kickers, (14, 13, 12, 11))

    def test_both_hole_cards_play(self):
        result = evaluate_holdem("As Ad 2c 3d 7h 9s Jc")
        self.assertEqual(result.name, "one pair")
        self.assertEqual(result.kickers, (14, 11, 9, 7))

    def test_kicker_comparison(self):
        ace_kicker = evaluate_holdem("As Qd Qc 8h 7s 4c 2d")
        king_kicker = evaluate_holdem("Ks Jd Qc 8h 7s 4c 2d")
        self.assertGreater(ace_kicker, king_kicker)

    def test_quads_selected_over_full_house_from_seven(self):
        result = evaluate_holdem("As Ah Ad Ac Kd Ks 2c")
        self.assertEqual(result.name, "four of a kind")
        self.assertEqual(result.kickers, (14, 13))


if __name__ == "__main__":
    unittest.main()
