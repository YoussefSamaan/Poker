import unittest

from Environment.DeckOfCards import Card, DeckOfCards
from Environment.Player import Player
from Environment.PokerGame import PokerGame
from Environment.hand_evaluator import HandEvaluator
from data.game_status import GameStatus


class LegacyRegressionTests(unittest.TestCase):
    def test_legacy_deck_has_all_52_cards(self):
        deck = DeckOfCards(1)
        self.assertEqual(len(deck.cards), 52)
        self.assertEqual(sum(card.value == "J" for card in deck.cards), 4)

    def test_legacy_showdown_uses_best_five_of_seven(self):
        alice = Player("Alice", Card("S", "A"), Card("S", "K"), lambda _: "check", 0)
        bob = Player("Bob", Card("H", "A"), Card("D", "A"), lambda _: "check", 0)
        evaluator = HandEvaluator("data/hand_rankings.csv")
        game = PokerGame((alice, bob), evaluator)
        game.community_cards = [
            Card("S", "Q"),
            Card("S", "J"),
            Card("S", "T"),
            Card("D", "2"),
            Card("C", "3"),
        ]
        game.pot = 20
        game.showdown()
        self.assertEqual(alice.get_chips(), 20)
        self.assertEqual(bob.get_chips(), 0)

    def test_game_status_lists_are_not_shared(self):
        first = GameStatus()
        second = GameStatus()
        first.cards.append("As")
        self.assertEqual(second.cards, [])


if __name__ == "__main__":
    unittest.main()
