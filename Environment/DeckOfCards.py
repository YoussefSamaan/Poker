from enum import Enum
import random

class DeckOfCards:
    def __init__(self, number_of_decks):
        self.cards = self._create_a_deck_of_cards(number_of_decks)
        self.shuffle()
        pass

    def shuffle(self):
        random.shuffle(self.cards)

    def deal(self):
        return self.cards.pop()

    def _create_a_deck_of_cards(self, number_of_decks):
        cards = []
        for _ in range(number_of_decks):
            cards.extend(self._get_a_deck_of_cards())
        return cards

    def _get_a_deck_of_cards(self):
        cards = []
        for suit in "HSCD":#Suits:
            for value in "23456789TQKA": #Values:
                cards.append(Card(suit, value))
        return cards

class Card:
    def __init__(self, suit, value):
        self.suit = suit
        self.value = value

    def __str__(self):
        return f"{self.value}{self.suit}"

    def get_suit(self):
        return self.suit

    def get_value(self):
        return self.value

    def __copy__(self):
        return Card(self.suit, self.value)