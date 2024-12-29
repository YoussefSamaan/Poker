# class Player:
#     def __init__(self, name, card1, card2, action_decider_fn, start_money=float("inf")):
#         """
#         :param name: Unique identifier for the player (str).
#         :param card1, card2: The two hole cards (None if not yet dealt).
#         :param action_decider_fn: A function that decides action based on the game_state.
#         :param start_money: Initial chip count for the player.
#         """
#         self.name = name
#         self.card1 = card1
#         self.card2 = card2
#         self.chips = start_money
#         self.decider_fn = action_decider_fn
#
#     def get_name(self):
#         return self.name
#
#     def get_cards(self):
#         """
#         :return: Tuple of the player's two hole cards (Card, Card).
#         """
#         return (self.card1, self.card2)
#
#     def receive_card(self, card):
#         """
#         Assign a card to one of the player's empty holes (card1, card2).
#         """
#         if self.card1 is None:
#             self.card1 = card
#         elif self.card2 is None:
#             self.card2 = card
#
#     def decide_action(self, game_state):
#         """
#         :param game_state: Dict describing the current betting round (pot, to_call, etc.).
#         :return: A string in one of these forms: 'fold', 'check', 'call', 'raise:X'
#         """
#         return self.decider_fn(game_state)
#
#     def place_bet(self, amount):
#         """
#         Deducts up to 'amount' from the player's chips (handle all-in scenario).
#         Returns the actual amount bet so the game can add it to the pot.
#         """
#         amount = min(amount, self.chips)
#         self.chips -= amount
#         return amount
#
#     def reset_for_new_hand(self):
#         """
#         Clear the hole cards before the start of a new hand.
#         (Everything else regarding the player's bet for the round
#          can be tracked in the poker game logic if desired.)
#         """
#         self.card1 = None
#         self.card2 = None
#
#     def __repr__(self):
#         """
#         For debugging/logging: Shows the player's name, cards, and chip count.
#         """
#         return f"{self.name}, Cards: {self.card1}, {self.card2}"


class Player:
    def __init__(self, name, card1, card2, action_decider_fn, start_money=float("inf")):
        """
        :param name: Unique identifier for the player (str).
        :param card1, card2: The two hole cards (None if not yet dealt).
        :param action_decider_fn: A function that decides action based on the game_state.
        :param start_money: Initial chip count for the player.
        """
        self._name = name
        self._card1 = card1
        self._card2 = card2
        self._chips = start_money
        self._decider_fn = action_decider_fn
        self._folded = False  # Track if the player has folded in the current hand

    def get_name(self):
        return self._name

    def get_cards(self):
        """
        :return: Tuple of the player's two hole cards (Card, Card).
        """
        # return self._card1.copy(), self._card2.copy()
        return self._card1, self._card2

    def receive_card(self, card):
        """
        Assign a card to one of the player's empty holes (card1, card2).
        """
        if self._card1 is None:
            self._card1 = card
        elif self._card2 is None:
            self._card2 = card

    def decide_action(self, game_state):
        """
        :param game_state: Dict describing the current betting round (pot, to_call, etc.).
        :return: A string in one of these forms: 'fold', 'check', 'call', 'raise:X'
        """
        # If this player has already folded, no further actions are taken
        if self._folded:
            return "fold"
        return self._decider_fn(game_state)

    def place_bet(self, amount):
        """
        Deducts up to 'amount' from the player's chips (handle all-in scenario).
        Returns the actual amount bet so the game can add it to the pot.
        """
        amount = min(amount, self._chips)
        self._chips -= amount
        return amount

    def fold_hand(self):
        """
        Marks the player's state as folded and discards their hole cards.
        """
        self._folded = True
        self._card1 = None
        self._card2 = None

    def reset_for_new_hand(self):
        """
        Clear the hole cards before the start of a new hand and reset folded state.
        """
        self._card1 = None
        self._card2 = None
        self._folded = False

    def get_chips(self):
        return self._chips

    def __repr__(self):
        """
        For debugging/logging: Shows the player's name, cards, and chip count.
        """
        return f"{self._name}, Cards: {self._card1}, {self._card2}, Folded: {self._folded}"
