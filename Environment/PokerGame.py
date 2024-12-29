from enum import Enum
import random

from Environment.DeckOfCards import DeckOfCards
from Environment.Player import Player
from Environment.hand_evaluator import HandEvaluator


class BettingRound(Enum):
    PRE_FLOP = 1
    FLOP = 2
    TURN = 3
    RIVER = 4
    SHOWDOWN = 5

class PokerGame:
    def __init__(self, players,  hand_evaluator, num_of_deck=1,
                 small_blind=10, big_blind=20):
        """
        :param players: list of Player objects
        :param deck: a DeckOfCards instance
        :param hand_evaluator: a HandEvaluator instance
        :param small_blind: the amount for the small blind
        :param big_blind: the amount for the big blind
        """
        self.players = players
        self.num_of_deck = num_of_deck
        self.deck = DeckOfCards(num_of_deck)
        self.evaluator = hand_evaluator

        # Simple blind amounts
        self.small_blind = small_blind
        self.big_blind = big_blind

        self.community_cards = []
        self.pot = 0
        self.current_betting_round = BettingRound.PRE_FLOP

        # Track how much each player has bet in the current round
        self.current_bets = {player.get_name(): 0 for player in self.players}

        # TODO: For more realism, you might rotate the dealer button each hand
        #  but here we'll just treat player[0] as small blind, player[1] as big blind
        #  in every hand, for demonstration.
        self.sb_player_index = -1

    def start_new_hand(self):
        """
        Reset state for a new hand. Shuffle deck, clear pot, deal new hole cards, post blinds, etc.
        """
        self.deck = DeckOfCards(self.num_of_deck)
        self.sb_player_index = (self.sb_player_index + 1) % len(self.players)
        self.pot = 0
        self.community_cards = []
        self.current_betting_round = BettingRound.PRE_FLOP

        # Reset players for new hand
        for player in self.players:
            player.reset_for_new_hand()

        # Clear previous bets
        self.current_bets = {player.get_name(): 0 for player in self.players}

        # Deal 2 hole cards each
        for _ in range(2):
            for player in self.players:
                card = self.deck.deal()
                player.receive_card(card)

        # Collect blinds (player[0] -> small blind, player[1] -> big blind) if there are at least 2 players
        self.post_blinds()

    def _player_places_amount(self, player: Player, amount: float):
        if player.get_chips() < amount:
            raise ValueError(f"Player {player.get_name()} does not have enough chips to bet {amount}")
        else:
            amount = player.place_bet(amount)
            self.current_bets[player.get_name()] += amount
            self.pot += amount
            return amount

    def post_blinds(self):
        """
        Deduct blinds from the first two players in the list for demonstration.
        """
        if len(self.players) < 2:
            return  # Can't post blinds if we have fewer than 2 players


        sb_player_index = self.sb_player_index
        bb_player_index = (sb_player_index + 1) % len(self.players)

        small_blind_player = self.players[sb_player_index]
        big_blind_player = self.players[bb_player_index]

        self._player_places_amount(small_blind_player, self.small_blind)

        self._player_places_amount(big_blind_player, self.big_blind)

    def proceed_to_next_betting_round(self):
        """
        Advance to the next stage in the betting cycle.
        """
        if self.current_betting_round == BettingRound.PRE_FLOP:
            self.current_betting_round = BettingRound.FLOP
            self.__deal_flop()
        elif self.current_betting_round == BettingRound.FLOP:
            self.current_betting_round = BettingRound.TURN
            self.__deal_turn()
        elif self.current_betting_round == BettingRound.TURN:
            self.current_betting_round = BettingRound.RIVER
            self.__deal_river()
        elif self.current_betting_round == BettingRound.RIVER:
            self.current_betting_round = BettingRound.SHOWDOWN
        else:
            # Already at showdown
            pass

        # Reset each player's bet for the new betting round:
        self.current_bets = {player.get_name(): 0 for player in self.players}

        #TODO: move sb index

    def betting_round_actions(self):
        """
        A simplified loop that continues until all players have acted and no new raises occur.
        """
        # TODO: Determine who is first to act. Normally, after pre-flop, the first active player
        #  left of the dealer is next. For demonstration, we’ll just start from player[0].
        action_index = 0
        num_players = len(self.players)

        # We'll keep track of how many times we cycle through the players
        # without a raise. Once we make a full pass with no raises, we end the round.
        have_we_cycled_without_raise = False
        last_player_to_raise = None
        highest_bet = max(self.current_bets.values())  # The largest bet so far

        while not have_we_cycled_without_raise:
            have_we_cycled_without_raise = True
            # loop_start_index = action_index ??????????

            # We'll do a full cycle (one turn for each player), but we might break if everyone is folded or calls
            for _ in range(num_players):
                player = self.players[action_index]
                if not player._folded and player._chips > 0:

                    if player == last_player_to_raise:
                        break

                    current_player_bet = self.current_bets[player.get_name()]
                    to_call = highest_bet - current_player_bet

                    game_state = {
                        "pot": self.pot,
                        "to_call": to_call,
                        "player_chips": player._chips,
                        "community_cards": self.community_cards,
                        "player_cards": player.get_cards(),
                        "betting_round": self.current_betting_round.name
                    }

                    action_str = player.decide_action(game_state)
                    parsed_action = self.__parse_action(action_str, player, highest_bet)

                    # Update pot and bets
                    if parsed_action["type"] == "fold":
                        player.fold_hand()
                    elif parsed_action["type"] == "call":
                        amount_to_call = parsed_action["amount"]
                        bet_placed = player.place_bet(amount_to_call)
                        self.current_bets[player.get_name()] += bet_placed
                        self.pot += bet_placed
                    elif parsed_action["type"] == "check":
                        # No chips in
                        pass
                    elif parsed_action["type"] == "raise":
                        # This is the total bet (including call portion)
                        total_bet = parsed_action["total_bet"]
                        # The difference between total_bet and what the player already bet
                        additional = total_bet - self.current_bets[player.get_name()]
                        last_player_to_raise = player
                        if additional > 0:
                            placed = player.place_bet(additional)
                            self.current_bets[player.get_name()] += placed
                            self.pot += placed
                            # If the total bet is now the highest, we set it as the new standard
                            if total_bet > highest_bet:
                                highest_bet = total_bet
                                have_we_cycled_without_raise = False  # We had a raise

                # Move to the next player
                action_index = (action_index + 1) % num_players

            # After we go around once, if a raise occurred, we do another pass.
            # The condition is checked in the while loop top.

    def __parse_action(self, action_str, player, highest_bet):
        """
        Parse the player's action string into a dict describing the action.
        e.g. "fold", "check", "call", "raise:20"
        """
        action_str = action_str.lower().strip()
        current_player_bet = self.current_bets[player.get_name()]
        to_call = highest_bet - current_player_bet

        if action_str == "fold":
            return {"type": "fold"}
        elif action_str == "check":
            # Checking is only valid if there's nothing to call
            if to_call > 0:
                # This would be an illegal move; interpret as call for simplicity
                return {"type": "call", "amount": to_call}
            return {"type": "check"}
        elif action_str == "call":
            if to_call < 0:
                to_call = 0
            return {"type": "call", "amount": to_call}
        elif action_str.startswith("raise:"):
            # Format raise:X
            try:
                raise_amount = int(action_str.split(":")[1])
            except ValueError:
                raise_amount = 0
            # The total bet includes the to_call
            total_bet = raise_amount + highest_bet
            # In a real game, you'd also limit total_bet by player's chips, etc.
            return {"type": "raise", "total_bet": total_bet}
        else:
            # Default fallback, interpret as "check"
            if to_call > 0:
                return {"type": "call", "amount": to_call}
            return {"type": "check"}

    def __deal_flop(self):
        """
        Deal 3 community cards from the deck.
        """
        self.community_cards.extend([self.deck.deal() for _ in range(3)])

    def __deal_turn(self):
        """
        Deal 1 turn card.
        """
        self.community_cards.append(self.deck.deal())

    def __deal_river(self):
        """
        Deal 1 river card.
        """
        self.community_cards.append(self.deck.deal())

    def showdown(self):
        """
        Compare the final 7-card hands. Award the pot to the winner(s).
        For simplicity, only one winner is chosen. No side pots, no ties.
        """
        active_players = [p for p in self.players if not p._folded]

        # Evaluate each player's best 5-card combination out of the 7
        best_ranks = {}
        for player in active_players:
            # In a real game, you'd pick the best 5 of the 7.
            # We'll do a naive approach: last 3 community cards + hole cards
            # purely to demonstrate using the HandEvaluator’s get_hand_ranking(5-card).
            if len(self.community_cards) < 5:
                # If the community has fewer than 5 cards in this naive approach, skip
                continue
            dummy_5_cards = self.community_cards[-3:] + list(player.get_cards())
            rank = self.evaluator.get_hand_ranking(dummy_5_cards)
            best_ranks[player.get_name()] = rank

        if not best_ranks:
            print("No active players at showdown. Pot remains unawarded.")
            return

        # Suppose lower rank is better
        winner = min(best_ranks, key=best_ranks.get)

        # Award entire pot to winner
        for player in self.players:
            if player.get_name() == winner:
                player._chips += self.pot

        print(f"The winner is {winner} with rank {best_ranks[winner]}!")
        self.pot = 0

    def play_hand(self):
        """
        Simulate a single hand from start to finish:
          1) Start new hand (deal hole cards, post blinds)
          2) Pre-Flop betting
          3) Deal Flop + betting
          4) Deal Turn + betting
          5) Deal River + betting
          6) Showdown
        """
        self.start_new_hand()
        self.print_status()

        # Pre-Flop
        self.betting_round_actions()
        self.proceed_to_next_betting_round()
        self.print_status()

        # Flop
        self.betting_round_actions()
        self.proceed_to_next_betting_round()
        self.print_status()

        # Turn
        self.betting_round_actions()
        self.proceed_to_next_betting_round()
        self.print_status()

        # River
        self.betting_round_actions()
        self.proceed_to_next_betting_round()
        self.print_status()

        # Showdown
        self.showdown()
        self.print_status()

    def print_status(self):
        """
        Prints a human-readable status of the game: pot, community cards,
        and each player's chips & fold status.
        """
        print("---- GAME STATUS ----")
        print(f"Betting Round: {self.current_betting_round.name}")
        print(f"Pot: {self.pot}")
        print("Community Cards:", [str(card) for card in self.community_cards])
        for p in self.players:
            folded_str = "(folded)" if p._folded else ""
            print(f"Player {p.get_name()}: {p._chips} chips {folded_str}")
        print("---------------------")

    def get_status(self):
        """
        Returns a dictionary summarizing the current game state:
          - betting_round
          - pot
          - community_cards
          - players: list of {name, chips, folded}
        """
        status = {
            "betting_round": self.current_betting_round.name,
            "pot": self.pot,
            "community_cards": [str(c) for c in self.community_cards],
            "players": []
        }
        for p in self.players:
            status["players"].append({
                "name": p.get_name(),
                "chips": p.chips,
                "folded": p.folded
            })
        return status

if __name__ == "__main__":
    # Set up players
    def always_call_decider(game_state=None):
        """
        A random decider for demonstration.
        game_state might look like:
            {
                'current_bet': int,
                'pot': int,
                'player_chips': int,
                'hand': [Card, Card],  # hole cards
                'community_cards': [...],
                ...
            }
        Returns one of: "fold", "call", "check", "raise:X"
        """
        options = ["call", "raise:20"] # fold  "check", "raise:10",
        return random.choice(options)

    players = [
        Player("Alice", None, None, always_call_decider, start_money=1000),
        Player("Bob", None, None, always_call_decider, start_money=1000),
    ]

    # Create a deck (1 standard deck of 52 cards)
    deck = DeckOfCards(1)
    # Create a hand evaluator
    evaluator = HandEvaluator(f_path="../data/hand_rankings.csv")
    # Create the poker game
    game = PokerGame(players, evaluator)

    # Play one hand
    game.play_hand()

    game.play_hand()


    # Print out the players' chip counts after the hand
    for p in players:
        print(f"{p.get_name()} has {p._chips} chips.")
