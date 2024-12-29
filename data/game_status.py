from dataclasses import dataclass

@dataclass
class GameStatus:
    num_players = 0
    bets = list()
    pot = 0
    cards = list()