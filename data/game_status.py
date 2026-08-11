from dataclasses import dataclass, field

@dataclass
class GameStatus:
    num_players: int = 0
    bets: list[float] = field(default_factory=list)
    pot: float = 0
    cards: list = field(default_factory=list)
