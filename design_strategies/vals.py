#############################
# ActionDecider Examples
#############################

def random_decider(game_state):
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
    options = ["fold", "call", "check", "raise:10", "raise:20"]
    return random.choice(options)

def always_call_decider(game_state):
    """
    Always calls if there's a bet, else checks.
    """
    if game_state['current_bet'] > 0:
        return "call"
    else:
        return "check"

def tight_decider(game_state):
    """
    Folds if there's a significant bet, otherwise calls or checks.
    """
    if game_state['current_bet'] > 50:
        return "fold"
    if game_state['current_bet'] > 0:
        return "call"
    else:
        return "check"