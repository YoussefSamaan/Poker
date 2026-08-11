import unittest

from poker_ai.cards import full_deck
from poker_ai.holdem import (
    BetTo,
    Call,
    Check,
    CheckCallPolicy,
    HoldemGame,
    IllegalAction,
    RandomLegalPolicy,
    RaiseTo,
    ScenarioBuilder,
    Street,
    TableConfig,
)


class ObservationTests(unittest.TestCase):
    def test_observation_exposes_only_hero_hole_cards(self):
        game = HoldemGame(TableConfig(("Hero", "Villain"), (100, 100)), seed=42)
        game.start_hand()
        observation = game.observation_for("Hero")
        internal = game.internal_state
        self.assertEqual(observation.hole_cards, internal.players[0].hole_cards)
        self.assertFalse(
            any(hasattr(player, "hole_cards") for player in observation.players)
        )
        self.assertEqual(len(internal.players[1].hole_cards), 2)
        self.assertEqual(observation.board, ())
        self.assertEqual(len(internal.remaining_deck), 48)

    def test_future_board_is_not_exposed(self):
        game = HoldemGame(TableConfig(("A", "B"), (100, 100)), seed=11)
        game.start_hand()
        future_flop = game.internal_state.remaining_deck[:3]
        observation = game.observation_for("A")
        self.assertEqual(observation.board, ())
        self.assertFalse(any(card in observation.hole_cards for card in future_flop))

    def test_non_actor_observation_has_no_legal_actions(self):
        game = HoldemGame(TableConfig(("A", "B"), (100, 100)), seed=1)
        game.start_hand()
        self.assertIsNotNone(game.observation_for("A").legal_actions)
        self.assertIsNone(game.observation_for("B").legal_actions)


class ScenarioTests(unittest.TestCase):
    def test_exact_three_player_hypothetical_reconstructs_normal_state(self):
        builder = ScenarioBuilder(TableConfig(("BTN", "SB", "BB"), (200, 200, 200)))
        builder.set_hole_cards("BTN", "As Qs")
        builder.set_board_runout("Qd 8c 4s")
        builder.action("BTN", RaiseTo(6))
        builder.action("SB", Call())
        builder.action("BB", Call())
        builder.action("SB", Check())
        builder.action("BB", BetTo(12))
        game = builder.build()

        self.assertEqual(game.street, Street.FLOP)
        self.assertEqual(tuple(map(str, game.board)), ("Qd", "8c", "4s"))
        self.assertEqual(game.current_player, "BTN")
        self.assertEqual(game.pot, 30)
        legal = game.legal_actions("BTN")
        self.assertEqual(legal.call_amount, 12)
        observation = game.observation_for("BTN")
        self.assertEqual(tuple(map(str, observation.hole_cards)), ("As", "Qs"))
        self.assertEqual(len(observation.history), 7)

    def test_illegal_hypothetical_history_is_rejected(self):
        builder = ScenarioBuilder(TableConfig(("A", "B"), (100, 100)))
        builder.action("A", Check())
        with self.assertRaises(IllegalAction):
            builder.build()

    def test_duplicate_scenario_cards_are_rejected(self):
        builder = ScenarioBuilder(TableConfig(("A", "B"), (100, 100)))
        builder.set_hole_cards("A", "As Qs")
        with self.assertRaisesRegex(ValueError, "duplicate"):
            builder.set_hole_cards("B", "As Kd")

    def test_preset_deck_is_dealt_deterministically(self):
        deck = full_deck()
        first = HoldemGame(TableConfig(("A", "B"), (100, 100)), preset_deck=deck)
        second = HoldemGame(TableConfig(("A", "B"), (100, 100)), preset_deck=deck)
        first.start_hand()
        second.start_hand()
        self.assertEqual(first.internal_state, second.internal_state)
        self.assertEqual(first.internal_state.players[1].hole_cards, (deck[0], deck[2]))
        self.assertEqual(first.internal_state.players[0].hole_cards, (deck[1], deck[3]))


class InvariantAndPolicyTests(unittest.TestCase):
    def test_seed_and_identical_actions_reproduce_complete_hand(self):
        def play(seed):
            game = HoldemGame(TableConfig(("A", "B", "C"), (100, 100, 100)), seed=seed)
            policy = CheckCallPolicy()
            game.start_hand()
            initial_hands = tuple(
                player.hole_cards for player in game.internal_state.players
            )
            while not game.is_terminal:
                actor = game.current_player
                observation = game.observation_for(actor)
                game.step(policy.decide(observation, observation.legal_actions))
            return initial_hands, tuple(game.board), game.result

        self.assertEqual(play(77), play(77))

    def test_chip_and_card_conservation_after_every_action(self):
        game = HoldemGame(TableConfig(("A", "B", "C"), (50, 40, 30)), seed=5)
        game.start_hand()
        policy = RandomLegalPolicy(5)
        while not game.is_terminal:
            game.assert_invariants()
            actor = game.current_player
            observation = game.observation_for(actor)
            game.step(policy.decide(observation, observation.legal_actions))
        game.assert_invariants()
        self.assertEqual(sum(game.result.final_stacks.values()), 120)

    def test_short_stack_call_is_naturally_all_in(self):
        game = HoldemGame(TableConfig(("A", "B", "C"), (100, 7, 100)), seed=3)
        game.start_hand()
        game.step(RaiseTo(20))
        legal = game.legal_actions("B")
        self.assertEqual(legal.call_amount, 6)
        transition = game.step(Call())
        self.assertTrue(transition.action_record.caused_all_in)
        self.assertEqual(transition.action_record.contribution_after, 7)
        self.assertEqual(game.current_player, "C")

    def test_randomized_legal_hands_preserve_invariants(self):
        completed = 0
        for player_count in range(2, 7):
            for seed in range(40):
                ids = tuple(f"P{i}" for i in range(player_count))
                stacks = tuple(40 + i * 3 for i in range(player_count))
                game = HoldemGame(
                    TableConfig(ids, stacks, 1, 2, seed % player_count), seed=seed
                )
                policy = RandomLegalPolicy(seed)
                game.start_hand()
                actions = 0
                while not game.is_terminal:
                    actor = game.current_player
                    observation = game.observation_for(actor)
                    game.step(policy.decide(observation, observation.legal_actions))
                    game.assert_invariants()
                    actions += 1
                    self.assertLess(actions, 200)
                self.assertEqual(sum(game.result.final_stacks.values()), sum(stacks))
                completed += 1
        self.assertEqual(completed, 200)


if __name__ == "__main__":
    unittest.main()
