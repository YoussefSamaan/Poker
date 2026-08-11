import unittest

from poker_ai.holdem import (
    BetTo,
    Call,
    Check,
    Fold,
    HoldemGame,
    IllegalAction,
    PlayerStatus,
    RaiseTo,
    Street,
    TableConfig,
)


def game_for(player_ids=("A", "B"), stacks=None, button=0, seed=1):
    stacks = stacks or tuple(100 for _ in player_ids)
    game = HoldemGame(
        TableConfig(tuple(player_ids), tuple(stacks), 1, 2, button), seed=seed
    )
    game.start_hand()
    return game


class TableOrderTests(unittest.TestCase):
    def test_heads_up_blinds_and_preflop_order(self):
        game = game_for()
        state = game.internal_state
        self.assertEqual(state.button_player, "A")
        self.assertEqual(state.small_blind_player, "A")
        self.assertEqual(state.big_blind_player, "B")
        self.assertEqual(game.current_player, "A")

    def test_heads_up_big_blind_acts_first_postflop(self):
        game = game_for()
        game.step(Call())
        transition = game.step(Check())
        self.assertTrue(transition.street_changed)
        self.assertEqual(game.street, Street.FLOP)
        self.assertEqual(game.current_player, "B")

    def test_three_player_order(self):
        game = game_for(("BTN", "SB", "BB"))
        self.assertEqual(game.internal_state.small_blind_player, "SB")
        self.assertEqual(game.internal_state.big_blind_player, "BB")
        self.assertEqual(game.current_player, "BTN")
        game.step(Call())
        game.step(Call())
        game.step(Check())
        self.assertEqual(game.current_player, "SB")

    def test_six_player_preflop_and_postflop_order(self):
        ids = tuple(f"P{i}" for i in range(6))
        game = game_for(ids)
        self.assertEqual(game.internal_state.small_blind_player, "P1")
        self.assertEqual(game.internal_state.big_blind_player, "P2")
        self.assertEqual(game.current_player, "P3")
        for _ in range(5):
            game.step(Call())
        game.step(Check())
        self.assertEqual(game.street, Street.FLOP)
        self.assertEqual(game.current_player, "P1")

    def test_button_rotates_between_hands(self):
        game = game_for()
        game.step(Fold())
        self.assertTrue(game.is_terminal)
        game.start_hand()
        self.assertEqual(game.internal_state.button_player, "B")
        self.assertEqual(game.internal_state.small_blind_player, "B")
        self.assertEqual(game.current_player, "B")


class LegalActionTests(unittest.TestCase):
    def test_table_rejects_non_integer_chip_accounting(self):
        with self.assertRaises(TypeError):
            TableConfig(("A", "B"), (100, 100), 0.5, 1)  # type: ignore[arg-type]

    def test_valid_fold_call_check_bet_and_raise(self):
        fold_game = game_for()
        fold_game.step(Fold())
        self.assertTrue(fold_game.is_terminal)

        game = game_for()
        game.step(Call())
        game.step(Check())
        self.assertEqual(game.current_player, "B")
        game.step(BetTo(2))
        self.assertEqual(game.current_player, "A")
        game.step(RaiseTo(4))
        self.assertEqual(game.current_player, "B")
        game.step(Call())

    def test_invalid_check_facing_bet(self):
        with self.assertRaisesRegex(IllegalAction, "check"):
            game_for().step(Check())

    def test_invalid_call_when_nothing_owed(self):
        game = game_for()
        game.step(Call())
        with self.assertRaisesRegex(IllegalAction, "call"):
            game.step(Call())

    def test_bet_is_rejected_when_raise_is_required(self):
        with self.assertRaisesRegex(IllegalAction, "RaiseTo"):
            game_for().step(BetTo(6))

    def test_raise_is_rejected_without_existing_wager(self):
        game = game_for()
        game.step(Call())
        game.step(Check())
        with self.assertRaisesRegex(IllegalAction, "raising"):
            game.step(RaiseTo(2))

    def test_undersized_and_over_stack_raises_are_rejected(self):
        game = game_for()
        with self.assertRaisesRegex(IllegalAction, "between 4 and 100"):
            game.step(RaiseTo(3))
        with self.assertRaisesRegex(IllegalAction, "between 4 and 100"):
            game.step(RaiseTo(101))

    def test_wrong_player_cannot_act(self):
        with self.assertRaisesRegex(IllegalAction, "current player"):
            game_for().step(Call(), "B")

    def test_folded_and_all_in_players_are_skipped(self):
        game = game_for(("A", "B", "C"), (100, 4, 100))
        game.step(Fold())
        self.assertEqual(game._player("A").status, PlayerStatus.FOLDED)
        game.step(RaiseTo(4))
        self.assertEqual(game._player("B").status, PlayerStatus.ALL_IN)
        self.assertEqual(game.current_player, "C")
        with self.assertRaises(IllegalAction):
            game.step(Call(), "A")
        with self.assertRaises(IllegalAction):
            game.step(Call(), "B")

    def test_action_history_records_complete_transition_data(self):
        game = game_for()
        transition = game.step(RaiseTo(6))
        record = transition.action_record
        self.assertEqual(record.amount_paid, 5)
        self.assertEqual(record.target_to, 6)
        self.assertEqual(record.contribution_before, 1)
        self.assertEqual(record.contribution_after, 6)
        self.assertEqual(record.amount_to_call_before, 1)
        self.assertEqual(record.pot_before, 3)
        self.assertEqual(record.pot_after, 8)
        self.assertEqual(record.stack_before, 99)
        self.assertEqual(record.stack_after, 94)

    def test_aggression_is_capped_at_effective_opponent_stack(self):
        game = game_for(("A", "B"), (100, 10))
        legal = game.legal_actions("A")
        self.assertEqual(legal.max_raise_to, 10)
        with self.assertRaises(IllegalAction):
            game.step(RaiseTo(11))


class MinimumRaiseTests(unittest.TestCase):
    def test_full_raise_updates_next_minimum(self):
        game = game_for(("A", "B", "C"))
        game.step(RaiseTo(6))
        self.assertEqual(game.legal_actions("B").min_raise_to, 10)
        game.step(RaiseTo(10))
        self.assertEqual(game.legal_actions("C").min_raise_to, 14)

    def test_full_all_in_raise_reopens_betting(self):
        game = game_for(("A", "B", "C"), (10, 100, 100))
        game.step(RaiseTo(10))
        self.assertEqual(game._player("A").status, PlayerStatus.ALL_IN)
        self.assertEqual(game.legal_actions("B").min_raise_to, 18)

    def test_short_all_in_does_not_reopen_for_prior_actor(self):
        game = game_for(("A", "B", "C"), (100, 100, 8))
        game.step(RaiseTo(6))
        game.step(Call())
        self.assertEqual(game.current_player, "C")
        game.step(RaiseTo(8))
        self.assertEqual(game._player("C").status, PlayerStatus.ALL_IN)
        legal_a = game.legal_actions("A")
        self.assertEqual(legal_a.call_amount, 2)
        self.assertIsNone(legal_a.min_raise_to)
        game.step(Call())
        legal_b = game.legal_actions("B")
        self.assertEqual(legal_b.call_amount, 2)
        self.assertIsNone(legal_b.min_raise_to)

    def test_short_all_in_preserves_raise_rights_for_unacted_player(self):
        game = game_for(("A", "B", "C"), (100, 8, 100))
        game.step(RaiseTo(6))
        game.step(RaiseTo(8))
        legal_c = game.legal_actions("C")
        self.assertEqual(legal_c.min_raise_to, 12)

    def test_short_all_in_must_use_exact_remaining_stack(self):
        game = game_for(("A", "B", "C"), (100, 8, 100))
        game.step(RaiseTo(6))
        legal = game.legal_actions("B")
        self.assertEqual((legal.min_raise_to, legal.max_raise_to), (8, 8))
        with self.assertRaises(IllegalAction):
            game.step(RaiseTo(7))

    def test_check_then_short_open_all_in_still_allows_raise(self):
        game = game_for(("A", "B", "C"), (100, 100, 3))
        game.step(Call())
        game.step(Call())
        game.step(Check())
        self.assertEqual(game.current_player, "B")
        game.step(Check())
        game.step(BetTo(1))
        self.assertEqual(game.legal_actions("A").min_raise_to, 2)

    def test_cumulative_short_all_ins_can_reopen_betting(self):
        game = game_for(("A", "B", "C", "D"), (100, 8, 10, 100))
        # D calls, then A raises to 6. B's shove to 8 and C's shove to 10
        # are each short, but together equal the prior full-raise increment.
        game.step(Call())
        game.step(RaiseTo(6))
        game.step(RaiseTo(8))
        game.step(RaiseTo(10))
        game.step(Call())
        self.assertEqual(game.current_player, "A")
        self.assertEqual(game.legal_actions("A").min_raise_to, 14)

    def test_short_big_blind_uses_full_bring_in_multiway(self):
        game = game_for(("A", "B", "C"), (100, 100, 1))
        self.assertEqual(game.current_bet, 2)
        self.assertEqual(game.legal_actions("A").call_amount, 2)


if __name__ == "__main__":
    unittest.main()
