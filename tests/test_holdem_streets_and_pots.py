import unittest

from poker_ai.holdem import (
    BetTo,
    Call,
    Check,
    Fold,
    HoldemGame,
    RaiseTo,
    ScenarioBuilder,
    Street,
    TableConfig,
)
from poker_ai.holdem.pots import build_side_pots


class StreetAndTerminalTests(unittest.TestCase):
    def test_complete_heads_up_hand_advances_every_street(self):
        game = HoldemGame(TableConfig(("A", "B"), (100, 100)), seed=4)
        game.start_hand()
        game.step(Call())
        first = game.step(Check())
        self.assertEqual((game.street, len(first.cards_revealed)), (Street.FLOP, 3))
        for expected in (Street.TURN, Street.RIVER):
            game.step(Check())
            transition = game.step(Check())
            self.assertEqual(game.street, expected)
            self.assertEqual(len(transition.cards_revealed), 1)
        game.step(Check())
        terminal = game.step(Check())
        self.assertTrue(terminal.hand_terminated)
        self.assertEqual(game.street, Street.SHOWDOWN)
        self.assertTrue(game.result.showdown)

    def test_everyone_folds_except_one(self):
        game = HoldemGame(TableConfig(("A", "B", "C"), (100, 100, 100)), seed=2)
        game.start_hand()
        game.step(Fold())
        game.step(Fold())
        self.assertTrue(game.is_terminal)
        self.assertEqual(game.result.reason, "all_others_folded")
        self.assertEqual(game.result.winners, ("C",))
        self.assertEqual(sum(game.result.final_stacks.values()), 300)

    def test_remaining_players_all_in_runs_out_board(self):
        game = HoldemGame(TableConfig(("A", "B"), (10, 10)), seed=8)
        game.start_hand()
        game.step(RaiseTo(10))
        transition = game.step(Call())
        self.assertTrue(transition.hand_terminated)
        self.assertEqual(len(transition.cards_revealed), 5)
        self.assertEqual(len(game.board), 5)
        self.assertEqual(game.street, Street.SHOWDOWN)

    def test_multiway_showdown(self):
        game = HoldemGame(TableConfig(("A", "B", "C"), (20, 20, 20)), seed=9)
        game.start_hand()
        game.step(RaiseTo(20))
        game.step(Call())
        game.step(Call())
        self.assertTrue(game.is_terminal)
        self.assertTrue(game.result.showdown)
        self.assertEqual(sum(game.result.payouts.values()), 60)


class PotTests(unittest.TestCase):
    def test_side_pot_derivation_includes_folded_chips(self):
        pots = build_side_pots({"A": 10, "B": 10, "C": 5}, {"A", "B"})
        self.assertEqual([pot.amount for pot in pots], [15, 10])
        self.assertEqual(pots[0].contributors, ("A", "B", "C"))
        self.assertEqual(pots[0].eligible_players, ("A", "B"))

    def test_folded_contribution_remains_in_engine_pots(self):
        builder = ScenarioBuilder(TableConfig(("A", "B", "C"), (100, 100, 100)))
        builder.set_hole_cards("A", "As Ad").set_hole_cards("B", "Ks Kd")
        builder.set_hole_cards("C", "Qs Qd").set_board_runout("2c 3d 4h 9s Tc")
        builder.action("A", RaiseTo(6)).action("B", Call()).action("C", Call())
        builder.action("B", BetTo(10)).action("C", Fold()).action("A", Call())
        builder.action("B", Check()).action("A", Check())
        builder.action("B", Check()).action("A", Check())
        game = builder.build()
        self.assertEqual([pot.amount for pot in game.result.pots], [18, 20])
        self.assertNotIn("C", game.result.pots[0].eligible_players)
        self.assertEqual(sum(game.result.payouts.values()), 38)

    def test_one_side_pot_with_different_main_and_side_winners(self):
        builder = ScenarioBuilder(TableConfig(("A", "B", "C"), (100, 60, 30)))
        builder.set_hole_cards("A", "As Kd")
        builder.set_hole_cards("B", "8s 7d")
        builder.set_hole_cards("C", "Ah Kc")
        builder.set_board_runout("2c 3d 4h 9s Jc")
        builder.action("A", RaiseTo(60)).action("B", Call()).action("C", Call())
        game = builder.build()
        self.assertEqual([pot.amount for pot in game.result.pots], [90, 60])
        self.assertEqual(game.result.pots[0].winners, ("C", "A"))
        self.assertEqual(game.result.pots[1].winners, ("A",))
        self.assertEqual(game.result.payouts, {"A": 105, "B": 0, "C": 45})

    def test_multiple_side_pots(self):
        builder = ScenarioBuilder(TableConfig(("A", "B", "C", "D"), (100, 60, 30, 16)))
        builder.set_hole_cards("A", "As Ad").set_hole_cards("B", "Ks Kd")
        builder.set_hole_cards("C", "Qs Qd").set_hole_cards("D", "Js Jd")
        builder.set_board_runout("2c 3d 4h 9s Tc")
        builder.action("D", Call()).action("A", RaiseTo(60))
        builder.action("B", Call()).action("C", Call()).action("D", Call())
        game = builder.build()
        self.assertEqual([pot.amount for pot in game.result.pots], [64, 42, 60])
        self.assertEqual(game.result.pots[0].winners, ("A",))
        self.assertEqual(sum(game.result.payouts.values()), 166)

    def test_odd_chip_goes_clockwise_left_of_button(self):
        builder = ScenarioBuilder(TableConfig(("A", "B", "C"), (10, 10, 5)))
        builder.set_hole_cards("A", "As Kd").set_hole_cards("B", "8s 7d")
        builder.set_hole_cards("C", "Ah Kc").set_board_runout("2c 3d 4h 9s Jc")
        builder.action("A", RaiseTo(10)).action("B", Call()).action("C", Call())
        game = builder.build()
        self.assertEqual(game.result.pots[0].amount, 15)
        self.assertEqual(game.result.pots[0].winners, ("C", "A"))
        self.assertEqual(game.result.pots[0].payouts, {"C": 8, "A": 7})
        self.assertEqual(game.result.payouts, {"A": 17, "B": 0, "C": 8})

    def test_single_pot_exact_tie(self):
        builder = ScenarioBuilder(TableConfig(("A", "B"), (10, 10)))
        builder.set_hole_cards("A", "2c 3d").set_hole_cards("B", "4c 5d")
        builder.set_board_runout("As Ks Qs Js Ts")
        builder.action("A", RaiseTo(10)).action("B", Call())
        game = builder.build()
        self.assertEqual(game.result.payouts, {"A": 10, "B": 10})


if __name__ == "__main__":
    unittest.main()
