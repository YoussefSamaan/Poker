from dataclasses import replace
import unittest

from poker_ai.agents import PRESETS, PersonalityAgent
from poker_ai.agents.features import (
    _straight_completion_ranks,
    extract_features,
)
from poker_ai.cards import parse_cards
from poker_ai.experiments import (
    Participant,
    SimulationConfig,
    SimulationRunner,
    build_schedule,
    run_crossplay,
)
from poker_ai.experiments.metrics import summarize_metrics
from poker_ai.experiments.records import SeatHandStats
from poker_ai.experiments.simulator import _three_bet_counts
from poker_ai.holdem import (
    ActionRecord,
    ActionType,
    BetTo,
    Call,
    Check,
    HoldemGame,
    Street,
    TableConfig,
)


class MethodologyHardeningTests(unittest.TestCase):
    def test_balanced_schedule_positions_and_seats_two_to_six(self):
        for count in range(2, 7):
            profiles = tuple(list(PRESETS.values())[:count])
            result = SimulationRunner(
                SimulationConfig(profiles, hands=count * count, master_seed=3)
            ).run()
            for participant in result.metadata["participants"]:
                participant_id = participant["id"]
                seats = [
                    seat
                    for record in result.records
                    for seat in record.seats
                    if seat.participant_id == participant_id
                ]
                position_counts = {
                    position: sum(item.position == position for item in seats)
                    for position in {item.position for item in seats}
                }
                seat_counts = {
                    seat: sum(item.seat == seat for item in seats)
                    for seat in range(count)
                }
                self.assertEqual(
                    len(set(position_counts.values())),
                    1,
                    (count, participant_id, position_counts),
                )
                self.assertEqual(
                    len(set(seat_counts.values())),
                    1,
                    (count, participant_id, seat_counts),
                )

    def test_duplicate_blocks_keep_button_deal_and_rotate_all_assignments(self):
        for count in (2, 3, 4):
            schedule = build_schedule(count * 2, count, True)
            for block in range(2):
                legs = schedule[block * count : (block + 1) * count]
                self.assertEqual(len({leg.button for leg in legs}), 1)
                for participant in range(count):
                    self.assertEqual(
                        {
                            leg.participant_indices_by_seat.index(participant)
                            for leg in legs
                        },
                        set(range(count)),
                    )
            profiles = tuple(list(PRESETS.values())[:count])
            result = SimulationRunner(
                SimulationConfig(
                    profiles, hands=count, duplicate_deals=True, master_seed=8
                )
            ).run()
            self.assertEqual(len({record.deal_seed for record in result.records}), 1)
            self.assertEqual(len({record.button for record in result.records}), 1)
            self.assertEqual(
                [record.duplicate_leg for record in result.records], list(range(count))
            )
            reconstructed = []
            for record in result.records:
                game = HoldemGame(
                    TableConfig(
                        tuple(f"P{seat + 1}" for seat in range(count)),
                        (200,) * count,
                        1,
                        2,
                        record.button,
                    ),
                    seed=record.deal_seed,
                )
                game.start_hand()
                reconstructed.append(
                    tuple(player.hole_cards for player in game.internal_state.players)
                )
            self.assertTrue(all(deal == reconstructed[0] for deal in reconstructed))

    def test_duplicate_incomplete_block_rejected(self):
        with self.assertRaisesRegex(ValueError, "divisible"):
            SimulationRunner(
                SimulationConfig(
                    (PRESETS["tag"], PRESETS["lag"], PRESETS["nit"]),
                    hands=4,
                    duplicate_deals=True,
                )
            )

    def test_participant_identity_prevents_same_name_merge(self):
        low = PRESETS["tag"].with_parameter("bluff_frequency", 0.1)
        high = PRESETS["tag"].with_parameter("bluff_frequency", 0.4)
        participants = (
            Participant("tag_10", "TAG 10", low),
            Participant("tag_40", "TAG 40", high),
        )
        result = SimulationRunner(
            SimulationConfig((low, high), hands=4, participants=participants)
        ).run()
        self.assertEqual(
            {metric.profile for metric in result.metrics}, {"tag_10", "tag_40"}
        )
        self.assertNotEqual(
            participants[0].profile_fingerprint, participants[1].profile_fingerprint
        )

    def test_crossplay_is_antisymmetric_with_mirrored_intervals(self):
        result = run_crossplay(
            (PRESETS["nit"], PRESETS["tag"], PRESETS["lag"]),
            hands_per_matchup=20,
            seed=4,
        )
        for row in range(3):
            self.assertEqual(result.matrix[row][row], 0)
            for column in range(3):
                self.assertAlmostEqual(
                    result.matrix[row][column], -result.matrix[column][row]
                )
                low, high = result.confidence_intervals[row][column]
                reverse = result.confidence_intervals[column][row]
                self.assertEqual(reverse, (-high, -low))


class AgentHardeningTests(unittest.TestCase):
    def test_free_check_has_no_fold_probability(self):
        game = HoldemGame(TableConfig(("SB", "BB"), (100, 100), 1, 2, 0), seed=5)
        game.start_hand()
        game.step(Call())
        observation = game.observation_for("BB")
        distribution = PersonalityAgent(PRESETS["nit"]).action_distribution(
            observation, game.legal_actions("BB")
        )
        self.assertNotIn("fold", distribution)

    def test_explicit_postflop_weights_change_independently(self):
        game = HoldemGame(TableConfig(("A", "B"), (100, 100), 1, 2, 0), seed=2)
        game.start_hand()
        game.step(Call())
        game.step(Check())
        actor = game.current_player
        game.step(BetTo(2), actor)
        observation = game.observation_for(game.current_player)
        legal = game.legal_actions(game.current_player)
        base = PRESETS["tag"]
        bucket = extract_features(observation, legal).bucket.value

        def changed(table_name, factor):
            table = tuple(
                (key, value * factor if key == bucket else value)
                for key, value in getattr(base, table_name)
            )
            return replace(base, **{table_name: table})

        original = PersonalityAgent(base).action_distribution(observation, legal)
        self.assertGreater(
            PersonalityAgent(changed("fold_weights", 4)).action_distribution(
                observation, legal
            )["fold"],
            original["fold"],
        )
        self.assertGreater(
            PersonalityAgent(changed("call_weights", 4)).action_distribution(
                observation, legal
            )["call"],
            original["call"],
        )
        self.assertGreater(
            PersonalityAgent(changed("aggression_weights", 4)).action_distribution(
                observation, legal
            )["raise"],
            original["raise"],
        )

    def test_bluff_and_real_draw_modifiers_but_not_wheel_edge(self):
        game = HoldemGame(TableConfig(("A", "B"), (100, 100), 1, 2, 0), seed=2)
        game.start_hand()
        game.step(Call())
        game.step(Check())
        observation = game.observation_for(game.current_player)
        legal = game.legal_actions(game.current_player)

        air = replace(
            observation,
            hole_cards=parse_cards("As 8d"),
            board=parse_cards("Kh 7c 2s"),
        )
        self.assertEqual(extract_features(air, legal).bucket.value, "air")
        tag_air = PersonalityAgent(PRESETS["tag"]).action_distribution(air, legal)
        bluff_air = PersonalityAgent(PRESETS["bluff_heavy"]).action_distribution(
            air, legal
        )
        self.assertGreater(bluff_air["bet"], tag_air["bet"])

        draw = replace(
            observation,
            hole_cards=parse_cards("Ah Th"),
            board=parse_cards("Kh 7h 2s"),
        )
        self.assertEqual(extract_features(draw, legal).bucket.value, "draw")
        low = replace(PRESETS["tag"], semi_bluff_multiplier=0.5)
        high = replace(PRESETS["tag"], semi_bluff_multiplier=2.0)
        self.assertGreater(
            PersonalityAgent(high).action_distribution(draw, legal)["bet"],
            PersonalityAgent(low).action_distribution(draw, legal)["bet"],
        )

        wheel_edge = replace(
            observation,
            hole_cards=parse_cards("As 2d"),
            board=parse_cards("3c 4h 9s"),
        )
        edge_features = extract_features(wheel_edge, legal)
        self.assertFalse(edge_features.open_ended_straight_draw)
        self.assertTrue(edge_features.gutshot_straight_draw)
        self.assertNotEqual(edge_features.bucket.value, "draw")

    def test_profile_and_sweep_validation(self):
        with self.assertRaises(ValueError):
            PRESETS["tag"].with_parameter("bluff_frequency", -0.1)
        with self.assertRaises(ValueError):
            PRESETS["tag"].with_parameter("bluff_frequency", 1.1)
        with self.assertRaises(ValueError):
            replace(PRESETS["tag"], call_open_range="not-a-range")

    def test_straight_completion_edges(self):
        self.assertEqual(
            _straight_completion_ranks(parse_cards("As 2d 3c 4h")), frozenset({5})
        )
        self.assertEqual(
            _straight_completion_ranks(parse_cards("2s 3d 4c 5h")), frozenset({6, 14})
        )
        self.assertEqual(
            _straight_completion_ranks(parse_cards("Ts Jd Qc Kh")), frozenset({9, 14})
        )
        self.assertEqual(
            _straight_completion_ranks(parse_cards("9s Td Jc Qh Ks")), frozenset()
        )

    def test_previous_aggressor_survives_intervening_call(self):
        game = HoldemGame(TableConfig(("A", "B"), (100, 100), 1, 2, 0), seed=9)
        game.start_hand()
        observation = game.observation_for("A")
        history = (
            _action(0, "A", ActionType.BET, Street.FLOP),
            _action(1, "B", ActionType.CALL, Street.FLOP),
        )
        features = extract_features(
            replace(observation, street=Street.FLOP, history=history),
            game.legal_actions("A"),
        )
        self.assertTrue(features.previous_aggressor)


class BehavioralMetricDefinitionTests(unittest.TestCase):
    def test_action_denominator_and_player_showdown_are_explicit(self):
        row = SeatHandStats(
            "P", "Profile", "participant", 0, "BTN", 0, 0.0,
            True, True, 2, 1, 1, 2, 3, 4, 6, 2, False,
        )
        metric = summarize_metrics("participant", (row,))
        self.assertEqual(metric.vpip, 1.0)
        self.assertEqual(metric.pfr, 1.0)
        self.assertEqual(metric.three_bet_frequency, 0.5)
        self.assertEqual(metric.fold_frequency, 0.1)
        self.assertEqual(metric.check_frequency, 0.2)
        self.assertEqual(metric.call_frequency, 0.3)
        self.assertEqual(metric.bet_raise_frequency, 0.4)
        self.assertEqual(metric.postflop_aggression_frequency, 2 / 6)
        self.assertEqual(metric.showdown_rate, 0.0)

    def test_three_bet_hand_history_cases(self):
        cases = {
            "open_fold": (("A", ActionType.RAISE), ("H", ActionType.FOLD)),
            "open_call": (("A", ActionType.RAISE), ("H", ActionType.CALL)),
            "open_three_bet": (("A", ActionType.RAISE), ("H", ActionType.RAISE)),
            "limp_raise_limper": (
                ("H", ActionType.CALL),
                ("A", ActionType.RAISE),
                ("H", ActionType.CALL),
            ),
            "open_three_bet_original_again": (
                ("H", ActionType.RAISE),
                ("A", ActionType.RAISE),
                ("H", ActionType.CALL),
            ),
        }
        expected = {
            "open_fold": (1, 0),
            "open_call": (1, 0),
            "open_three_bet": (1, 1),
            "limp_raise_limper": (1, 0),
            "open_three_bet_original_again": (0, 0),
        }
        for name, actions in cases.items():
            history = tuple(
                _action(index, player, action, Street.PREFLOP)
                for index, (player, action) in enumerate(actions)
            )
            self.assertEqual(_three_bet_counts(history, "H"), expected[name], name)


def _action(
    sequence: int, player: str, action_type: ActionType, street: Street
) -> ActionRecord:
    return ActionRecord(
        sequence, street, player, action_type, 0, None, 0, 0, 0, 0, 0, 100, 100,
        False,
    )


if __name__ == "__main__":
    unittest.main()
