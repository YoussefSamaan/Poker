import unittest

from poker_ai.agents import (
    PRESETS,
    PersonalityAgent,
    StrategyProfile,
    extract_features,
    position_name,
)
from poker_ai.holdem import BetTo, HoldemGame, PlayerObservation, TableConfig


class AgentTests(unittest.TestCase):
    def observation(self, players=6, seed=3):
        ids = tuple(f"P{i}" for i in range(players))
        game = HoldemGame(TableConfig(ids, (200,) * players, 1, 2, 0), seed=seed)
        game.start_hand()
        actor = game.current_player
        return game, game.observation_for(actor), game.legal_actions(actor)

    def test_positions_two_to_six(self):
        expected = {
            2: {"BTN/SB", "BB"},
            3: {"BTN", "SB", "BB"},
            4: {"BTN", "SB", "BB", "CO"},
            5: {"BTN", "SB", "BB", "HJ", "CO"},
            6: {"BTN", "SB", "BB", "UTG", "HJ", "CO"},
        }
        for count, names in expected.items():
            game, observation, _ = self.observation(count)
            self.assertEqual(
                {
                    position_name(observation, player.player_id)
                    for player in observation.players
                },
                names,
            )

    def test_features_are_observation_only_and_preflop_fields(self):
        _, observation, legal = self.observation(2)
        features = extract_features(observation, legal)
        self.assertIsInstance(observation, PlayerObservation)
        self.assertIn(features.hand_class[-1], {"s", "o"})
        self.assertGreaterEqual(features.pot_odds, 0)
        self.assertIsNone(features.spr)

    def test_profiles_round_trip_and_presets(self):
        self.assertEqual(
            set(PRESETS),
            {"nit", "tag", "lag", "calling_station", "maniac", "bluff_heavy"},
        )
        for profile in PRESETS.values():
            self.assertEqual(StrategyProfile.from_dict(profile.to_dict()), profile)

    def test_distributions_are_legal_normalized_and_meaningfully_ordered(self):
        game = HoldemGame(TableConfig(("Hero", "V"), (100, 100), 1, 2, 0), seed=9)
        game.start_hand()
        actor = game.current_player
        observation, legal = game.observation_for(actor), game.legal_actions(actor)
        distributions = {
            name: PersonalityAgent(profile).action_distribution(observation, legal)
            for name, profile in PRESETS.items()
        }
        for distribution in distributions.values():
            self.assertAlmostEqual(sum(distribution.values()), 1)
            self.assertTrue(all(value >= 0 for value in distribution.values()))
        self.assertGreaterEqual(
            distributions["lag"].get("raise", 0), distributions["nit"].get("raise", 0)
        )

    def test_postflop_personality_parameter_ordering(self):
        # Fixed full-hand search reaches a postflop legal state without privileged inputs.
        game = HoldemGame(TableConfig(("A", "B"), (100, 100), 1, 2, 0), seed=2)
        game.start_hand()
        from poker_ai.holdem import Call, Check

        game.step(Call())
        game.step(Check())
        observation = game.observation_for(game.current_player)
        legal = game.legal_actions(game.current_player)
        tag = PersonalityAgent(PRESETS["tag"]).action_distribution(observation, legal)
        station = PersonalityAgent(PRESETS["calling_station"]).action_distribution(
            observation, legal
        )
        maniac = PersonalityAgent(PRESETS["maniac"]).action_distribution(
            observation, legal
        )
        bluff = PersonalityAgent(PRESETS["bluff_heavy"]).action_distribution(
            observation, legal
        )
        self.assertGreater(maniac.get("bet", 0), station.get("bet", 0))
        self.assertGreaterEqual(bluff.get("bet", 0), tag.get("bet", 0))

        game.step(BetTo(2))
        facing = game.observation_for(game.current_player)
        facing_legal = game.legal_actions(game.current_player)
        nit = PersonalityAgent(PRESETS["nit"]).action_distribution(facing, facing_legal)
        station_facing = PersonalityAgent(
            PRESETS["calling_station"]
        ).action_distribution(facing, facing_legal)
        self.assertGreater(station_facing.get("call", 0), nit.get("call", 0))
        self.assertGreater(nit.get("fold", 0), station_facing.get("fold", 0))

    def test_determinism_trace_and_randomized_full_hand_legality(self):
        game, observation, legal = self.observation(3, 14)
        first = PersonalityAgent(PRESETS["lag"], 44).decide_with_trace(
            observation, legal
        )
        second = PersonalityAgent(PRESETS["lag"], 44).decide_with_trace(
            observation, legal
        )
        self.assertEqual(first, second)
        for name, profile in PRESETS.items():
            for seed in range(4):
                game = HoldemGame(
                    TableConfig(("A", "B", "C"), (60, 60, 60), 1, 2, 0), seed=seed
                )
                agents = {
                    player: PersonalityAgent(profile, 100 + seat)
                    for seat, player in enumerate(("A", "B", "C"))
                }
                game.start_hand()
                while not game.is_terminal:
                    actor = game.current_player
                    observation = game.observation_for(actor)
                    self.assertIsInstance(observation, PlayerObservation)
                    game.step(
                        agents[actor].decide(observation, game.legal_actions(actor))
                    )
                self.assertTrue(
                    any(agent.last_trace is not None for agent in agents.values()), name
                )

    def test_preset_configuration_implies_expected_behavioral_order(self):
        from poker_ai.ranges import PreflopRange

        nit = (
            PreflopRange.parse(PRESETS["nit"].open_range("BTN")).stats().raw_combo_count
        )
        tag = (
            PreflopRange.parse(PRESETS["tag"].open_range("BTN")).stats().raw_combo_count
        )
        lag = (
            PreflopRange.parse(PRESETS["lag"].open_range("BTN")).stats().raw_combo_count
        )
        self.assertLess(nit, tag)
        self.assertLess(tag, lag)
        self.assertGreater(
            PRESETS["calling_station"].call_open_frequency,
            PRESETS["nit"].call_open_frequency,
        )
        self.assertGreater(
            PRESETS["maniac"].three_bet_frequency,
            PRESETS["calling_station"].three_bet_frequency,
        )
        self.assertGreater(
            PRESETS["bluff_heavy"].bluff_frequency, PRESETS["tag"].bluff_frequency
        )


if __name__ == "__main__":
    unittest.main()
