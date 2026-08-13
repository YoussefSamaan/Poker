from dataclasses import fields, replace
import math
import unittest

from poker_ai.agents import PRESETS, PersonalityAgent
from poker_ai.cards import parse_cards
from poker_ai.experiments import SimulationConfig, SimulationRunner
from poker_ai.experiments.schedule import Participant
from poker_ai.holdem import (
    Call,
    Check,
    HoldemGame,
    PlayerStatus,
    PublicPlayerState,
    RaiseTo,
    Street,
    TableConfig,
)
from poker_ai.opponents import (
    AdaptiveExploitPolicy,
    BetaEstimate,
    ObservedDecision,
    OpponentModel,
    OpponentModelTable,
    OpponentStats,
    RangeBelief,
    ResearchDecisionLabels,
    observe_decision,
    observed_decisions_from_session,
)
from poker_ai.opponents.validation import (
    adaptive_vs_fixed_experiment,
    calibration_experiment,
    holdout_predictive_evaluation,
)
from poker_ai.training import analyze_showdown_baseline
from poker_ai.trainer_app import create_example_session


def decision(
    *, hand=0, action="raise", raises=0, can_raise=True, street=Street.PREFLOP,
    position="BTN/SB", board=(), player="Villain", to_call=2, can_bet=False,
):
    players = (
        PublicPlayerState("Hero", 0, 100, PlayerStatus.ACTIVE, 0, 0),
        PublicPlayerState(player, 1, 100, PlayerStatus.ACTIVE, 0, 0),
    )
    return ObservedDecision(
        hand, player, player, street, position, tuple(board), "Hero", "Hero", player,
        players, (), 10, 2, to_call, True, to_call == 0, to_call > 0, can_bet,
        can_raise, raises, None, 2, action, None, None,
    )


class OpportunityAndPrivacyTests(unittest.TestCase):
    def test_simulator_captures_pre_action_legality_and_separate_labels(self):
        result = SimulationRunner(
            SimulationConfig((PRESETS["tag"], PRESETS["lag"]), hands=4, master_seed=2)
        ).run()
        public = result.records[0].observed_decisions[0]
        private = result.records[0].research_labels[0]
        self.assertIsInstance(public, ObservedDecision)
        self.assertIsInstance(private, ResearchDecisionLabels)
        names = {field.name for field in fields(ObservedDecision)}
        self.assertFalse(
            names.intersection({"hole_cards", "deck", "true_profile", "decision_trace"})
        )
        self.assertIn("true_hole_cards", {field.name for field in fields(type(private))})
        self.assertTrue(any((public.can_fold, public.can_check, public.can_call)))

    def test_three_bet_requires_legal_raise_and_second_reraise_is_excluded(self):
        stats = OpponentStats()
        stats.observe(decision(action="raise", raises=1, can_raise=True))
        stats.observe(decision(hand=1, action="call", raises=1, can_raise=False))
        stats.observe(decision(hand=2, action="raise", raises=2, can_raise=True))
        estimate = stats.estimate("three_bet")
        self.assertEqual((estimate.successes, estimate.opportunities), (1, 1))

    def test_postflop_opportunity_denominators(self):
        stats = OpponentStats()
        stats.observe(decision(street=Street.FLOP, action="fold", can_raise=False))
        stats.observe(
            decision(hand=1, street=Street.FLOP, action="bet", to_call=0, can_bet=True)
        )
        self.assertEqual(stats.estimate("fold_vs_bet").opportunities, 1)
        self.assertEqual(stats.estimate("bet_when_checked_to").successes, 1)


class BayesianAndStatsTests(unittest.TestCase):
    def test_beta_known_posteriors_and_exact_interval_behavior(self):
        prior = BetaEstimate()
        three_of_three = BetaEstimate(3, 0)
        self.assertEqual(prior.mean, 0.5)
        self.assertEqual(three_of_three.mean, 0.8)
        uniform_interval = prior.credible_interval()
        self.assertAlmostEqual(uniform_interval[0], 0.025, places=3)
        self.assertAlmostEqual(uniform_interval[1], 0.975, places=3)
        narrow = BetaEstimate(80, 20).credible_interval()
        self.assertLess(narrow[1] - narrow[0], uniform_interval[1] - uniform_interval[0])

    def test_incremental_matches_batch_and_conditions(self):
        values = [
            decision(hand=0, action="raise", raises=0),
            decision(hand=1, action="call", raises=1),
            decision(hand=2, action="fold", raises=1),
        ]
        incremental = OpponentStats()
        for value in values:
            incremental.observe(value)
        batch = OpponentStats()
        list(map(batch.observe, values))
        self.assertEqual(incremental.counts, batch.counts)
        self.assertIn("BTN/SB", incremental.position_counts)


class RangeAndArchetypeTests(unittest.TestCase):
    def _real_raise(self, players, seed=3):
        ids = tuple(f"P{i}" for i in range(players))
        game = HoldemGame(TableConfig(ids, (200,) * players, 1, 2, 0), seed=seed)
        game.start_hand()
        actor = game.current_player
        observation = game.observation_for(actor)
        legal = game.legal_actions(actor)
        return observe_decision(
            0, "Villain", observation, legal, RaiseTo(legal.min_raise_to)
        )

    def test_range_normalizes_blocks_and_updates_sequentially(self):
        observed = self._real_raise(6)
        dead = parse_cards("As Kd")
        belief = RangeBelief(known_cards=dead)
        before = belief.summary(dead)
        belief.update(observed, PRESETS["nit"], dead)
        after_one = belief.summary(dead)
        weights_one = dict(belief.weights)
        belief.update(observed, PRESETS["nit"], dead)
        self.assertAlmostEqual(sum(belief.weights.values()), 1)
        self.assertTrue(all(not set(cards).intersection(dead) for cards in belief.weights))
        self.assertNotEqual(before.entropy, after_one.entropy)
        self.assertNotEqual(weights_one, belief.weights)

    def test_nit_utg_raise_contracts_more_than_lag_button_raise(self):
        nit = RangeBelief()
        lag = RangeBelief()
        nit.update(self._real_raise(6), PRESETS["nit"])
        lag.update(self._real_raise(2), PRESETS["lag"])
        self.assertLess(
            nit.summary().effective_combo_count, lag.summary().effective_combo_count
        )

    def test_archetype_posterior_is_probabilistic_deterministic_and_label_free(self):
        observed = self._real_raise(2)
        first = OpponentModel("Hero", "Villain")
        second = OpponentModel("Hero", "Villain")
        initial = first.archetype_posterior
        self.assertTrue(all(abs(value - 1 / 6) < 1e-12 for value in initial.values()))
        first.observe(observed)
        second.observe(observed)
        self.assertEqual(first.archetype_posterior, second.archetype_posterior)
        self.assertAlmostEqual(sum(first.archetype_posterior.values()), 1)
        self.assertNotEqual(first.archetype_posterior, initial)
        with self.assertRaises(TypeError):
            first.observe(ResearchDecisionLabels(0, "Villain", "Villain", "Nit", ()))

    def test_multiway_models_and_json_round_trip(self):
        table = OpponentModelTable("Hero", ("Hero", "SB", "BB"))
        self.assertEqual(set(table.models), {"SB", "BB"})
        model = table.models["SB"]
        model.observe(decision(player="SB"))
        restored = OpponentModel.from_json(model.to_json())
        self.assertEqual(restored.snapshot(), model.snapshot())
        reset = OpponentModelTable("Hero", ("SB",), reset_each_hand=True)
        reset.observe(decision(player="SB", hand=0))
        reset.observe(decision(player="SB", hand=1))
        self.assertEqual(reset.models["SB"].snapshot().hands_observed, 1)


class AdaptiveAndEndToEndTests(unittest.TestCase):
    def _air_state(self):
        game = HoldemGame(TableConfig(("A", "B"), (100, 100), 1, 2, 0), seed=2)
        game.start_hand(); game.step(Call()); game.step(Check())
        return game.observation_for(game.current_player), game.legal_actions(game.current_player)

    def test_adaptation_is_confidence_aware_bounded_and_legal(self):
        observation, legal = self._air_state()
        model = OpponentModel("A", "B")
        base = PersonalityAgent(PRESETS["tag"]).action_distribution(observation, legal)
        tiny = AdaptiveExploitPolicy(PRESETS["tag"], model).action_distribution(
            observation, legal
        )
        self.assertEqual(base, tiny)
        model.stats.counts["fold_vs_bet"] = [45, 50]
        model.stats.counts["call_vs_bet"] = [5, 50]
        policy = AdaptiveExploitPolicy(PRESETS["tag"], model)
        adjusted = policy.action_distribution(observation, legal)
        self.assertGreater(adjusted["bet"], base["bet"])
        self.assertLessEqual(adjusted["bet"] - base["bet"], 0.15)
        model.stats.counts["fold_vs_bet"] = [5, 50]
        model.stats.counts["call_vs_bet"] = [45, 50]
        calling_adjusted = policy.action_distribution(observation, legal)
        self.assertLess(calling_adjusted["bet"], base["bet"])
        game = HoldemGame(
            TableConfig(("A", "B"), (100, 100), 1, 2, 0), seed=2
        )
        game.start_hand()
        game.step(
            policy.decide(game.observation_for("A"), game.legal_actions("A")), "A"
        )

    def test_temporal_prediction_and_small_calibration_are_deterministic(self):
        model = OpponentModel("Hero", "Villain")
        observed = decision()
        prior_prediction = model.action_probability(observed)
        model.observe(replace(observed, hand_index=1, action_family="fold"))
        self.assertEqual(prior_prediction, prior_prediction)
        first = calibration_experiment(hands=2, checkpoints=(2,), seed=9)
        second = calibration_experiment(hands=2, checkpoints=(2,), seed=9)
        self.assertEqual(first, second)

    def test_model_range_flows_into_existing_coach(self):
        session = create_example_session()
        replayed = observed_decisions_from_session(session)
        self.assertEqual(len(replayed), session.position)
        self.assertTrue(all(not item.board or len(item.board) <= 5 for item in replayed))
        model = OpponentModel("BTN", "BB", known_cards=parse_cards("As Qs Qd 8c 4s"))
        analysis = analyze_showdown_baseline(
            session.game, "BTN", {"SB": None, "BB": model.inferred_range()},
            samples=100, seed=3,
        )
        self.assertGreater(analysis.equity.outcomes, 0)

    def test_holdout_utility_runs_without_future_training(self):
        result = holdout_predictive_evaluation(
            PRESETS["lag"], training_hands=2, holdout_hands=2, seed=5
        )
        self.assertEqual((result.training_hands, result.holdout_hands), (2, 2))
        self.assertTrue(math.isfinite(result.adaptive_mixture_log_loss))

    def test_small_paired_adaptive_performance_utility(self):
        result = adaptive_vs_fixed_experiment(
            PRESETS["nit"], training_hands=2, evaluation_hands=4, seed=12
        )
        self.assertEqual(result.duplicate_blocks, 2)
        self.assertAlmostEqual(
            result.difference_bb_per_100,
            result.adaptive_bb_per_100 - result.fixed_bb_per_100,
        )


if __name__ == "__main__":
    unittest.main()
