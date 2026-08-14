from dataclasses import fields, replace
import math
import unittest

from poker_ai.agents import PRESETS, PersonalityAgent
from poker_ai.cards import parse_cards
from poker_ai.experiments import SimulationConfig, SimulationRunner
from poker_ai.experiments.schedule import Participant, participants_from_profiles
from poker_ai.experiments import run_crossplay
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
    HandKey,
    ObservedDecision,
    ObserverContext,
    OpponentModel,
    OpponentModelTable,
    OpponentStats,
    RangeBelief,
    ResearchDecisionLabels,
    observe_decision,
    observed_decisions_from_session,
    observer_context_from_session,
)
from poker_ai.opponents.validation import (
    adaptive_vs_fixed_experiment,
    calibration_summary,
    calibration_experiment,
    holdout_predictive_evaluation,
    tendency_convergence_experiment,
)
from poker_ai.training import analyze_showdown_baseline
from poker_ai.trainer_app import create_example_session


def decision(
    *, hand=0, action="raise", raises=0, can_raise=True, street=Street.PREFLOP,
    position="BTN/SB", board=(), player="Villain", to_call=2, can_bet=False,
    session="test",
):
    players = (
        PublicPlayerState("Hero", 0, 100, PlayerStatus.ACTIVE, 0, 0),
        PublicPlayerState(player, 1, 100, PlayerStatus.ACTIVE, 0, 0),
    )
    return ObservedDecision(
        HandKey(session, hand), player, player, street, position, tuple(board),
        "Hero", "Hero", player,
        players, (), 10, 2, to_call, True, to_call == 0, to_call > 0, can_bet,
        can_raise, raises, None, 2, action, None, None,
    )


def context(value: ObservedDecision, cards="As Ks", observer="Hero"):
    return ObserverContext(observer, value.hand_key, parse_cards(cards))


class OpportunityAndPrivacyTests(unittest.TestCase):
    def test_default_participant_ids_are_opaque(self):
        participants = participants_from_profiles((PRESETS["tag"], PRESETS["lag"]))
        identities = " ".join(item.participant_id for item in participants).lower()
        for revealing in ("tag", "lag", "tight", "loose", "aggressive"):
            self.assertNotIn(revealing, identities)

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

    def test_public_export_semantically_excludes_ground_truth(self):
        result = SimulationRunner(
            SimulationConfig((PRESETS["tag"], PRESETS["lag"]), hands=4, master_seed=22)
        ).run()
        public = result.public_observation_json()
        research = result.research_json()
        self.assertNotIn("Tight Aggressive", public)
        self.assertNotIn("Loose Aggressive", public)
        self.assertNotIn("fingerprint", public)
        for record in result.records:
            for label in record.research_labels:
                for card in label.true_hole_cards:
                    self.assertNotIn(f'"{card}"', public)
        self.assertIn("Tight Aggressive", research)
        self.assertIn("true_hole_cards", research)

    def test_adaptive_observer_context_contains_only_heros_current_cards(self):
        captured = []

        def collect(observed, private_context):
            if observed.participant_id == "villain":
                captured.append((observed, private_context))

        hero = Participant("hero", "Hero", PRESETS["tag"])
        villain = Participant("villain", "Opponent", PRESETS["lag"])
        result = SimulationRunner(
            SimulationConfig(
                (hero.profile, villain.profile),
                hands=2,
                participants=(hero, villain),
                session_id="observer-context-test",
            ),
            decision_observer=collect,
            observer_participant_id="hero",
        ).run()
        self.assertTrue(captured)
        for observed, private in captured:
            record = result.records[observed.hand_index]
            villain_label = next(
                label
                for label in record.research_labels
                if label.participant_id == "villain"
            )
            self.assertNotEqual(private.observer_known_cards, villain_label.true_hole_cards)

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

    def test_each_facing_action_and_limp_has_its_own_legal_denominator(self):
        stats = OpponentStats()
        short = decision(
            action="call", raises=1, can_raise=False, to_call=2
        )
        stats.observe(short)
        self.assertEqual(stats.estimate("fold_vs_open").opportunities, 1)
        self.assertEqual(stats.estimate("call_vs_open").opportunities, 1)
        self.assertEqual(stats.estimate("three_bet").opportunities, 0)
        big_blind_check = decision(
            hand=1, action="check", raises=0, can_raise=True, to_call=0
        )
        stats.observe(big_blind_check)
        self.assertEqual(stats.estimate("limp").opportunities, 0)
        small_blind_limp = decision(
            hand=2, action="call", raises=0, can_raise=True, to_call=1
        )
        stats.observe(small_blind_limp)
        self.assertEqual(stats.estimate("limp").opportunities, 1)


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
            HandKey(f"real-raise-{players}-{seed}", 0),
            "Villain",
            observation,
            legal,
            RaiseTo(legal.min_raise_to),
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
        first.observe(observed, observer_context=context(observed))
        second.observe(observed, observer_context=context(observed))
        self.assertEqual(first.archetype_posterior, second.archetype_posterior)
        self.assertAlmostEqual(sum(first.archetype_posterior.values()), 1)
        self.assertNotEqual(first.archetype_posterior, initial)
        with self.assertRaises(TypeError):
            first.observe(
                ResearchDecisionLabels(
                    HandKey("research", 0), "Villain", "Villain", "Nit", ()
                ),
                observer_context=context(observed),
            )

    def test_multiway_models_and_json_round_trip(self):
        table = OpponentModelTable("Hero", ("Hero", "SB", "BB"))
        self.assertEqual(set(table.models), {"SB", "BB"})
        model = table.models["SB"]
        observed = decision(player="SB")
        model.observe(observed, observer_context=context(observed))
        restored = OpponentModel.from_json(model.to_json())
        self.assertEqual(restored.snapshot(), model.snapshot())
        reset = OpponentModelTable("Hero", ("SB",), reset_each_hand=True)
        first_hand = decision(player="SB", hand=0)
        second_hand = decision(player="SB", hand=1)
        reset.observe(first_hand, observer_context=context(first_hand))
        reset.observe(second_hand, observer_context=context(second_hand))
        self.assertEqual(reset.models["SB"].snapshot().hands_observed, 1)

    def test_observer_blockers_reset_per_hand_and_history_persists(self):
        model = OpponentModel("Hero", "Villain")
        first = replace(self._real_raise(2), hand_key=HandKey("session", 0))
        first_context = ObserverContext(
            "Hero", first.hand_key, parse_cards("As Ks")
        )
        model.observe(first, observer_context=first_context)
        historical = dict(model.archetype_posterior)
        first_range = model.infer_range_for_hand(
            (first,), observer_context=first_context
        ).weighted_range
        self.assertTrue(
            all(
                not set(combo.cards).intersection(parse_cards("As Ks"))
                for combo in first_range.combos
            )
        )

        second_context = ObserverContext(
            "Hero", HandKey("session", 1), parse_cards("7c 7d")
        )
        second = model.infer_range_for_hand(
            (), observer_context=second_context
        ).weighted_range
        self.assertTrue(any(parse_cards("As Ks")[0] in combo.cards for combo in second.combos))
        self.assertTrue(
            all(
                not set(combo.cards).intersection(parse_cards("7c 7d"))
                for combo in second.combos
            )
        )
        self.assertEqual(model.archetype_posterior, historical)
        self.assertFalse(hasattr(model, "inferred_range"))

    def test_same_local_index_from_two_sessions_counts_as_two_hands(self):
        model = OpponentModel("Hero", "Villain")
        for session in ("one", "two"):
            observed = decision(session=session)
            model.observe(observed, observer_context=context(observed))
        self.assertEqual(model.snapshot().hands_observed, 2)


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
        private = context(observed)
        prior_prediction = model.action_probability(
            observed, observer_context=private
        )
        future = replace(
            observed,
            hand_key=HandKey("test", 1),
            action_family="fold",
        )
        model.observe(future, observer_context=context(future))
        restored = OpponentModel("Hero", "Villain")
        self.assertEqual(
            prior_prediction,
            restored.action_probability(observed, observer_context=private),
        )
        first = calibration_experiment(hands=2, checkpoints=(2,), seed=9)
        second = calibration_experiment(hands=2, checkpoints=(2,), seed=9)
        self.assertEqual(first, second)

    def test_model_range_flows_into_existing_coach(self):
        session = create_example_session()
        replayed = observed_decisions_from_session(session)
        self.assertEqual(len(replayed), session.position)
        self.assertTrue(all(not item.board or len(item.board) <= 5 for item in replayed))
        model = OpponentModel("BTN", "BB")
        private = observer_context_from_session(session, "BTN")
        inference = model.infer_range_for_hand(
            replayed, observer_context=private
        )
        analysis = analyze_showdown_baseline(
            session.game, "BTN", {"SB": None, "BB": inference.weighted_range},
            samples=100, seed=3,
        )
        self.assertGreater(analysis.equity.outcomes, 0)

    def test_rewind_and_branch_reconstruct_different_current_ranges(self):
        session = create_example_session()
        model = OpponentModel("BTN", "BB")
        full = model.infer_range_for_hand(
            observed_decisions_from_session(session),
            observer_context=observer_context_from_session(session, "BTN"),
        )
        branch = session.branch(at_action=4)
        branch.act(Check())
        checked = model.infer_range_for_hand(
            observed_decisions_from_session(branch),
            observer_context=observer_context_from_session(branch, "BTN"),
        )
        self.assertNotEqual(full.summary.matrix, checked.summary.matrix)

    def test_holdout_utility_runs_without_future_training(self):
        result = holdout_predictive_evaluation(
            PRESETS["lag"], training_hands=2, holdout_hands=2, seed=5
        )
        self.assertEqual((result.training_hands, result.holdout_hands), (2, 2))
        self.assertTrue(math.isfinite(result.adaptive_prequential_log_loss))
        self.assertTrue(
            all(
                trace.model_version_after_observation
                == trace.model_version_before_prediction + 1
                for trace in result.prequential_trace
            )
        )

    def test_small_paired_adaptive_performance_utility(self):
        result = adaptive_vs_fixed_experiment(
            PRESETS["nit"], training_hands=2, evaluation_hands=4, seed=12
        )
        self.assertEqual(result.duplicate_blocks, 2)
        self.assertGreater(result.online_observations, 0)
        self.assertIsInstance(result.reset_each_hand_bb_per_100, float)

    def test_replicated_calibration_convergence_and_crossplay_naming(self):
        summaries = calibration_summary(hands=2, trials=2, seed=14)
        self.assertEqual(len(summaries), len(PRESETS))
        self.assertTrue(all(item.trials == 2 for item in summaries))
        convergence = tendency_convergence_experiment(
            PRESETS["calling_station"],
            reference_hands=4,
            checkpoints=(2,),
            seed=4,
        )
        self.assertEqual(len(convergence), 1)
        matchup = run_crossplay(
            (PRESETS["nit"], PRESETS["tag"]), hands_per_matchup=4, seed=2
        ).matchups[0]
        self.assertAlmostEqual(matchup.a_paired_bb_per_100, matchup.a_bb_per_100)
        self.assertFalse(hasattr(matchup, "paired_difference_bb_per_100"))


if __name__ == "__main__":
    unittest.main()
