from dataclasses import asdict, replace
import json
import tempfile
from pathlib import Path
import unittest

import joblib
import numpy as np

from poker_ai.cards import parse_cards
from poker_ai.opponents import ObserverContext, OpponentModel
from poker_ai.opponents.dataset import grouped_train_validation_test_split
from poker_ai.opponents.learned import (
    ACTION_CLASSES,
    ContextActionModel,
    HandConditionedActionModel,
    HistoryAwareActionModel,
    LearnedRangeBelief,
    LegalFrequencyBaseline,
    build_metadata,
    candidate_hand_features,
    causal_history_examples,
    evaluate_action_predictions,
    evaluate_learned_range,
    generate_balanced_synthetic_dataset,
    grouped_log_loss_difference_bootstrap,
    legal_action_mask,
    load_trusted_local_artifact,
    save_learned_artifact,
    temporal_subject_split,
)


def key(example):
    return (
        example.dataset_session_id,
        example.hand_index,
        example.decision_sequence,
        example.public_subject_id,
    )


class LearnedOpponentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bundle = generate_balanced_synthetic_dataset(
            hands_per_personality=10,
            sessions_per_personality=2,
            seed=71,
        )
        cls.split = grouped_train_validation_test_split(
            cls.bundle.public_examples,
            validation_fraction=0.2,
            test_fraction=0.2,
            seed=4,
        )
        cls.context = ContextActionModel(seed=4).fit(cls.split.train)
        histories = tuple(
            value
            for value in causal_history_examples(cls.bundle.results)
            if value.public.public_subject_id == "public_player_1"
        )
        train_keys = {key(value) for value in cls.split.train}
        test_keys = {key(value) for value in cls.split.test}
        cls.history_train = tuple(
            value for value in histories if key(value.public) in train_keys
        )
        cls.history_test = tuple(
            value for value in histories if key(value.public) in test_keys
        )
        cls.history = HistoryAwareActionModel(seed=4).fit(cls.history_train)
        cls.research_train = tuple(
            value
            for value in cls.bundle.research.examples
            if key(value.public) in train_keys
        )
        cls.hand = HandConditionedActionModel(seed=4).fit(cls.research_train)

    def test_context_pipeline_unknown_category_legality_and_determinism(self):
        example = self.split.test[0]
        novel = replace(
            example,
            features=replace(example.features, position="UNSEEN_POSITION"),
        )
        first = self.context.predict_probabilities((novel,))[0]
        second = self.context.predict_probabilities((novel,))[0]
        self.assertTrue(np.allclose(first, second, atol=1e-12))
        self.assertAlmostEqual(float(first.sum()), 1.0)
        legality = (
            novel.features.can_fold,
            novel.features.can_check,
            novel.features.can_call,
            novel.features.can_bet,
            novel.features.can_raise,
        )
        self.assertTrue(all(value == 0 for value, legal in zip(first, legality) if not legal))
        self.assertEqual(tuple(ACTION_CLASSES), ("fold", "check", "call", "bet", "raise"))
        fallback = legal_action_mask(np.zeros(len(ACTION_CLASSES)), novel.features)
        self.assertAlmostEqual(sum(fallback.values()), 1.0)
        self.assertTrue(
            all(
                value == 0
                for action, value in fallback.items()
                if not dict(zip(ACTION_CLASSES, legality))[action]
            )
        )

    def test_public_model_feature_matrix_has_no_target_or_identity_leakage(self):
        forbidden = {
            "chosen_action_family",
            "action_amount",
            "bet_fraction_of_pot",
            "public_subject_id",
            "dataset_session_id",
            "hand_index",
            "decision_sequence",
            "correlation_group_id",
            "true_profile",
            "true_hole_cards",
        }
        self.assertTrue(forbidden.isdisjoint(self.context.feature_names))
        self.assertTrue(forbidden.isdisjoint(self.history.feature_names))
        public_json = json.dumps(asdict(self.split.train[0].features))
        self.assertNotIn("chosen_action_family", public_json)
        self.assertNotIn("true_hole_cards", public_json)

    def test_history_features_are_causal_and_session_subject_scoped(self):
        histories = causal_history_examples(self.bundle.results)
        first_by_identity = {}
        for value in histories:
            identity = (
                value.public.dataset_session_id,
                value.public.public_subject_id,
            )
            first_by_identity.setdefault(identity, value)
        for value in first_by_identity.values():
            features = value.history.as_dict()
            self.assertEqual(features["history_vpip_mean"], 0.5)
            self.assertEqual(features["history_vpip_log_opportunities"], 0.0)
        target = next(
            values
            for identity, values in first_by_identity.items()
            if identity[1] == "public_player_1"
        )
        self.assertEqual(target.history.as_dict()["history_pfr_mean"], 0.5)

    def test_temporal_and_grouped_splits_keep_hands_and_duplicates_atomic(self):
        temporal = temporal_subject_split(
            self.bundle.public_examples,
            validation_fraction=0.2,
            test_fraction=0.2,
        )
        for split in (self.split, temporal):
            groups = [
                {value.correlation_group_id for value in rows}
                for rows in (split.train, split.validation, split.test)
            ]
            self.assertTrue(groups[0].isdisjoint(groups[1]))
            self.assertTrue(groups[0].isdisjoint(groups[2]))
            self.assertTrue(groups[1].isdisjoint(groups[2]))

    def test_metrics_frequency_baseline_calibration_and_grouped_bootstrap(self):
        baseline = LegalFrequencyBaseline().fit(self.split.train)
        probabilities = baseline.predict_probabilities(self.split.test)
        metrics = evaluate_action_predictions(self.split.test, probabilities)
        self.assertGreaterEqual(metrics.log_loss, 0)
        self.assertGreaterEqual(metrics.multiclass_brier, 0)
        self.assertGreaterEqual(metrics.expected_calibration_error, 0)
        learned = self.context.predict_probabilities(self.split.test)
        difference = grouped_log_loss_difference_bootstrap(
            self.split.test, learned, probabilities, samples=50, seed=3
        )
        self.assertLessEqual(
            difference.confidence_interval_95[0],
            difference.confidence_interval_95[1],
        )

    def test_coefficients_and_trusted_local_persistence_round_trip(self):
        self.assertTrue(self.context.inspect_coefficients(limit=2))
        metadata = build_metadata(
            self.context,
            dataset_payload="deterministic-public-dataset",
            training_rows=len(self.split.train),
            training_correlation_groups=len(
                {value.correlation_group_id for value in self.split.train}
            ),
            metrics_summary={"log_loss": 1.0},
            seed=4,
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "context.joblib"
            save_learned_artifact(path, self.context, metadata)
            restored, restored_metadata = load_trusted_local_artifact(path)
            self.assertEqual(restored_metadata, metadata)
            expected = self.context.predict_probabilities(self.split.test)
            actual = restored.predict_probabilities(self.split.test)
            self.assertTrue(np.allclose(expected, actual, atol=1e-12))
            invalid = Path(directory) / "invalid.joblib"
            joblib.dump({"arbitrary": "pickle"}, invalid)
            with self.assertRaises(ValueError):
                load_trusted_local_artifact(invalid)
            mismatched = Path(directory) / "mismatched.joblib"
            payload = joblib.load(path)
            payload["metadata"]["schema_version"] = 999
            joblib.dump(payload, mismatched)
            with self.assertRaises(ValueError):
                load_trusted_local_artifact(mismatched)

    def test_hand_conditioned_builder_batch_prediction_and_learned_range(self):
        example = self.bundle.research.examples[0]
        self.assertEqual(len(example.true_hole_cards), 2)
        cards = parse_cards(example.true_hole_cards)
        candidate = candidate_hand_features(cards, ())
        batch = self.hand.predict_candidate_probabilities(
            example.public,
            example.history,
            (candidate, candidate),
        )
        self.assertEqual(batch.shape, (2, len(ACTION_CLASSES)))
        result = self.bundle.results[0]
        record = next(
            record
            for record in result.records
            if any(
                item.public_subject_id == "public_player_1"
                for item in record.observed_decisions
            )
        )
        observed = next(
            item
            for item in record.observed_decisions
            if item.public_subject_id == "public_player_1"
        )
        history = next(
            value.history
            for value in causal_history_examples((result,))
            if key(value.public)
            == (
                observed.hand_key.session_id,
                observed.hand_index,
                len(observed.history),
                observed.public_subject_id,
            )
        )
        belief = LearnedRangeBelief(self.hand)
        before = dict(belief.range.weights)
        belief.update(observed, history)
        self.assertAlmostEqual(sum(belief.range.weights.values()), 1.0)
        self.assertNotEqual(before, belief.range.weights)
        label = next(
            item
            for item in record.research_labels
            if item.public_subject_id == "public_player_1"
            and item.player_id == observed.player_id
        )
        evaluation = evaluate_learned_range(
            belief, tuple(label.true_hole_cards)
        )
        self.assertGreaterEqual(evaluation.true_combo_negative_log_probability, 0)
        blocked_cards = parse_cards("As Ks")
        blocked = LearnedRangeBelief(self.hand, known_cards=blocked_cards)
        self.assertTrue(
            all(
                not set(combo).intersection(blocked_cards)
                for combo in blocked.range.weights
            )
        )

    def test_old_hand_checkpoint_survives_later_commits_without_concrete_range(self):
        result = self.bundle.results[0]
        model = OpponentModel("observer", "public_player_1")
        first_record = next(
            record
            for record in result.records
            if any(
                value.public_subject_id == "public_player_1"
                for value in record.observed_decisions
            )
        )
        first_decisions = tuple(
            value
            for value in first_record.observed_decisions
            if value.public_subject_id == "public_player_1"
        )
        first_context = ObserverContext(
            "observer", first_decisions[0].hand_key, ()
        )
        model.begin_hand(first_context)
        original = dict(model.archetype_posterior)
        model.commit_hand(first_decisions, observer_context=first_context)
        model.finish_hand()
        for record in result.records[1:6]:
            decisions = tuple(
                value
                for value in record.observed_decisions
                if value.public_subject_id == "public_player_1"
            )
            if not decisions:
                continue
            private = ObserverContext("observer", decisions[0].hand_key, ())
            model.commit_hand(decisions, observer_context=private)
            model.finish_hand()
        replayed = model.infer_range_for_hand(
            first_decisions, observer_context=first_context
        )
        self.assertEqual(replayed.historical_archetype_prior, original)
        self.assertFalse(hasattr(model, "_learning_hands"))


if __name__ == "__main__":
    unittest.main()
