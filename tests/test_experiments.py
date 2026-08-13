import json
import unittest

from poker_ai.agents import PRESETS
from poker_ai.experiments import (
    SimulationConfig,
    SimulationRunner,
    run_crossplay,
    summarize_metrics,
    sweep_parameter,
)
from poker_ai.experiments.records import SeatHandStats


class ExperimentTests(unittest.TestCase):
    def test_metrics_known_formula(self):
        rows = tuple(
            SeatHandStats(
                player_id="P",
                profile="X",
                participant_id="x",
                seat=0,
                position="BTN",
                net_chips=net,
                net_bb=net / 2,
                vpip=vpip,
                pfr=pfr,
                three_bet_opportunities=1,
                three_bets=three,
                folds=folds,
                checks=0,
                calls=calls,
                bets_raises=raises,
                postflop_actions=2,
                postflop_bets_raises=raises,
                player_reached_showdown=showdown,
            )
            for net, vpip, pfr, three, folds, calls, raises, showdown in (
                (2, True, True, 1, 0, 0, 1, True),
                (-2, False, False, 0, 1, 0, 0, False),
            )
        )
        metric = summarize_metrics("X", rows)
        self.assertEqual(metric.bb_per_100, 0)
        self.assertEqual(metric.vpip, 0.5)
        self.assertEqual(metric.pfr, 0.5)
        self.assertEqual(metric.three_bet_frequency, 0.5)
        self.assertAlmostEqual(metric.fold_frequency, 0.5)
        self.assertEqual(metric.showdown_rate, 0.5)

    def test_simulation_reproducibility_reset_rotation_and_zero_sum(self):
        config = SimulationConfig(
            (PRESETS["tag"], PRESETS["lag"], PRESETS["calling_station"]),
            hands=18,
            master_seed=33,
        )
        first = SimulationRunner(config).run()
        second = SimulationRunner(config).run()
        self.assertEqual(first, second)
        self.assertEqual(len(first.records), 18)
        self.assertEqual(
            sum(seat.net_chips for record in first.records for seat in record.seats), 0
        )
        self.assertEqual({record.button for record in first.records}, {0, 1, 2})
        self.assertEqual({record.assignments for record in first.records}.__len__(), 3)
        self.assertEqual(json.loads(first.to_json())["config"]["hands"], 18)
        self.assertIn("bb_per_100", first.metrics_csv())
        self.assertEqual(len(first.cumulative_net_bb()), 18)
        different = SimulationRunner(
            SimulationConfig(config.profiles, hands=18, master_seed=34)
        ).run()
        self.assertNotEqual(first.records, different.records)

    def test_duplicate_deals_and_crossplay(self):
        result = SimulationRunner(
            SimulationConfig(
                (PRESETS["nit"], PRESETS["lag"]),
                hands=4,
                master_seed=5,
                duplicate_deals=True,
            )
        ).run()
        self.assertEqual(result.records[0].deal_seed, result.records[1].deal_seed)
        self.assertNotEqual(
            result.records[0].assignments, result.records[1].assignments
        )
        self.assertAlmostEqual(sum(metric.total_net_bb for metric in result.metrics), 0)
        cross = run_crossplay(
            (PRESETS["nit"], PRESETS["tag"]), hands_per_matchup=12, seed=8
        )
        self.assertEqual(len(cross.matrix), 2)
        self.assertTrue(all(len(row) == 2 for row in cross.matrix))

    def test_parameter_sweep(self):
        results = sweep_parameter(
            PRESETS["tag"],
            "bluff_frequency",
            (0.0, 0.5),
            (PRESETS["calling_station"],),
            hands=6,
            seed=1,
        )
        self.assertEqual([value for value, _ in results], [0.0, 0.5])
        identities = [result.metrics[0].profile for _, result in results]
        self.assertEqual(len(set(identities)), 2)
        self.assertIn("bluff_frequency_0", identities[0])

    def test_optional_full_history_remains_json_exportable(self):
        result = SimulationRunner(
            SimulationConfig(
                (PRESETS["nit"], PRESETS["tag"]),
                hands=1,
                record_full_history=True,
            )
        ).run()
        self.assertTrue(result.records[0].history)
        self.assertEqual(json.loads(result.to_json())["records"][0]["hand_index"], 0)


if __name__ == "__main__":
    unittest.main()
