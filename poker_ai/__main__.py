from __future__ import annotations

import argparse
from dataclasses import asdict
import json

from .cfr import KuhnCFR
from .agents import PRESETS
from .experiments import SimulationConfig, SimulationRunner, run_crossplay
from .scenario import HeadsUpScenario, ScenarioAnalyzer


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Offline poker research tools")
    commands = parser.add_subparsers(dest="command", required=True)

    analyze = commands.add_parser(
        "analyze", help="analyze a heads-up hypothetical scenario"
    )
    analyze.add_argument("--hero", nargs=2, required=True, metavar=("CARD1", "CARD2"))
    analyze.add_argument("--board", nargs="*", default=[])
    analyze.add_argument("--pot", type=float, required=True)
    analyze.add_argument("--to-call", type=float, required=True)
    analyze.add_argument("--hero-stack", type=float, required=True)
    analyze.add_argument("--villain-stack", type=float, required=True)
    analyze.add_argument("--raise-cost", type=float, action="append", default=[])
    analyze.add_argument("--fold-equity", type=float, default=0.0)
    analyze.add_argument("--samples", type=int, default=20_000)
    analyze.add_argument("--seed", type=int, default=0)
    analyze.add_argument("--exact", action="store_true")

    kuhn = commands.add_parser(
        "train-kuhn", help="train the from-scratch Kuhn CFR baseline"
    )
    kuhn.add_argument("--iterations", type=int, default=100_000)
    kuhn.add_argument("--seed", type=int, default=0)
    simulate = commands.add_parser(
        "simulate", help="run independent-hand synthetic personality simulation"
    )
    simulate.add_argument(
        "--profiles", required=True, help="comma-separated preset keys"
    )
    simulate.add_argument("--hands", type=int, default=1_000)
    simulate.add_argument("--stack-bb", type=int, default=100)
    simulate.add_argument("--seed", type=int, default=0)
    simulate.add_argument("--duplicate-deals", action="store_true")
    crossplay = commands.add_parser(
        "crossplay", help="run heads-up synthetic personality cross-play"
    )
    crossplay.add_argument("--profiles", required=True)
    crossplay.add_argument("--hands-per-matchup", type=int, default=1_000)
    crossplay.add_argument("--seed", type=int, default=0)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.command == "train-kuhn":
        kuhn_result = KuhnCFR().train(args.iterations, args.seed)
        print(json.dumps(asdict(kuhn_result), indent=2, sort_keys=True))
        return
    if args.command in {"simulate", "crossplay"}:
        try:
            profiles = tuple(PRESETS[key.strip()] for key in args.profiles.split(","))
        except KeyError as error:
            raise SystemExit(f"unknown profile key: {error.args[0]}") from error
        if args.command == "simulate":
            simulation_result = SimulationRunner(
                SimulationConfig(
                    profiles,
                    hands=args.hands,
                    stack_bb=args.stack_bb,
                    master_seed=args.seed,
                    duplicate_deals=args.duplicate_deals,
                )
            ).run()
            print(simulation_result.to_json())
        else:
            crossplay_result = run_crossplay(
                profiles, hands_per_matchup=args.hands_per_matchup, seed=args.seed
            )
            print(json.dumps(asdict(crossplay_result), indent=2, sort_keys=True))
        return

    scenario = HeadsUpScenario.from_text(
        hero=" ".join(args.hero),
        board=" ".join(args.board),
        pot=args.pot,
        to_call=args.to_call,
        hero_stack=args.hero_stack,
        villain_stack=args.villain_stack,
    )
    analysis_result = ScenarioAnalyzer().analyze(
        scenario,
        raise_costs=args.raise_cost,
        fold_equity=args.fold_equity,
        samples=args.samples,
        seed=args.seed,
        exact=True if args.exact else None,
    )
    print(json.dumps(asdict(analysis_result), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
