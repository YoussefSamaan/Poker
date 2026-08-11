from __future__ import annotations

import argparse
from dataclasses import asdict
import json

from .cfr import KuhnCFR
from .scenario import HeadsUpScenario, ScenarioAnalyzer


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Offline poker research tools")
    commands = parser.add_subparsers(dest="command", required=True)

    analyze = commands.add_parser("analyze", help="analyze a heads-up hypothetical scenario")
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

    kuhn = commands.add_parser("train-kuhn", help="train the from-scratch Kuhn CFR baseline")
    kuhn.add_argument("--iterations", type=int, default=100_000)
    kuhn.add_argument("--seed", type=int, default=0)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.command == "train-kuhn":
        result = KuhnCFR().train(args.iterations, args.seed)
        print(json.dumps(asdict(result), indent=2, sort_keys=True))
        return

    scenario = HeadsUpScenario.from_text(
        hero=" ".join(args.hero),
        board=" ".join(args.board),
        pot=args.pot,
        to_call=args.to_call,
        hero_stack=args.hero_stack,
        villain_stack=args.villain_stack,
    )
    result = ScenarioAnalyzer().analyze(
        scenario,
        raise_costs=args.raise_cost,
        fold_equity=args.fold_equity,
        samples=args.samples,
        seed=args.seed,
        exact=True if args.exact else None,
    )
    print(json.dumps(asdict(result), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
