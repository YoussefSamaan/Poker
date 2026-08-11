# Poker Research Lab

An offline, reproducible foundation for studying Texas Hold'em decisions and
imperfect-information game solving. The project is for simulation, hypothetical
scenarios, and completed-hand review. It intentionally has no integration with a
live poker client and should not be used for real-time assistance.

## What works now

- A canonical 52-card model with strict duplicate-card validation.
- Correct five-card ranking and best-five-of-seven Hold'em evaluation.
- Heads-up equity against random, explicit, or weighted ranges.
- Exact enumeration when the state is small and seeded Monte Carlo otherwise.
- Win/tie/loss probabilities, equity, Monte Carlo standard error, and a 95%
  sampling interval.
- Decision-point EV for fold, check/call, and parameterized raises.
- Explicit fold-equity and fixed-range assumptions—heuristics are never labeled
  as GTO.
- Vanilla CFR implemented from scratch for Kuhn poker, checked against the
  known game value, and evaluated by exhaustive imperfect-information best
  responses (NashConv).

The older `Environment/` and `template-python-poker-bot/` directories remain for
compatibility and historical context. New research code belongs in `poker_ai/`.

## Quick start

No runtime dependencies are required for the research core.

```bash
python3 -m unittest discover -s tests -v

python3 -m poker_ai analyze \
  --hero As Qs \
  --board Qd 8c 4s \
  --pot 18 \
  --to-call 12 \
  --hero-stack 100 \
  --villain-stack 100 \
  --raise-cost 30 \
  --fold-equity 0.25 \
  --samples 20000 \
  --seed 7

python3 -m poker_ai train-kuhn --iterations 100000 --seed 7
```

Card input accepts ordinary notation (`As`, `Qd`, `Th`) and Unicode suits
(`A♠`, `Q♦`, `T♥`).

## Interpreting scenario EV

All action values are measured from the current decision point, so folding is
zero. For a pot `P`, call cost `C`, and showdown equity `q`, the implemented
one-step call model is:

```text
EV(call) = q(P + C) - C
required equity = C / (P + C)
```

`raise_cost` is the hero's total additional contribution from this decision,
including the call. A raise calculation assumes the supplied fold probability;
when called, both hands run to showdown with no later betting and the original
opponent range remains fixed. Those assumptions make the result auditable, but
they are not a no-limit equilibrium solution. Future work must update ranges by
action, model future streets, and solve the resulting subgame.

Example Python use with a weighted range:

```python
from poker_ai import HeadsUpScenario, ScenarioAnalyzer, WeightedRange

villain = WeightedRange.from_mapping({
    "QhJh": 3.0,
    "8s8d": 1.0,
    "AhKh": 0.5,
})
spot = HeadsUpScenario.from_text(
    hero="As Qs",
    board="Qd 8c 4s",
    pot=18,
    to_call=12,
    hero_stack=100,
    villain_stack=100,
    opponent_range=villain,
)
analysis = ScenarioAnalyzer().analyze(spot, samples=50_000, seed=42)
print(analysis.equity)
print(analysis.actions)
```

## Architecture and development sequence

```text
poker_ai/
  cards.py       immutable cards, parsing, seeded decks
  evaluation.py  transparent 5/6/7-card ranking
  ranges.py      weighted concrete opponent combinations
  equity.py      exact enumeration and Monte Carlo estimation
  scenario.py    auditable decision-point EV models
  cfr/kuhn.py    vanilla CFR learning baseline
tests/            mathematical and regression checks
```

The recommended sequence is deliberately incremental:

1. **Foundation (current milestone):** prove card, evaluator, equity, EV, and
   reproducibility correctness; use Kuhn to validate CFR math.
2. **Leduc solver:** introduce a general extensive-form game interface, CFR+,
   linear CFR, best response, NashConv/exploitability, and convergence plots.
3. **Hold'em simulator:** legal no-limit actions, blinds/button order, all-ins,
   side pots, public observations versus privileged engine state, complete hand
   histories, duplicate dealing, and deterministic replay.
4. **Offline coach baseline:** range notation (`QQ+`, `AKs`, percentages),
   Bayesian action updates, multi-street rollouts, action-size abstraction, and
   calibrated uncertainty. Compare every learned range model to simple priors.
5. **Advanced solving:** MCCFR on abstractions, depth-limited subgame solving,
   then neural value/advantage approximation only after tabular baselines and
   profiling show the need.
6. **Population evaluation:** duplicate matches, cross-play matrices, bb/100
   intervals, exploitability where tractable, bootstrap analysis, and ablations.

Do not jump directly to PPO, a Transformer, or an LLM policy. Standard policy
gradient methods do not by themselves address hidden information, and a strong
language explanation is not evidence that the numeric strategy is sound.

## Research basis

The technical direction follows the original
[CFR paper](https://papers.nips.cc/paper_files/paper/2007/hash/08d98638c6fcd194a4b1e6992063e944-Abstract.html),
the sample-based
[MCCFR paper](https://proceedings.neurips.cc/paper/2009/hash/00411460f7c92d2124a67ea0f4cb5f85-Abstract.html),
[Deep CFR](https://proceedings.mlr.press/v97/brown19b.html), and
[ReBeL](https://proceedings.neurips.cc/paper/2020/hash/c61f571dbd2fb949d3fe5ae1608dd48b-Abstract.html).
For independent game definitions and solver comparisons, use
[OpenSpiel](https://github.com/google-deepmind/open_spiel) as a validation
oracle rather than as a replacement for the learning implementation.

The practical lesson from these systems is architectural: strong poker agents
combine equilibrium-oriented learning with search, abstractions or function
approximation, and rigorous evaluation. Raw hand equity alone is not a strategy.

## Current limitations

- Scenario analysis is heads-up and assumes a fixed concrete-card range.
- Raise EV needs a user-supplied fold probability and is a one-step model.
- Monte Carlo intervals quantify sampling noise, not range/model uncertainty.
- Exact enumeration is intentionally simple and is not optimized for preflop
  exhaustive analysis.
- The legacy simulator still lacks side pots and a complete legal-action state
  machine; it should not yet be used for training.
- Kuhn CFR is a mathematical baseline, not a Hold'em policy.

These limitations are kept visible so later experiments can measure real
improvement instead of silently changing assumptions.
