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
- An authoritative 2–6 player No-Limit Hold'em state machine with one-action-at-
  a-time control, legal-action discovery, real action order, full/short raises,
  all-ins, side pots, deterministic odd chips, and structured hand histories.
- Strict separation between privileged engine state and leak-free player
  observations, plus deterministic seeds and exact scenario replay.
- A replayable training-session layer with undo, timeline navigation, independent
  branches, manual/policy seats, JSON persistence, and a local Streamlit trainer.
- Poker Coach v1 range notation and 13×13 matrices, blocker-aware weighted
  expansion, multiway equity with fractional tie shares, public-safe Player View,
  objective board/hand descriptions, range sensitivity, and decision reviews.
- Product-conditioned joint opponent sampling, shared showdown worlds for every
  side-pot layer, chip-denominated passive-EV uncertainty, and interactive
  quick/standard/custom precision controls.
- Serializable synthetic personality profiles, explainable seeded agents, and an
  independent-hand population simulation lab with behavior, position, bb/100,
  uncertainty, duplicate deals, cross-play, sweeps, JSON/CSV, UI, and CLI access.
- Versioned public ML datasets, causal opponent-history features, interpretable
  sklearn logistic baselines, grouped/temporal evaluation, calibrated metrics,
  trusted-local artifacts, and batched learned hidden-hand range updates.
- Vanilla CFR implemented from scratch for Kuhn poker, checked against the
  known game value, and evaluated by exhaustive imperfect-information best
  responses (NashConv).

The older `Environment/` and `template-python-poker-bot/` directories remain for
compatibility and historical context. New research code belongs in `poker_ai/`.

## Quick start

Install the package and its CPU scikit-learn dependency:

```bash
python3 -m pip install -e .
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

## Running the trainer

Install the optional local UI dependency and launch Streamlit:

```bash
python3 -m pip install -e '.[trainer]'
streamlit run poker_ai/trainer_app.py
```

The trainer starts with the documented three-player `As Qs` / `Qd 8c 4s`
decision. It can also create deterministic 2–6 player hands with one human seat
and check/call or seeded-random policies. **Player View** hides opponents' cards;
the clearly marked **Research View** reveals privileged cards for offline study.
Policies always receive only their own `PlayerObservation`, regardless of that
display toggle.

Use **Next AI action** for one automated transition or **Auto-play until human**
to stop at the next manual decision. Undo, redo, direct timeline navigation, and
**Branch here** all reconstruct state by replaying the original deal and actions.
The Scenario Builder accepts ordinary card notation and one legal action per line
without requiring JSON. Engine validation rejects duplicate cards, impossible
boards, illegal actions, and stack violations.

Sessions can be exported and imported as versioned JSON. At heads-up decision
points, the baseline panel reports equity against a random or explicit weighted
concrete-card range, pot odds, and one-step EV for convenient legal bet/raise
sizes. This is an assumption-bound reference calculation—not GTO, a Nash
strategy, or a full poker solver. Aggressive-action EV remains heads-up only.

Poker Coach v1 also accepts conventional ranges such as `QQ+, AKs, AKo` and
independent ranges for every opponent. Multiway fold/check/call values use a
simplified check/call-to-showdown model with side-pot eligibility. All opponents'
physical cards and one runout remain fixed across every pot layer in a sampled
world. Multiway raise EV remains intentionally unsupported. See
[`docs/POKER_COACH.md`](docs/POKER_COACH.md) for grammar, algorithms, privacy,
uncertainty, assumptions, and benchmark results.

Synthetic agents and experiments are documented in
[`docs/PERSONALITIES.md`](docs/PERSONALITIES.md) and
[`docs/SIMULATION_LAB.md`](docs/SIMULATION_LAB.md). These fixed illustrative
policies are not inferred opponent models and are not claims about real poker
populations.

Opponent Model v1 learns Bayesian public-action tendencies, maintains
blocker-aware action-conditioned ranges, infers a probability mixture over the
synthetic archetypes, and can feed inferred ranges into Poker Coach. See
[`docs/OPPONENT_MODEL.md`](docs/OPPONENT_MODEL.md).

Learned Opponent Model v2 provides context-only, causal-history, and privileged
synthetic hand-conditioned logistic models without replacing v1. The current
acceptance experiment is a negative/null result, so v1 remains the baseline. See
[`docs/LEARNED_OPPONENT_MODEL.md`](docs/LEARNED_OPPONENT_MODEL.md) and the
machine-readable [`docs/MILESTONE7_RESULTS.json`](docs/MILESTONE7_RESULTS.json).

## No-Limit Hold'em engine

The new engine never asks a player object to act and never silently repairs an
illegal action. A caller controls every transition:

```python
from poker_ai.holdem import Call, CheckCallPolicy, HoldemGame, TableConfig

game = HoldemGame(
    TableConfig(("Alice", "Bob", "Carol"), (200, 200, 200), 1, 2, button=0),
    seed=42,
)
policy = CheckCallPolicy()
game.start_hand()

while not game.is_terminal:
    player_id = game.current_player
    observation = game.observation_for(player_id)
    action = policy.decide(observation, observation.legal_actions)
    transition = game.step(action)

print(game.result)
```

`BetTo(12)` and `RaiseTo(12)` always mean a total street contribution of 12,
not 12 additional chips. All chip accounting uses integers. See
[`docs/HOLDEM_ENGINE.md`](docs/HOLDEM_ENGINE.md) for rule semantics, observations,
side pots, replay, and the short-all-in reopening model.

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
  holdem/         authoritative no-limit state machine and replay tools
  training/       replayable sessions and decision-analysis adapter
  trainer_app.py  thin local Streamlit UI
tests/            mathematical and regression checks
```

The recommended sequence is deliberately incremental:

1. **Mathematical/research primitives — done:** cards, evaluator, equity, simple
   EV, weighted ranges, and Kuhn CFR.
2. **No-Limit Hold'em engine — done:** step API, complete betting rules, all-ins,
   side pots, observations, invariants, histories, and deterministic replay.
3. **Interactive scenario/trainer UI — done:** offline state entry, action
   stepping, replay/branching, persistence, and transparent heads-up baselines.
4. **Poker Coach v1 — done:** range notation (`QQ+`, `AKs`), multiway equity,
   side-pot-aware passive baselines, explanations, and sampling uncertainty.
5. **Parameterized synthetic personalities and simulation — done:** traced
   heuristic agents, independent-hand evaluation, cross-play, sweeps, and metrics.
6. **Opponent Model v1 — done:** public decision opportunities, Bayesian
   tendencies, sequential range inference, synthetic-archetype mixtures,
   holdout evaluation, and a bounded adaptive-policy baseline.
7. **General CFR interface, Leduc, CFR+, and MCCFR.**
8. **Hold'em abstraction and depth-limited subgame solving.**
9. **Neural methods only where profiling and tabular baselines justify them.**
10. **Rigorous population/statistical evaluation:** duplicate matches, cross-play
    matrices, bb/100 intervals, bootstrap analysis, and ablations.

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
- `Environment/PokerGame.py` remains a legacy prototype. New training and UI
  code must use `poker_ai.holdem.HoldemGame`.
- Kuhn CFR is a mathematical baseline, not a Hold'em policy.
- The Hold'em engine currently models cash-style blinds without antes, rake,
  straddles, burn cards, or tournament-specific dead-button rules.

These limitations are kept visible so later experiments can measure real
improvement instead of silently changing assumptions.
