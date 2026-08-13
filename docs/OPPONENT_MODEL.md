# Opponent Model v1

This module is an offline, transparent statistical baseline. Synthetic
personalities generate actions; the opponent model observes public actions and
infers uncertain tendencies and ranges. It does not inspect the generating
profile, hidden cards, a policy trace, future board cards, or the deck.

## Data boundary and opportunities

`ObservedDecision` is captured immediately before an action, while legal actions
are known. It contains the public board, public player states and contributions,
pot and call amount, prior public action history, position, legal action families,
and the subsequently chosen public action/amount. `ResearchDecisionLabels` is a
separate synthetic-validation object containing the true profile and cards. An
`OpponentModel` accepts only `ObservedDecision`.

This fixes the former reconstructed 3-bet denominator: a decision is a 3-bet
opportunity only after exactly one voluntary preflop raise and when raising was
actually legal. Short all-in calls are therefore excluded. Fold/call/raise versus
bet, open raise, limp, and street aggression similarly use explicit decision
opportunities.

`observed_decisions_from_session()` replays a recorded TrainingSession prefix and
captures the public state at each historical decision. Because it replays in
chronological order, later actions and future board cards cannot affect earlier
records.

## Bayesian statistics

Binary tendencies use a configurable Beta-Binomial model. The default is the
neutral weak prior `Beta(1, 1)`—it contains no assumed poker-population rates.
For `s` successes and `f` failures, the posterior is `Beta(1+s, 1+f)`. Means,
variances, opportunity counts, and equal-tailed 95% credible intervals are
exposed. Intervals use numerical inversion of the regularized incomplete beta
function, rather than an inaccurate small-sample normal approximation.

Preflop estimates are conditioned on the existing exact position labels.
Postflop bet-when-checked-to, fold/call/raise-versus-bet, and aggression estimates
are separated by flop, turn, and river. Zero-opportunity estimates visibly remain
prior-dominated with wide intervals.

## Range and archetype inference

`RangeBelief` stores normalized weights over concrete two-card combinations. For
an observed action `a` and candidate hand `h`, it applies:

```text
P(h | a, context) ∝ P(a | h, context, profile) × P(h)
```

The likelihood comes from the existing interpretable `PersonalityAgent` action
distribution after inserting only the candidate hidden hand. Updates compose
within a hand. Hero-known cards and the currently revealed board remove blocked
combinations; other opponents' unknown cards do not. Separate models retain
independent marginal beliefs in multiway pots, leaving joint physical
conditioning to the existing showdown sampler.

The range summary reports normalized class weights, a 13×13 matrix, entropy in
nats, and effective range size `exp(H)`. An opponent model maintains a separate
belief for each shipped synthetic archetype. It marginalizes action likelihood
over every legal hidden combo and updates uniform archetype priors in log-space
with a small numerical likelihood floor. Output is a posterior mixture—not a
hard label. These probabilities describe synthetic models, not real-player
prevalence.

## Prediction, adaptation, and evaluation

`OpponentModel.action_probability()` makes a prediction without mutating state,
supporting sequential train/holdout evaluation. Development utilities provide
archetype convergence rows, confusion matrices, future-action log loss, and a
duplicate-block paired adaptive-versus-fixed comparison.

`AdaptiveExploitPolicy` composes with a base `PersonalityAgent`. On postflop AIR
states it modestly shifts aggression according to posterior fold-versus-bet minus
call-versus-bet. The shift is capped at 15 percentage points by default and
multiplied by `n / (n + 40)`, so tiny samples remain close to the base policy.
It is an interpretable experimental exploit policy, not an optimal or GTO agent.

`OpponentModelTable` keeps one model per opponent and supports persistent learning
or a reset-each-hand ablation. `SimulationRunner` exposes policy-factory and
public-decision observer hooks without coupling modeling logic to `HoldemGame`.
Model state round-trips through versioned JSON.

## Poker Coach and uncertainty

The Trainer's Opponent Model tab displays Bayesian estimates, archetype mixture,
range entropy/effective size, and the weighted matrix. When matching models are
available, Poker Coach can select `Opponent Model v1` instead of manual ranges
and passes the resulting `WeightedRange` to the existing equity engine.

The equity interval measures exact/Monte Carlo showdown sampling uncertainty. It
does **not** include opponent-model or archetype uncertainty; the UI states this
provenance explicitly.

## Development smoke results

A deterministic 10-hand-per-profile calibration smoke run (seed 31) assigned the
following posterior probability to the true synthetic archetype:

| True generator | True-model posterior | Top model |
|---|---:|---|
| Nit | 32.7% | Nit |
| TAG | 23.8% | Nit |
| LAG | 27.3% | LAG |
| Calling Station | 96.3% | Calling Station |
| Maniac | 90.0% | Maniac |
| Bluff Heavy | 11.9% | Nit |

This intentionally small run demonstrates both learnability and ambiguity; it is
not a performance claim. TAG/Nit and Bluff Heavy overlap heavily at this sample
size. A 10-train/10-holdout smoke run produced adaptive mixture log loss 0.128
for Nit, 0.289 for LAG, and 0.208 for Calling Station. Tiny holdouts are noisy.

In a 50-training-hand, 200-physical-hand paired smoke comparison, the bounded
adaptive TAG and fixed TAG produced identical actions and 0.0 bb/100 differences
against both Nit and Calling Station. The evidence was insufficient to activate
a policy change in those particular seeded samples. This is a useful null result,
not evidence of equivalence; larger preregistered runs should use the provided
utilities.
