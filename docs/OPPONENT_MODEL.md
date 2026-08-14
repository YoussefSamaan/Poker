# Opponent Model v1.2

This is an offline, transparent statistical baseline. Synthetic personalities
generate behavior; opponent models consume public actions and infer uncertain
tendencies and ranges. They never receive the generating profile, opponent hole
cards, policy traces, future board cards, or deck state.

## Information boundary and identity

The three information layers are separate types:

- `ObservedDecision` contains table-public state, a globally scoped `HandKey`,
  legal action families captured before action, and the chosen public action.
- `ObserverContext` contains the observer identity, matching hand key, and only
  cards privately known to that observer (normally Hero's current hole cards).
- `ResearchDecisionLabels` contains synthetic ground truth for explicit
  privileged validation exports.

Simulator-owned public subject IDs are opaque (`public_player_0`,
`public_player_1`) and are independent even from custom research participant
names. `HandKey` combines
an opaque session ID with a local hand index. Training sessions receive a
persistent UUID; branches preserve it, while unrelated sessions get different
IDs even if both begin at hand zero.

`ExperimentResult.public_observation_json()` is the safe dataset export. It
contains neither research labels, hole cards, true profile names, fingerprints,
deck seeds, nor future runouts. `research_json()` is explicitly privileged and
includes synthetic ground truth.

## Historical state versus current-hand belief

`OpponentModel` now contains only persistent information:

- Bayesian behavioral sufficient statistics;
- historical synthetic-archetype posterior;
- model version and observation count;
- deduplication and in-progress learning state needed for serialization.

### Explicit hand lifecycle

Milestone 6.2 adds `begin_hand`, `observe_current_hand`, `current_hand_range`,
`commit_hand`, and `finish_hand`. The sole `HandModelState` snapshots the
historical model version and archetype posterior before the first active-hand
action. Transient analysis conditions that frozen prior exactly once. Updating
the long-term model during online play therefore cannot cause the same action to
be applied again when the range is reconstructed.

`commit_hand` is the only TrainingSession path into persistent history and is
deduplicated by `HandKey`. Scenario edits, undo, and branches remain hypothetical
until explicitly committed after the game is terminal. Every branch replays its
visible actions from the same hand-start prior.

Only one concrete `OpponentHandBelief` may be active. Finishing or beginning a
different hand evicts its per-archetype 1,326-combo distributions. Completed
history retains Bayesian counts, log posterior, and compact hand commit markers;
it never retains historical ranges or per-decision dedup records. Memory is
O(sufficient statistics + archetypes + compact hand keys + one active range),
rather than O(hands × archetypes × combos). Schema version 4 serializes at most
that one active concrete belief plus compact per-hand archetype checkpoints for
reproducible completed-hand replay.

It has no ambiguous `inferred_range()` method. A range exists only through an
`OpponentHandBelief`, created with a specific `HandKey` and `ObserverContext`, or
through the non-mutating API:

```python
inference = model.infer_range_for_hand(
    visible_decisions,
    observer_context=current_hero_context,
)
```

The returned `HandRangeInference` records the hand key, observer blockers,
historical model version, historical hand-start archetype prior, current
within-hand posterior, conditioned action families, weighted range, entropy,
effective size, top classes, and 13×13 matrix. It cannot accept decisions from
another hand.

Every new hand starts from the historical archetype posterior but a fresh
1,326-combo prior. Hero's current cards and the currently revealed board are
conditioned out. Cards from a previous hand do not remain blocked. Independent
multiway marginal beliefs remain separate; the existing showdown sampler handles
joint physical compatibility.

Poker Coach replays only the current TrainingSession prefix, reconstructs a
fresh current-hand belief without updating historical statistics, and sends that
`WeightedRange` to the existing equity engine. Undo, timeline navigation, and
branching therefore produce ranges from exactly the actions visible at that
point. The UI displays historical hand count separately from current actions and
observer blockers. Equity sampling intervals still exclude model uncertainty.

## Opportunity denominators

All action-specific rates require that action to have been legal:

- 3-bet and raise-versus-bet require `can_raise`;
- call-versus-open/bet/3-bet require `can_call`;
- fold-versus-open/bet/3-bet require `can_fold`;
- bet-when-checked-to requires `can_bet`.

A limp opportunity is an unraised preflop decision where calling requires
voluntarily adding chips (`can_call` and `to_call > 0`). A big blind taking a
free check is not a limp opportunity. VPIP/PFR remain hand-level outcomes.

Binary tendencies use configurable Beta-Binomial priors, defaulting to neutral
`Beta(1,1)`. Equal-tailed credible intervals numerically invert the regularized
incomplete beta function rather than using a small-sample normal approximation.
Preflop estimates retain exact position conditioning; postflop estimates retain
flop/turn/river conditioning.

## Range and archetype likelihood

For candidate hand `h` and public action `a`:

```text
P(h | a, context) ∝ P(a | h, context, profile) × P(h)
```

The action-family likelihood comes from the existing interpretable
`PersonalityAgent`, marginalized over all concrete legal hands. Archetype priors
default to a uniform synthetic mixture and update in log-space. These are beliefs
over shipped synthetic models—not real-player prevalence.

## Frozen, prequential, and adaptive evaluation

`holdout_predictive_evaluation()` reports two distinct protocols:

- frozen holdout trains on the first N hands, freezes historical model state,
  then scores the next M while still conditioning fresh within-hand ranges;
- prequential evaluation predicts each next action, records the model version,
  scores it, and only then observes/updates.

Its explicit baselines are:

- uniform archetype weights, held uniform while current-hand ranges condition;
- fixed TAG archetype with sequential current-hand range conditioning;
- learned adaptive mixture, frozen or prequential as labeled.

`adaptive_vs_fixed_experiment()` runs four policies on identical duplicate deals:

1. fixed TAG;
2. frozen-pretrained adaptive TAG;
3. online persistent adaptive TAG;
4. online reset-each-hand adaptive TAG.

The simulation's decision observer updates the persistent model only after a
Villain action has occurred. It passes Hero's cards through `ObserverContext`
from Hero's correct physical seat in each duplicate leg; Villain cards are never
sent. Paired block-level differences and intervals compare persistent/reset
policies with fixed TAG. No policy is called superior when its interval contains
zero.

Online duplicate evaluation is block-synchronous. Every leg in a duplicate
block runs before any observation from that block reaches historical learning.
Buffered observations are committed after the final leg, so an artificial
replay cannot teach a later leg about the same latent deal. Compared systems use
common deal, seat-assignment, and baseline policy RNG streams; only adaptive
state may introduce divergence.

## ML-safe public dataset

Simulator participants and public subjects are separate identities. Custom
participant names—even sweep IDs containing parameter names and values—remain
privileged. Public rows receive simulator-owned `public_player_N` subjects and
an opaque hashed dataset session ID.

`PublicObservationDataset` schema version 1 contains `PublicDecisionExample`
rows with safe split keys (`dataset_session_id`, `hand_index`,
`decision_sequence`, `public_subject_id`, and `correlation_group_id`), public
features, and the chosen action-family target. `OpponentFeatureVector` is a pure
function of `ObservedDecision`. It includes street/position, player count,
blind-normalized pot/call/effective-stack values, pot odds, stack-to-pot ratio,
the legal-action mask, preflop raises, previous-aggressor relationship, board
texture, and public history counts. It contains no private cards, profile labels,
research participant identity, traces, policy state, or generating parameters.

`grouped_train_validation_test_split` deterministically splits correlation
groups. Duplicate legs share one group; ordinary hands receive one group each.
Decisions from one hand must remain in the same data split.

## Milestone 6.2 development benchmark

`python3 benchmarks/opponent_model_lifecycle.py` separates state-lifecycle scale
from expensive action-likelihood throughput. On the development machine, compact
commit-marker state at 100/1,000/10,000 synthetic hand keys used approximately
13 KB/117 KB/1.38 MB of traced live memory and serialized to 5 KB/48 KB/489 KB.
No historical concrete ranges were retained. A separate 100-action,
single-archetype likelihood benchmark processed about 3.3 updates/s and 3.0
current-hand inferences/s; public feature extraction processed about 17.0k
decisions/s. Serialization processed about 6.8k rows/s and produced an 878 KB
JSON file for 1,000 examples. These are diagnostic Python baselines, not
production guarantees.

`AdaptiveExploitPolicy` remains deliberately small. On postflop AIR states it
shifts aggression according to fold-versus-bet minus call-versus-bet, capped at
15 percentage points and multiplied by `n/(n+40)`. Small samples stay close to
the base strategy.

## Validation utilities and UI

Calibration accepts configurable independent trials and returns accuracy, mean
true-profile posterior, log loss, and confusion counts. The tendency-convergence
utility estimates an emergent reference rate from a larger simulation and
compares shorter-prefix posterior means/credible intervals; it never equates
profile parameters directly with observed VPIP/PFR.

The Opponent Model tab keeps synthetic-generator research mode visually separate
from transient **Analyze current hand** and explicit **Commit completed hand to
history** operations. Current-session analysis replays only the visible prefix,
uses Hero cards only through `ObserverContext`, and never inspects Research View
opponent cards. Model state uses schema version 4 JSON and includes enough active
state for a mid-hand round trip plus compact historical hand-start checkpoints.

## Corrected development smoke results

A three-trial, 10-hand-per-profile calibration run (seed 301) deliberately shows
how unreliable tiny histories are. Top-1 accuracy was 67% for Nit and Maniac and
0% for TAG, LAG, Calling Station, and Bluff Heavy. Mean posterior mass on the
true model ranged from 15% to 36%. These are diagnostic smoke results, not model
quality claims; the configurable repeated-trial utility is intended for larger
runs.

For Calling Station, a 50-hand empirical reference run produced VPIP 24.1% over
29 observed opportunities. The 10-hand prefix posterior was 37.5% with a 95%
credible interval of 9.9%–71.0%; at 25 hands it remained 37.5% with interval
16.3%–61.6%. The reference lies inside both intervals, illustrating the intended
uncertainty display without equating profile parameters to emergent VPIP.

A 10-training-hand, 50-physical-hand duplicate smoke run executed fixed, frozen,
online-persistent, and reset-each-hand modes against every preset. The persistent
models processed 26–32 evaluation observations, proving online updates occurred,
but all four policies emitted identical actions in these small seeded samples;
paired differences were 0.0 bb/100. This null activation is not evidence of
equivalent long-run performance.
