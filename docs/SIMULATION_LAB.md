# Population Simulation Lab

`SimulationRunner` evaluates 2–6 synthetic personality policies using reset
cash-game hands. Every hand resets all seats to the configured stack, so an early
bust-out cannot remove a strategy from later evaluation. A `Participant` supplies
a stable experiment ID, display label, and profile. Automatically created public
IDs are opaque (`participant_0`, `participant_1`) rather than profile-derived.
Results aggregate by that ID, not by the profile's display name. Privileged
research metadata includes a compact SHA-256 fingerprint of the serialized
profile; the public observation export excludes it.

In the ordinary schedule, the profile-to-seat assignment stays fixed during each
N-hand block while the button visits every seat. The assignment rotates only
after that complete button cycle. Thus every participant sees every position
once per N hands and every physical seat once per N blocks. Partial ordinary
cycles have position imbalance of at most one.

A master seed is hashed into independent deck and per-seat policy seeds. The same
configuration and seed reproduce identical records without global randomness.
In duplicate mode, `hands` means physical hands and must be divisible by the
player count. Each N-leg duplicate block holds the deck seed and button fixed and
cyclically rotates participants through all N physical seats. Seat 0 therefore
receives the same private cards and latent runout in every leg, while every
participant receives that seat exactly once. Later blocks use a new deal and
rotate the button. Records contain explicit `duplicate_block_id` and
`duplicate_leg` values; incomplete correlated blocks are never included.

Compact records store deal seed, button, profile and participant assignments,
duplicate identifiers, per-seat net results and behavior counters, winners,
showdown flag, and action count. Full histories are optional. JSON metadata also
records schedule type, master seed, physical hands, independent duplicate blocks,
participant IDs/fingerprints, and the complete button schedule.

Use `public_observation_json()` for a public-only modeling dataset. Use
`research_json()` only when synthetic profile labels and true hole cards are
explicitly required for offline validation.

## Metrics

For per-hand net big blinds `x_i`:

```text
bb/100 = mean(x_i) × 100
SE(bb/100) = sample_sd(x_i) / sqrt(hands) × 100
95% CI = bb/100 ± 1.96 × SE(bb/100)
```

This is the ordinary per-hand interval and treats hands as independent. It is
retained for ordinary simulations, but is not used as the comparison interval
for correlated duplicate legs.

For a heads-up duplicate matchup, the independent observation is a block. For
participant A, each block value is the average of A's net BB across its two
swapped-seat legs. The sample SD, SE, and 95% CI are calculated across those
block values, then multiplied by 100. `MatchupResult` labels physical hands and
independent duplicate blocks separately. With no rake, B's estimate is exactly
the negative of A's and B's interval is `[-A_upper, -A_lower]`.

- **VPIP:** hand contained a voluntary preflop call or raise; forced blinds do
  not count.
- **PFR:** player made at least one voluntary preflop raise.
- **3-bet frequency:** 3-bets divided by player decisions made after exactly one
  voluntary preflop raise and before a second voluntary raise. Blinds do not
  count as raises; action after an open and 3-bet is not another opportunity.
- **Fold/check/call/bet-raise frequencies:** each family count divided by all
  decisions in those four families, including checks.
- **Postflop aggression frequency:** postflop bets and raises divided by all
  postflop actions.
- **Player showdown rate:** fraction of hands that went to showdown in which the
  participant had not folded. The separate hand record retains whether the hand
  itself went to showdown.

Metrics are grouped by participant ID and position. Cross-play evaluates each
unordered pair once using a balanced duplicate matchup, fills both cells from
that shared zero-sum result, mirrors confidence intervals, and sets the diagonal
to zero. `sweep_parameter` gives every parameter value a unique participant ID
and label so variants of the same base profile cannot merge.

## CLI

```bash
python3 -m poker_ai simulate --profiles tag,lag,calling_station --hands 10000 --stack-bb 100 --seed 42
python3 -m poker_ai crossplay --profiles nit,tag,lag,calling_station,maniac,bluff_heavy --hands-per-matchup 5000 --seed 42
```

Duplicate simulation hand counts must be complete blocks. The Simulation Lab
rounds its requested count down to a complete block and visibly reports both the
resulting physical-hand count and independent-block count. The CLI rejects an
incomplete block with an explanatory error.

## Development benchmark

On the development Mac, using the same seeded workloads before and after cached
immutable range precompilation (timings are single-run orientation figures):

| Experiment | Before | After |
|---|---:|---:|
| 1,000 heads-up hands | 2,327 hands/s | 2,020 hands/s |
| 10,000 heads-up hands | 2,244 hands/s | 2,223 hands/s |
| 1,000 six-player hands | 532 hands/s | 592 hands/s |

These pure-Python figures are orientation measurements, not cross-machine
guarantees. The hardened feature and record path adds work beyond range lookup,
so the 10,000-hand heads-up result is effectively flat while six-player
throughput improves by about 11%. Policies deliberately avoid per-decision equity
simulation.
