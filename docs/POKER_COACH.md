# Poker Coach v1

Poker Coach v1 adds range-aware, multiway, deterministic teaching baselines to
the offline trainer. These calculations are not GTO, Nash, or complete poker
solutions.

## Preflop range grammar

Expressions are comma- or whitespace-separated. A class can carry a positive
weight after `:`:

```text
QQ+:1, AKs:1, AQs:0.5, AJo+:0.25
```

- `AA` expands to six pair combinations.
- `AKs`, `AKo`, and `AK` expand to 4, 12, and 16 combinations.
- Pair `+` includes the named pair and every higher pair: `TT+` is
  `TT,JJ,QQ,KK,AA`.
- Non-pair `+` fixes the first, higher rank and raises the second rank through
  the rank immediately below it: `ATs+` is `ATs,AJs,AQs,AKs`; `KTs+` is
  `KTs,KJs,KQs`.
- Reversed, ambiguous, malformed, non-positive, duplicate, and overlapping
  classes are rejected rather than guessed.

Expansion produces the existing concrete `WeightedRange`. Known Hero and board
cards remove blocked combinations. Statistics report expanded combos, legal
combos, legal total weight, and raw coverage out of 1,326 possible starting
hands. Coverage is descriptive; no invented “top X%” ordering is used. The
13×13 matrix is a separate UI representation with pairs on the diagonal, suited
hands above it, and offsuit hands below it.

## Multiway equity

`MultiwayEquityCalculator` supports one to five opponents. Small joint spaces
enumerate every mutually compatible opponent-hand tuple and attach the product
of its seat-specific weights. Large spaces independently propose one weighted
hand per opponent and reject the entire tuple on any collision. Rejection
sampling exactly conditions the independent product distribution on physical
card compatibility, without the earlier seat-order bias. It then samples one
runout from the remaining deck. Impossible overlapping deals cannot enter a
world.

Small estimated spaces of at most 10,000 weighted deal/runout outcomes use exact
enumeration; larger spaces use seeded Monte Carlo. Hero's sample value is actual
pot share: `1` for an outright win, `1/n` when tied among `n` winners, and `0`
otherwise. Standard error is calculated from the empirical variance of that
pot-share variable. The normal 95% interval describes Monte Carlo sampling noise
only—not range correctness or opponent-model uncertainty.

## Simplified decision model

Fold is zero at the decision point because prior chips are sunk. The passive
baseline is explicitly **check/call to showdown**: after Hero checks or calls,
each remaining active player checks or calls the current wager up to its stack,
nobody raises, fixed ranges remain unchanged, and the board runs out. The
analyzer reuses `build_side_pots`. Every pot layer consumes the same complete
`ShowdownWorld`, so an ineligible short stack's physical cards still block other
hands and runout cards. Expected payouts are averaged and Hero's deterministic
new contribution is subtracted. Heads-up aggressive actions retain the older
explicit fold-frequency model. Multiway bet/raise EV is visibly unsupported.

Monte Carlo passive-action output includes payout/EV standard error and a normal
95% sampling interval. Exact output has zero simulation error. Neither interval
measures uncertainty in the supplied ranges.

Made-hand text reuses the existing evaluator. Board features are objective:
paired/double-paired, rainbow/two-tone/monotone, highest rank, and maximum gap
between adjacent distinct board ranks. No range-advantage or other subjective
jargon is inferred.

## Privacy and review

Player View constructs `PlayerTableView` solely from `PlayerObservation`; it has
no opponent hole-card or remaining-deck fields. Only the explicit Research View
branch requests `InternalState`. `DecisionReview` records the decision context,
selected range labels, chosen action, best supported baseline action, and
estimated baseline regret. Unsupported actions record no fabricated regret.

## Reference benchmark

On the development Mac using Python and the existing transparent evaluator,
seeded flop Monte Carlo measured approximately:

| Case | 10,000 samples |
|---|---:|
| Heads-up | 3.48 s |
| 3-way | 4.47 s |
| 6-way | 14.58 s |

These are orientation measurements, not a cross-machine performance guarantee.
The previous measurements were approximately 6.96 s, 12.06 s, and 29.01 s.
Reuse of the immutable deck and a bounded rank cache improved the corrected
implementation without native extensions. Run `python3 -m poker_ai.validation`
for deterministic small-space exact-versus-Monte-Carlo comparisons.
