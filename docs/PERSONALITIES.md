# Synthetic Poker Personalities

These presets are synthetic research configurations, not estimates of
real-player population statistics. A personality is a fixed policy configuration;
it is not an opponent model inferred from observed behavior.

Every `PersonalityAgent` receives only `PlayerObservation` and `LegalActions`.
It extracts inexpensive objective features, constructs a normalized distribution
over legal action families, samples with an agent-local seeded RNG, converts the
family to a legal engine action, and emits a deterministic `DecisionTrace`.

## Features and buckets

Features include position, active players, pot/call/pot odds, effective stack,
SPR, current-street raises/aggressor, canonical preflop class, made-hand category,
flush/open-ended/gutshot draws, and objective board texture. The previous
aggressor is the player who made the most recent bet or raise on the street, even
after intervening calls or checks. Straight draws are classified by distinct
rank values that complete a straight: two or more is open-ended, exactly one is
a gutshot, and an already-made straight is not a draw.

- `AIR`: evaluated high card without a recognized draw.
- `DRAW`: flush or open-ended straight draw without a made pair.
- `MEDIUM_MADE`: one pair.
- `STRONG_MADE`: two pair or trips.
- `MONSTER`: straight or better.
- `WEAK_MADE`: fallback for incomplete/non-evaluated states.

No simulation decision calls the Monte Carlo equity calculator.

## Presets

- **Nit:** narrow position-aware opening ranges, low calls and bluffs, moderate
  value aggression, 50%/75% postflop sizes.
- **Tight Aggressive (TAG):** wider than Nit, high raise preference inside its
  opening range, moderate calls/bluffs, 50%/75% sizing.
- **Loose Aggressive (LAG):** broad late-position ranges, high aggression and
  bluffing, 33%/75%/125% sizing.
- **Calling Station:** very broad participation, independently high call and low
  aggression/bluff parameters, predominantly 50% sizing when aggressive.
- **Maniac:** broad ranges, very high aggression and bluff parameters,
  75%/100%/150% sizing.
- **Bluff Heavy:** LAG-like ranges with especially high explicit pure-bluff
  frequency and elevated semi-bluff multiplier.

Presets instantiate `StrategyProfile`; policy code never branches on a preset
name. Profiles serialize to explicit dictionaries and support safe replacement
of selected sweep parameters. Construction validates probability-like frequency
fields in `[0, 1]`, non-negative finite behavioral weights, positive finite size
choices and weights, a finite non-negative semi-bluff multiplier, and every
configured range expression. Invalid sweep values fail immediately.

## Preflop and sizing model

Position names are `BTN/SB,BB` heads-up; `BTN,SB,BB` three-handed; then CO, HJ,
and UTG are added as table size grows. Profiles distinguish unopened pots, one
prior raise, and multiple raises. Range membership and configured frequencies
produce fold/call/raise probabilities. All preflop ranges are parsed once when an
agent is created and stored as immutable hand-class sets. A legal free check
always removes folding from the distribution for all shipped profiles.

Postflop `fold_weights`, `call_weights`, and `aggression_weights` are independent
behavioral weights, not literal probabilities. The agent applies pure-bluff or
semi-bluff modifiers, removes illegal action families, then normalizes the
remaining weights. Consequently changing one configured tendency changes its
normalized share without silently rewriting either of the other tendencies.

Open sizes are weighted BB choices and re-raises use a simplified multiplier.
Postflop sizes are weighted pot fractions. Every target is clamped to the exact
engine-advertised minimum/maximum; unavailable aggressive families are removed
before normalization. These agents are interpretable heuristics, not GTO agents.
