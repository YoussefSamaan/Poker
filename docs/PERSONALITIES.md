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
flush/open-ended draws, and objective board texture. Postflop buckets are:

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
of selected sweep parameters.

## Preflop and sizing model

Position names are `BTN/SB,BB` heads-up; `BTN,SB,BB` three-handed; then CO, HJ,
and UTG are added as table size grows. Profiles distinguish unopened pots, one
prior raise, and multiple raises. Range membership and configured frequencies
produce fold/call/raise probabilities.

Open sizes are weighted BB choices and re-raises use a simplified multiplier.
Postflop sizes are weighted pot fractions. Every target is clamped to the exact
engine-advertised minimum/maximum; unavailable aggressive families are removed
before normalization. These agents are interpretable heuristics, not GTO agents.
