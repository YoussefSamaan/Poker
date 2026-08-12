# Population Simulation Lab

`SimulationRunner` evaluates 2–6 synthetic personality policies using independent
cash-game hands. Every hand resets all seats to the configured stack, so an early
bust-out cannot remove a strategy from later evaluation. Button and profile-to-
seat assignments rotate deterministically.

A master seed is hashed into independent deck and per-seat policy seeds. The same
configuration and seed reproduce identical records without global randomness.
Duplicate mode reuses each deal seed across adjacent rotated assignments. This is
a variance-reduction device; paired hands still have independent policy RNGs.

Compact records store deal seed, button, assignments, per-seat net results and
behavior counters, winners, showdown flag, and action count. Full histories are
optional. Results export to JSON and metrics to CSV.

## Metrics

For per-hand net big blinds `x_i`:

```text
bb/100 = mean(x_i) × 100
SE(bb/100) = sample_sd(x_i) / sqrt(hands) × 100
95% CI = bb/100 ± 1.96 × SE(bb/100)
```

The interval is an ordinary per-hand normal interval. Wide intervals must not be
presented as evidence that a positive point estimate is better.

- **VPIP:** hand contained a voluntary preflop call or raise; forced blinds do
  not count.
- **PFR:** player made at least one voluntary preflop raise.
- **3-bet frequency:** 3-bets divided by opportunities after one voluntary
  preflop raise.
- **Fold/call/bet-raise frequencies:** family count divided by voluntary action
  count.
- **Postflop aggression frequency:** postflop bets and raises divided by all
  postflop actions.
- **Showdown:** stored per hand as engine showdown status.

Metrics are also grouped by profile and position. Heads-up cross-play runs both
rotated orientations and returns bb/100 plus confidence intervals. Multiway
lineups use the same runner. `sweep_parameter` safely replaces an allowed scalar
profile parameter and runs otherwise identical experiments.

## CLI

```bash
python3 -m poker_ai simulate --profiles tag,lag,calling_station --hands 10000 --stack-bb 100 --seed 42
python3 -m poker_ai crossplay --profiles nit,tag,lag,calling_station,maniac,bluff_heavy --hands-per-matchup 5000 --seed 42
```

## Development benchmark

On the development Mac:

| Experiment | Throughput |
|---|---:|
| 1,000 heads-up hands | 2,327 hands/s; 7,263 actions/s |
| 10,000 heads-up hands | 2,244 hands/s; 7,069 actions/s |
| 1,000 six-player hands | 532 hands/s; 4,580 actions/s |

These pure-Python figures are orientation measurements, not cross-machine
guarantees. Policies deliberately avoid per-decision equity simulation.
