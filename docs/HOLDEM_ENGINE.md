# No-Limit Hold'em engine

`poker_ai.holdem.HoldemGame` is the authoritative environment for future agents,
trainers, simulations, and offline analysis. `Environment/PokerGame.py` remains
legacy code and is not part of this API.

## State-machine contract

The engine accepts exactly one action per `step()` call. Only `current_player`
may act. `legal_actions(player_id)` describes the available action family and
its exact bounds. Invalid actions raise `IllegalAction`; they are never converted
to another action.

Actions are explicit objects:

- `Fold()`
- `Check()`
- `Call()` (the engine calculates a full or stack-limited call)
- `BetTo(amount)`
- `RaiseTo(amount)`

Bet and raise amounts are total contributions on the current street. Chips use
integer units throughout. For a 0.5/1 game, callers should normalize the chip
unit and configure blinds as 1/2.

## Seating, blinds, and action order

Seats are stored clockwise in `TableConfig.player_ids`. The configured button is
used for the first hand and advances to the next funded seat on later hands.

- With three or more players, the small and big blinds are the first two live
  seats left of the button. Preflop begins left of the big blind; postflop begins
  with the first active player left of the button.
- Heads-up, the button posts the small blind and acts first preflop. The big blind
  acts first on every postflop street.
- Folded, all-in, and empty seats are skipped.

If a multiway big blind is all-in for less than the configured blind, the full
big-blind amount remains the bring-in while at least two players can still wager.
In heads-up play, only the actually posted amount is contestable.

## Full raises and short all-ins

The engine tracks both the current highest street contribution and the size of
the most recent full raise. With blinds 1/2, a raise to 6 is an increment of 4,
so the next ordinary minimum is 10.

Betting-round state uses two independent sets:

1. `pending_players`: players who must respond before the round can close.
2. `raise_rights`: players whose action is open for a bet or raise.

A full bet or raise resets both sets for every other active player. A short
all-in updates the amount owed but does not reopen raising for a player who has
already acted against the prior wager. Players who had not acted retain their
raise rights. A prior check also retains the right to raise if a short opening
all-in occurs later. Multiple short all-ins reopen action once their cumulative
increase faced by a prior actor reaches the last full-raise size.

An undersized wager is legal only when it is the player's maximum effective
all-in. Aggressive actions are capped at the largest amount another non-folded
opponent can contest, so unmatched excess never enters the pot.

## All-ins, pots, and odd chips

`Call()` automatically commits the lesser of the amount owed and the player's
stack. A stack of zero changes the player's status to `ALL_IN`; the player remains
showdown-eligible but is skipped for future actions. Once no further betting
decision exists, all remaining community cards are dealt automatically.

Total contribution is tracked separately for every player. At showdown, sorted
contribution thresholds derive the main pot and every side pot. Folded chips stay
in those layers, but folded players are removed from eligibility. Each pot is
evaluated independently with `poker_ai.evaluation.evaluate_holdem`.

Tied pots are divided with integer arithmetic. Odd chips are awarded clockwise
starting with the first tied winner left of the button. There is no rake.

## Information boundary

`observation_for(player_id)` returns only public state and that player's own hole
cards. Opponent hole cards, undealt board cards, and deck order have no field in
`PlayerObservation`. The current actor also receives `LegalActions`; non-actors
do not.

`internal_state` is an explicitly privileged immutable snapshot for tests and
offline research. It includes all hole cards, remaining deck order, pending
actors, and raise rights. Agent policies must consume `PlayerObservation`, not
`InternalState`.

## History and transitions

Blinds and every player action produce immutable `ActionRecord` objects with:

- sequence number, street, player, and action type;
- amount paid and unambiguous bet/raise target;
- contribution, amount-to-call, pot, and stack before/after;
- explicit all-in indication.

`step()` returns a `Transition` containing that record, newly revealed cards,
whether the street changed, whether the hand ended, and the next actor. Terminal
state exposes a structured `HandResult` with per-pot eligibility, winners,
payouts, and final stacks.

## Determinism and scenarios

`seed=` uses a game-local random generator; global random state is never used.
A preset deck must contain all 52 unique cards in deal order. Physical burn cards
are deliberately omitted, making scenario deck positions direct and testable.

`ScenarioBuilder` accepts known hole cards, a board runout, and a sequence of
typed actions. It creates a compatible complete deck and replays every action
through the normal legality engine. Duplicate cards and illegal histories fail
instead of constructing an impossible state. The returned object is an ordinary
`HoldemGame` ready for `current_player`, `legal_actions`, `observation_for`, and
`step`.

## Invariants and scope

`assert_invariants()` checks after every engine transition that:

```text
sum(stacks) + sum(hand contributions) == chips at hand start
```

and that hole cards, board cards, and the remaining deck account for exactly 52
unique physical cards.

The engine supports ordinary 2–6 player Hold'em with blinds, but currently omits
antes, straddles, rake, burn cards, run-it-twice, sit-out re-entry rules, and
tournament-specific moving/dead-button edge cases. It contains no poker strategy
logic and no live-client integration.
