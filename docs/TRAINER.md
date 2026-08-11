# Local Poker Trainer

The trainer is an offline study application over the authoritative
`HoldemGame`. `TrainingSession` stores the initial configuration, deterministic
seed or preset deal, seat controls, policy configuration, action timeline, and
current position. Undo and branch operations rebuild the game from those inputs;
they do not copy and mutate a second rules state.

## Launch

```bash
python3 -m pip install -e '.[trainer]'
streamlit run poker_ai/trainer_app.py
```

## Views and controls

- **Player View** shows the acting player's cards and masks every opponent.
- **Research View** is an explicit privileged display mode that reveals all dealt
  cards. It never changes policy inputs.
- Manual seats expose only actions returned by `LegalActions`. Bet and raise
  inputs are target total contributions on the current street (`BetTo` and
  `RaiseTo`), not additional chips.
- Policy seats use the deterministic check/call baseline or a seeded random-legal
  baseline. They can take exactly one step or continue only until a human acts.
- Undo, redo, go-to-position, and branch reconstruct the exact original cards,
  runout, stacks, and action state.

## Scenario builder and persistence

Configure 2–6 seat IDs, stacks, blinds, button, hero cards, board/runout, and an
action script in the Scenario Builder. An action line looks like `BTN raise_to 6`
or `SB call`. The domain builder and game perform card and action validation.

Exported JSON uses `schema_version: 1` and explicit DTOs for configuration,
cards, policies, actions, timeline position, and scenario metadata. It does not
serialize Python objects or arbitrary class internals.

## Analysis scope

Decision context is constructed from the acting player's leak-free observation.
The current bridge supports exactly one non-folded opponent. It reuses the
existing equity calculator, concrete `WeightedRange`, `HeadsUpScenario`, and
`ScenarioAnalyzer`. Candidate sizes are legal convenience points, not strategy
recommendations. EV assumes a fixed range, supplied fold frequency, and no future
betting; its regret values are relative only to actions in that simplified model.
