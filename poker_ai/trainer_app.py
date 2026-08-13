"""Local Streamlit hand trainer built on the replayable application layer."""

from __future__ import annotations

from typing import Any

from poker_ai.holdem import (
    Action,
    BetTo,
    Call,
    Check,
    Fold,
    RaiseTo,
    ScenarioBuilder,
    TableConfig,
)
from poker_ai.agents import PRESETS
from poker_ai.experiments import SimulationConfig, SimulationRunner
from poker_ai.experiments.schedule import Participant
from poker_ai.opponents import OpponentModel
from poker_ai.ranges import PreflopRange, WeightedRange
from poker_ai.training import (
    PolicyConfig,
    PolicyKind,
    TrainingSession,
    analyze_current_decision,
    analyze_showdown_baseline,
    board_features,
    capture_decision_review,
    compare_ranges,
    decision_context,
    describe_current_hand,
    player_table_view,
    range_matrix_rows,
    sensitivity_rows,
)


def create_new_hand_session(
    player_count: int = 3,
    stack: int = 200,
    hero_seat: int = 0,
    seed: int = 0,
    policy_kind: PolicyKind = PolicyKind.CHECK_CALL,
    personality: str | None = None,
) -> TrainingSession:
    names = tuple(f"P{seat + 1}" for seat in range(player_count))
    config = TableConfig(names, (stack,) * player_count, 1, 2, button=0)
    hero = names[hero_seat]
    selected_kind = PolicyKind(personality) if personality is not None else policy_kind
    policies = {
        player_id: PolicyConfig(selected_kind, seed + seat + 1)
        for seat, player_id in enumerate(names)
        if player_id != hero
    }
    session = TrainingSession.new_hand(
        config, seed=seed, human_players={hero}, policy_configs=policies
    )
    return session


def create_example_session() -> TrainingSession:
    """The documented 100-BB, three-player AsQs flop decision."""
    config = TableConfig(("BTN", "SB", "BB"), (200, 200, 200), 1, 2, button=0)
    builder = (
        ScenarioBuilder(config)
        .set_hole_cards("BTN", "As Qs")
        .set_board_runout("Qd 8c 4s")
        .action("BTN", RaiseTo(6))
        .action("SB", Call())
        .action("BB", Call())
        .action("SB", Check())
        .action("BB", BetTo(12))
    )
    policies = {
        "SB": PolicyConfig(PolicyKind.CHECK_CALL),
        "BB": PolicyConfig(PolicyKind.CHECK_CALL),
    }
    return TrainingSession.from_scenario(
        builder, human_players={"BTN"}, policy_configs=policies
    )


def parse_weighted_range(text: str) -> WeightedRange | None:
    """Parse ordinary ``AsKh:1`` lines; blank means a random legal range."""
    if not text.strip():
        return None
    mapping: dict[str, float] = {}
    for line_number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        try:
            combo, weight = line.split(":", 1)
            mapping[combo.strip()] = float(weight.strip())
        except ValueError as error:
            raise ValueError(f"range line {line_number} must be 'AsKh:1'") from error
    return WeightedRange.from_mapping(mapping)


def parse_action_script(text: str) -> tuple[tuple[str, Action], ...]:
    """Parse one action per line, e.g. ``BTN raise_to 6``."""
    actions: list[tuple[str, Action]] = []
    constructors = {"fold": Fold, "check": Check, "call": Call}
    for line_number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        parts = line.split()
        try:
            player_id, verb = parts[:2]
            if verb in constructors and len(parts) == 2:
                action = constructors[verb]()
            elif verb in {"bet_to", "raise_to"} and len(parts) == 3:
                amount = int(parts[2])
                action = BetTo(amount) if verb == "bet_to" else RaiseTo(amount)
            else:
                raise ValueError
        except (ValueError, IndexError) as error:
            raise ValueError(
                f"action line {line_number} must be 'PLAYER check/call/fold' or "
                "'PLAYER bet_to/raise_to AMOUNT'"
            ) from error
        actions.append((player_id, action))
    return tuple(actions)


def _card_text(cards: tuple[Any, ...]) -> str:
    return " ".join(str(card) for card in cards) or "—"


def _render_table(st: Any, session: TrainingSession, research_view: bool) -> None:
    game = session.game
    hero_id = next(
        player_id
        for player_id in session.config.player_ids
        if session.controls[player_id].value == "human"
    )
    observation = game.observation_for(hero_id)
    seats: tuple[tuple[Any, tuple[Any, ...] | None], ...]
    if research_view:
        # This is the only table-rendering path allowed to request privileged state.
        privileged = game.internal_state
        seats = tuple((player, player.hole_cards) for player in privileged.players)
    else:
        public = player_table_view(observation)
        seats = tuple((player, player.cards) for player in public.seats)
    st.subheader(f"{observation.street.value.title()} · pot {observation.pot}")
    st.markdown(f"Board: **{_card_text(observation.board)}**")
    columns = st.columns(len(seats))
    for column, (player, cards) in zip(columns, seats):
        with column:
            marker = " (BTN)" if player.player_id == observation.button_player else ""
            st.markdown(f"**{player.player_id}{marker}**")
            st.write(f"Stack: {player.stack}")
            st.write(f"Status: {player.status.value}")
            st.write(f"Street chips: {player.street_contribution}")
            st.write(f"Cards: {_card_text(cards) if cards else '🂠 🂠'}")


def _render_controls(st: Any, session: TrainingSession) -> None:
    actor = session.current_actor
    if session.game.is_terminal:
        st.success("Hand complete")
        st.write(session.game.result)
        return
    st.subheader(f"Action: {actor}")
    if session.needs_human_action:
        legal = session.available_actions()
        if legal is None:
            raise AssertionError("a human actor must have legal actions")
        columns = st.columns(3)
        if legal.can_fold and columns[0].button("Fold"):
            session.act(Fold())
            st.rerun()
        if legal.can_check and columns[1].button("Check"):
            session.act(Check())
            st.rerun()
        if legal.can_call and columns[2].button(f"Call {legal.call_amount}"):
            session.act(Call())
            st.rerun()
        minimum = legal.min_bet_to or legal.min_raise_to
        maximum = legal.max_bet_to or legal.max_raise_to
        if minimum is not None and maximum is not None:
            target = st.number_input(
                "Target total on this street", minimum, maximum, minimum
            )
            label = "Bet" if legal.can_bet else "Raise"
            if st.button(f"{label} to {target}"):
                session.act(BetTo(target) if legal.can_bet else RaiseTo(target))
                st.rerun()
    else:
        left, right = st.columns(2)
        if left.button("Next AI action"):
            session.next_policy_action()
            st.rerun()
        if right.button("Auto-play until human"):
            session.auto_play_until_human()
            st.rerun()
    if session.last_policy_trace is not None:
        trace = session.last_policy_trace
        with st.expander("Latest synthetic-agent DecisionTrace", expanded=True):
            st.write(f"**{trace.profile} — {trace.features.player_id}**")
            st.write(
                f"{trace.features.position}; {trace.features.street.value}; "
                f"hand {trace.features.hand_class}; bucket {trace.features.bucket.value}."
            )
            st.dataframe(
                [
                    {"action family": name, "probability": value}
                    for name, value in trace.probabilities
                ],
                hide_index=True,
            )
            for line in trace.rationale:
                st.write(line)


def _render_timeline(st: Any, session: TrainingSession) -> None:
    st.subheader("Replay timeline")
    if session.position < len(session.timeline):
        st.info(
            "Reconstructed earlier state: later recorded actions remain available to redo."
        )
    left, middle, right, restart = st.columns(4)
    if left.button("Undo", disabled=not session.can_undo):
        session.undo()
        st.rerun()
    if middle.button("Redo", disabled=not session.can_redo):
        session.redo()
        st.rerun()
    if right.button("Branch here"):
        st.session_state.training_session = session.branch()
        st.rerun()
    if restart.button("Restart hand"):
        session.goto_action(0)
        st.rerun()
    position = st.slider("Action position", 0, len(session.timeline), session.position)
    if position != session.position and st.button("Go to position"):
        session.goto_action(position)
        st.rerun()
    rows = []
    for index, record in enumerate(session.game.history, 1):
        rows.append(
            {
                "#": index,
                "street": record.street.value,
                "player": record.player_id,
                "action": record.action_type.value,
                "paid": record.amount_paid,
                "target": record.target_to,
                "pot": record.pot_after,
            }
        )
    st.dataframe(rows, use_container_width=True, hide_index=True)


def _render_analysis(st: Any, session: TrainingSession) -> None:
    actor = session.current_actor
    if actor is None or session.game.is_terminal:
        return
    st.subheader("Poker Coach v1")
    context = decision_context(session.game, actor)
    features = board_features(context.board)
    st.write(
        f"**Hero hand:** {describe_current_hand(context.hero_cards, context.board)}"
    )
    st.write(
        f"**Board:** {features.suit_texture}; paired={features.paired}; "
        f"highest={features.highest_rank or '—'}; max rank gap="
        f"{features.maximum_adjacent_rank_gap if features.maximum_adjacent_rank_gap is not None else '—'}"
    )
    st.write(
        f"**Pot odds:** call {context.to_call} into {context.pot}; "
        f"required equity {context.required_equity:.1%}."
    )
    model_map = st.session_state.get("opponent_models", {})
    model_available = all(opponent in model_map for opponent in context.opponent_ids)
    range_source = st.radio(
        "Range source",
        ("Manual", "Opponent Model v1"),
        horizontal=True,
        disabled=not model_available,
    )
    if not model_available:
        st.caption(
            "Build matching opponent models in the Opponent Model tab to enable "
            "inferred ranges."
        )
    advanced = st.toggle(
        "Advanced concrete-combo input", disabled=range_source != "Manual"
    )
    range_inputs: dict[str, str] = {}
    if range_source == "Manual":
        for opponent in context.opponent_ids:
            range_inputs[opponent] = st.text_area(
                f"{opponent} range",
                "QQ+, AKs, AKo" if not advanced else "AhKh:1\nQcQd:0.5",
                key=f"coach_range_{opponent}",
            )
    default_samples = (
        10_000
        if len(context.opponent_ids) == 1
        else max(1_000, 8_000 // len(context.opponent_ids))
    )
    precision = st.selectbox(
        "Estimate precision", ("Quick", "Standard", "Custom"), index=1
    )
    if precision == "Quick":
        samples = 1_000
        st.caption("Quick estimate: 1,000 samples")
    elif precision == "Standard":
        samples = default_samples
        st.caption(f"Standard estimate: {samples:,} samples")
    else:
        samples = st.number_input(
            "Monte Carlo samples", 100, 200_000, default_samples, 100
        )
    fold_equity = (
        st.slider("Heads-up assumed fold frequency", 0.0, 1.0, 0.0, 0.05)
        if len(context.opponent_ids) == 1
        else 0.0
    )
    if st.button("Run baseline analysis"):
        try:
            dead = context.hero_cards + context.board
            ranges = {}
            parsed_preflop = {}
            if range_source == "Opponent Model v1":
                ranges = {
                    opponent: model_map[opponent].inferred_range().without_blocked(dead)
                    for opponent in context.opponent_ids
                }
            else:
                for opponent, text in range_inputs.items():
                    if advanced:
                        ranges[opponent] = parse_weighted_range(text)
                    else:
                        parsed = PreflopRange.parse(text)
                        parsed_preflop[opponent] = parsed
                        ranges[opponent] = parsed.to_weighted_range(dead)
            result = analyze_showdown_baseline(
                session.game, actor, ranges, samples=samples, seed=0
            )
        except ValueError as error:
            st.error(str(error))
        else:
            st.warning(result.model_label)
            if range_source == "Opponent Model v1":
                st.info(
                    "Range source: Opponent Model v1. Model uncertainty is not included "
                    "in the Monte Carlo equity interval."
                )
                for opponent in context.opponent_ids:
                    snapshot = model_map[opponent].snapshot()
                    st.write(
                        f"{opponent}: {snapshot.hands_observed} observed hands; "
                        f"range entropy {snapshot.range_summary.entropy:.2f} nats."
                    )
            for opponent, parsed in parsed_preflop.items():
                stats = parsed.stats(dead)
                with st.expander(f"{opponent} range matrix", expanded=True):
                    st.write(
                        f"{stats.raw_combo_count} raw combos; "
                        f"{stats.legal_combo_count} legal; "
                        f"{stats.blocked_combo_count} blocked; "
                        f"{stats.raw_preflop_coverage:.1%} raw preflop coverage; "
                        f"legal total weight {stats.legal_total_weight:g}."
                    )
                    st.dataframe(
                        range_matrix_rows(parsed),
                        hide_index=True,
                        use_container_width=True,
                    )
            equity = result.equity
            interval = equity.confidence_interval_95
            metrics = st.columns(3)
            metrics[0].metric("Win", f"{equity.win_probability:.1%}")
            metrics[1].metric("Tie", f"{equity.tie_probability:.1%}")
            metrics[2].metric("Expected pot share", f"{equity.equity:.1%}")
            st.caption(
                f"{equity.method}; {equity.outcomes:,} outcomes; standard error "
                f"{equity.standard_error:.3%}; 95% sampling interval "
                f"{interval[0]:.1%}–{interval[1]:.1%}."
            )
            baseline_rows = [
                {
                    "action": value.action,
                    "EV": None if value.ev is None else round(value.ev, 3),
                    "EV SE": value.standard_error,
                    "EV 95% interval": value.confidence_interval_95,
                    "assumptions": value.assumptions,
                }
                for value in result.actions
            ]
            st.dataframe(baseline_rows, hide_index=True, use_container_width=True)
            if len(context.opponent_ids) == 1:
                aggressive = analyze_current_decision(
                    session.game,
                    actor,
                    opponent_range=ranges[context.opponent_ids[0]],
                    fold_equity=fold_equity,
                    samples=samples,
                    seed=0,
                )
                st.write("**Heads-up aggressive-action extension**")
                st.dataframe(
                    [
                        {
                            "size": value.sizing.label,
                            "target to": value.sizing.target_to,
                            "decision cost": value.sizing.decision_cost,
                            "EV": round(value.ev, 3),
                            "regret in model": round(value.regret, 3),
                            "assumed fold frequency": fold_equity,
                        }
                        for value in aggressive.candidate_values
                    ],
                    hide_index=True,
                    use_container_width=True,
                )
            for line in result.explanation:
                st.write(line)
            review_choices: dict[str, Action] = {}
            legal = context.legal_actions
            if legal.can_fold:
                review_choices["Fold"] = Fold()
            if legal.can_check:
                review_choices["Check"] = Check()
            if legal.can_call:
                review_choices["Call"] = Call()
            if legal.can_bet and legal.min_bet_to is not None:
                review_choices[f"Bet to {legal.min_bet_to}"] = BetTo(legal.min_bet_to)
            if legal.can_raise and legal.min_raise_to is not None:
                review_choices[f"Raise to {legal.min_raise_to}"] = RaiseTo(
                    legal.min_raise_to
                )
            selected_review = st.selectbox(
                "Action to record in DecisionReview", tuple(review_choices)
            )
            if st.button("Save analysis review"):
                review = capture_decision_review(
                    session.position,
                    result,
                    review_choices[selected_review],
                    tuple(range_inputs.values()),
                )
                st.session_state.setdefault("decision_reviews", []).append(review)
                st.success(
                    f"Saved review: best baseline action {review.best_baseline_action}; "
                    f"estimated baseline regret {review.estimated_baseline_regret}."
                )
    if len(context.opponent_ids) == 1 and not advanced:
        st.write("**Range sensitivity**")
        alternatives = st.text_area(
            "Named alternative ranges",
            placeholder="Tight = QQ+,AKs\nMy wider range = 22+,A2s+,KTs+,QJs,AJo+",
            help="Names and definitions are user supplied; no canonical labels are assumed.",
        )
        if st.button("Compare named ranges"):
            try:
                dead = context.hero_cards + context.board
                named = {}
                for line_number, line in enumerate(alternatives.splitlines(), 1):
                    if not line.strip():
                        continue
                    if "=" not in line:
                        raise ValueError(
                            f"sensitivity line {line_number} must use NAME = RANGE"
                        )
                    name, expression = line.split("=", 1)
                    if not name.strip():
                        raise ValueError(f"sensitivity line {line_number} needs a name")
                    named[name.strip()] = PreflopRange.parse(
                        expression
                    ).to_weighted_range(dead)
                if not named:
                    raise ValueError("enter at least one named range")
                compared = compare_ranges(
                    session.game, actor, named, samples=samples, seed=101
                )
                st.dataframe(
                    sensitivity_rows(compared),
                    hide_index=True,
                    use_container_width=True,
                )
            except ValueError as error:
                st.error(str(error))


def _render_scenario_builder(st: Any) -> None:
    st.write(
        "Build an exact spot with ordinary card notation and a legal action script."
    )
    count = st.number_input("Seats", 2, 6, 3, key="scenario_seats")
    default_names = " ".join(("BTN", "SB", "BB", "UTG", "HJ", "CO")[:count])
    names_text = st.text_input("Seat names, in order", default_names)
    stacks_text = st.text_input("Starting stacks", " ".join(["200"] * count))
    blind_cols = st.columns(3)
    small_blind = blind_cols[0].number_input("Small blind", 1, 1000, 1)
    big_blind = blind_cols[1].number_input("Big blind", 2, 2000, 2)
    button = blind_cols[2].number_input("Button seat", 0, count - 1, 0)
    hero = st.text_input("Hero player", "BTN")
    hero_cards = st.text_input("Hero cards", "As Qs")
    reveal_known = st.toggle("Add known opponent cards (research mode)")
    known_cards = st.text_area(
        "Known opponent cards",
        placeholder="SB Ah Kh\nBB 8s 8d",
        disabled=not reveal_known,
        help="One player ID followed by two cards per line.",
    )
    board = st.text_input("Board runout", "Qd 8c 4s")
    actions = st.text_area(
        "Action history",
        "BTN raise_to 6\nSB call\nBB call\nSB check\nBB bet_to 12",
        help="One line per action. Targets are total street contributions.",
    )
    if st.button("Build scenario"):
        try:
            names = tuple(names_text.split())
            stacks = tuple(int(value) for value in stacks_text.split())
            config = TableConfig(names, stacks, small_blind, big_blind, button)
            builder = ScenarioBuilder(config).set_hole_cards(hero, hero_cards)
            if reveal_known:
                for line_number, line in enumerate(known_cards.splitlines(), 1):
                    if not line.strip():
                        continue
                    parts = line.split()
                    if len(parts) != 3:
                        raise ValueError(
                            f"known-card line {line_number} must be 'PLAYER As Kh'"
                        )
                    builder.set_hole_cards(parts[0], parts[1:])
            if board.strip():
                builder.set_board_runout(board)
            for player_id, action in parse_action_script(actions):
                builder.action(player_id, action)
            st.session_state.training_session = TrainingSession.from_scenario(
                builder, human_players={hero}
            )
            st.rerun()
        except (ValueError, KeyError) as error:
            st.error(str(error))


def main() -> None:
    import streamlit as st  # type: ignore[import-not-found]

    st.set_page_config(page_title="Poker Research Trainer", layout="wide")
    st.title("Poker Research Trainer")
    st.caption("Offline study only. Baselines are simplified and are not GTO advice.")
    if "training_session" not in st.session_state:
        st.session_state.training_session = create_example_session()

    with st.sidebar:
        st.header("Session")
        if st.button("Load AsQs example"):
            st.session_state.training_session = create_example_session()
            st.rerun()
        with st.form("new_hand"):
            players = st.number_input("Players", 2, 6, 3)
            stack = st.number_input("Starting stack", 2, 100_000, 200)
            hero_seat = st.number_input("Human seat", 0, players - 1, 0)
            seed = st.number_input("Seed", 0, 2**31 - 1, 0)
            policy = st.selectbox(
                "Opponent policy", list(PolicyKind), format_func=lambda x: x.value
            )
            if st.form_submit_button("New hand"):
                st.session_state.training_session = create_new_hand_session(
                    players, stack, hero_seat, seed, policy
                )
                st.rerun()
        uploaded = st.file_uploader("Load session JSON", type="json")
        if uploaded is not None and st.button("Import JSON"):
            try:
                st.session_state.training_session = TrainingSession.from_json(
                    uploaded.getvalue().decode("utf-8")
                )
                st.rerun()
            except (ValueError, KeyError, TypeError) as error:
                st.error(str(error))
        session = st.session_state.training_session
        st.download_button(
            "Export session JSON",
            session.to_json(),
            "poker-training-session.json",
            "application/json",
        )

    trainer, builder_tab, simulation_tab, opponent_tab = st.tabs(
        ("Trainer", "Scenario builder", "Simulation Lab", "Opponent Model")
    )
    with trainer:
        session = st.session_state.training_session
        research_view = st.toggle(
            "Research view (reveals all private cards)", value=False
        )
        if research_view:
            st.warning(
                "Privileged research view is on. Opponent policies never receive this state."
            )
        _render_table(st, session, research_view)
        _render_controls(st, session)
        _render_timeline(st, session)
        _render_analysis(st, session)
    with builder_tab:
        _render_scenario_builder(st)
    with simulation_tab:
        st.subheader("Synthetic Population Simulation Lab")
        player_count = st.number_input("Simulation players", 2, 6, 2)
        selected_profiles = []
        keys = tuple(PRESETS)
        for seat in range(player_count):
            selected_profiles.append(
                st.selectbox(
                    f"Seat {seat + 1} profile",
                    keys,
                    index=seat % len(keys),
                    key=f"simulation_profile_{seat}",
                    format_func=lambda value: PRESETS[value].name,
                )
            )
        budget = st.selectbox("Hands", (1_000, 10_000, "Custom"))
        simulation_hands = (
            st.number_input("Custom hands", 1, 1_000_000, 1_000)
            if budget == "Custom"
            else budget
        )
        simulation_stack = st.number_input("Starting stack (BB)", 10, 500, 100)
        simulation_seed = st.number_input("Experiment seed", 0, 2**31 - 1, 42)
        duplicate = st.toggle("Duplicate-deal mode")
        if duplicate:
            rounded_hands = simulation_hands - (simulation_hands % player_count)
            st.info(
                "Duplicate balanced blocks reuse one deal and button while rotating "
                f"all {player_count} participants through physical seats. Requested "
                f"physical hands: {simulation_hands:,}; complete-block hands: {rounded_hands:,}; "
                f"independent blocks: {rounded_hands // player_count:,}."
            )
            simulation_hands = rounded_hands
        can_run = simulation_hands >= player_count if duplicate else True
        if duplicate and not can_run:
            st.error(
                f"Duplicate mode needs at least one complete {player_count}-hand block."
            )
        if st.button("Run simulation", disabled=not can_run):
            with st.spinner("Running independent offline hands..."):
                result = SimulationRunner(
                    SimulationConfig(
                        tuple(PRESETS[key] for key in selected_profiles),
                        hands=simulation_hands,
                        stack_bb=simulation_stack,
                        master_seed=simulation_seed,
                        duplicate_deals=duplicate,
                    )
                ).run()
            st.session_state.simulation_result = result
        if "simulation_result" in st.session_state:
            result = st.session_state.simulation_result
            st.caption(
                f"Method: {result.metadata['schedule_type']}; physical hands: "
                f"{result.metadata['physical_hands']}; independent duplicate blocks: "
                f"{result.metadata['independent_duplicate_blocks']}."
            )
            st.dataframe(
                [
                    {
                        "Profile": metric.profile,
                        "Hands": metric.hands,
                        "Net BB": metric.total_net_bb,
                        "bb/100": metric.bb_per_100,
                        "95% CI": metric.confidence_interval_95_bb_per_100,
                        "VPIP": metric.vpip,
                        "PFR": metric.pfr,
                        "3-bet": metric.three_bet_frequency,
                        "Call": metric.call_frequency,
                        "Fold": metric.fold_frequency,
                        "Check": metric.check_frequency,
                        "Aggression": metric.postflop_aggression_frequency,
                        "Showdown": metric.showdown_rate,
                    }
                    for metric in result.metrics
                ],
                hide_index=True,
                use_container_width=True,
            )
            st.warning(
                "Wide confidence intervals mean the experiment does not identify a winner."
            )
            st.line_chart(result.cumulative_net_bb(), x="hand")
            position_counts: dict[str, dict[str, int]] = {}
            for record in result.records:
                for seat in record.seats:
                    key = seat.participant_id
                    position_counts.setdefault(key, {})
                    position_counts[key][seat.position] = (
                        position_counts[key].get(seat.position, 0) + 1
                    )
            st.write("**Position-balance diagnostics**")
            st.dataframe(
                [
                    {"Participant": key, **counts}
                    for key, counts in position_counts.items()
                ],
                hide_index=True,
                use_container_width=True,
            )
            st.download_button(
                "Export experiment JSON",
                result.to_json(),
                "simulation.json",
                "application/json",
            )
            st.download_button(
                "Export metrics CSV",
                result.metrics_csv(),
                "simulation-metrics.csv",
                "text/csv",
            )
    with opponent_tab:
        _render_opponent_model_lab(st)


def _render_opponent_model_lab(st: Any) -> None:
    st.subheader("Opponent Model v1 — offline synthetic study")
    target_id = st.text_input("Opponent ID", "P2")
    profile_key = st.selectbox(
        "Synthetic generator used only for research validation", tuple(PRESETS)
    )
    hands = st.number_input("Observed hands", 2, 10_000, 100)
    seed = st.number_input("Model experiment seed", 0, 2**31 - 1, 17)
    if st.button("Generate public observations and fit model"):
        observer = Participant("P1", "Observer", PRESETS["tag"])
        villain = Participant(target_id, "Unlabeled opponent", PRESETS[profile_key])
        result = SimulationRunner(
            SimulationConfig(
                (observer.profile, villain.profile),
                hands=hands,
                master_seed=seed,
                participants=(observer, villain),
            )
        ).run()
        model = OpponentModel("P1", target_id)
        for record in result.records:
            for decision in record.observed_decisions:
                model.observe(decision)
        st.session_state.setdefault("opponent_models", {})[target_id] = model
    model = st.session_state.get("opponent_models", {}).get(target_id)
    if model is None:
        st.caption("No fitted model for this opponent ID yet.")
        return
    snapshot = model.snapshot()
    st.write("\n".join(model.explanation()))
    rows = []
    for name, estimate in snapshot.tendencies.items():
        rows.append(
            {
                "Tendency": name,
                "Mean": estimate.mean,
                "95% credible interval": estimate.credible_interval_95,
                "Opportunities": estimate.opportunities,
            }
        )
    st.dataframe(rows, hide_index=True, use_container_width=True)
    st.write("**Synthetic-archetype posterior (uniform prior)**")
    st.dataframe(
        [
            {"Archetype": name, "Probability": probability}
            for name, probability in sorted(
                snapshot.archetype_posterior.items(),
                key=lambda item: item[1],
                reverse=True,
            )
        ],
        hide_index=True,
        use_container_width=True,
    )
    summary = snapshot.range_summary
    st.caption(
        f"{summary.legal_combo_count} legal combos; entropy {summary.entropy:.2f}; "
        f"effective combos exp(H)={summary.effective_combo_count:.1f}."
    )
    st.dataframe(
        [
            {
                "Rank": rank,
                **{
                    column: row[index]
                    for index, column in enumerate("AKQJT98765432")
                },
            }
            for rank, row in zip("AKQJT98765432", summary.matrix)
        ],
        hide_index=True,
        use_container_width=True,
    )


if __name__ == "__main__":
    main()
