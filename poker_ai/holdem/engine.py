from __future__ import annotations

import random
from typing import Iterable

from ..cards import Card, full_deck, parse_cards
from ..evaluation import evaluate_holdem
from .actions import (
    Action,
    BetTo,
    Call,
    Check,
    Fold,
    IllegalAction,
    LegalActions,
    RaiseTo,
)
from .pots import build_side_pots
from .state import (
    ActionRecord,
    ActionType,
    HandResult,
    InternalPlayerState,
    InternalState,
    PlayerObservation,
    PlayerState,
    PlayerStatus,
    PotResult,
    PublicPlayerState,
    Street,
    TableConfig,
    Transition,
)


class HoldemGame:
    """Authoritative, one-action-at-a-time No-Limit Texas Hold'em engine.

    Chips are integer units and the deck does not model burn cards. A preset deck
    is interpreted in deal order, with index zero dealt first.
    """

    def __init__(
        self,
        config: TableConfig,
        *,
        seed: int | None = None,
        preset_deck: Iterable[Card | str] | None = None,
    ) -> None:
        self.config = config
        self._rng = random.Random(seed)
        self.seed = seed
        self._preset_deck = (
            parse_cards(preset_deck) if preset_deck is not None else None
        )
        if self._preset_deck is not None:
            if len(self._preset_deck) != 52 or set(self._preset_deck) != set(
                full_deck()
            ):
                raise ValueError(
                    "preset_deck must contain every physical card exactly once"
                )

        self._players = [
            PlayerState(
                player_id,
                seat,
                stack,
                PlayerStatus.OUT if stack == 0 else PlayerStatus.ACTIVE,
            )
            for seat, (player_id, stack) in enumerate(
                zip(config.player_ids, config.starting_stacks)
            )
        ]
        self.button_index = config.button
        self.small_blind_index = -1
        self.big_blind_index = -1
        self.street = Street.PREFLOP
        self.board: list[Card] = []
        self.current_bet = 0
        self.last_full_raise_size = config.big_blind
        self.current_player: str | None = None
        self.history: list[ActionRecord] = []
        self.result: HandResult | None = None
        self._deck: list[Card] = []
        self._deck_position = 0
        self._pending: set[str] = set()
        self._raise_rights: set[str] = set()
        self._hand_number = 0
        self._hand_started = False
        self._hand_start_total = sum(config.starting_stacks)

    @property
    def is_terminal(self) -> bool:
        return self.result is not None

    @property
    def pot(self) -> int:
        return sum(player.total_contribution for player in self._players)

    def start_hand(self) -> None:
        if self._hand_started and not self.is_terminal:
            raise RuntimeError(
                "cannot start a new hand before the current hand terminates"
            )
        live_seats = [player.seat for player in self._players if player.stack > 0]
        if len(live_seats) < 2:
            raise RuntimeError("at least two players with chips are required")

        if self._hand_number:
            self.button_index = self._next_seat(
                self.button_index, lambda player: player.stack > 0
            )
        elif self._players[self.button_index].stack == 0:
            self.button_index = self._next_seat(
                self.button_index, lambda player: player.stack > 0
            )
        self._hand_number += 1
        self._hand_started = True
        self._hand_start_total = sum(player.stack for player in self._players)
        self.street = Street.PREFLOP
        self.board = []
        self.current_bet = 0
        self.last_full_raise_size = self.config.big_blind
        self.current_player = None
        self.history = []
        self.result = None
        self._pending = set()
        self._raise_rights = set()

        for player in self._players:
            player.status = (
                PlayerStatus.ACTIVE if player.stack > 0 else PlayerStatus.OUT
            )
            player.hole_cards = ()
            player.street_contribution = 0
            player.total_contribution = 0

        if self._preset_deck is not None:
            self._deck = list(self._preset_deck)
        else:
            self._deck = list(full_deck())
            self._rng.shuffle(self._deck)
        self._deck_position = 0

        active_seats = [
            player.seat
            for player in self._players
            if player.status == PlayerStatus.ACTIVE
        ]
        if len(active_seats) == 2:
            self.small_blind_index = self.button_index
            self.big_blind_index = self._next_seat(self.button_index, self._is_in_hand)
        else:
            self.small_blind_index = self._next_seat(
                self.button_index, self._is_in_hand
            )
            self.big_blind_index = self._next_seat(
                self.small_blind_index, self._is_in_hand
            )

        first_dealt = self._next_seat(self.button_index, self._is_in_hand)
        deal_order = self._seat_order_from(first_dealt, self._is_in_hand)
        for _ in range(2):
            for seat in deal_order:
                player = self._players[seat]
                player.hole_cards += (self._deal_one(),)

        self._post_blind(
            self.small_blind_index, self.config.small_blind, ActionType.SMALL_BLIND
        )
        self._post_blind(
            self.big_blind_index, self.config.big_blind, ActionType.BIG_BLIND
        )
        posted_maximum = max(player.street_contribution for player in self._players)
        # With two or more players still able to wager, a short all-in big blind
        # does not reduce the table's nominal bring-in. Heads-up excess would be
        # uncalled, so only the amount actually posted is live there.
        self.current_bet = (
            max(self.config.big_blind, posted_maximum)
            if self._active_count() >= 2
            else posted_maximum
        )
        self._pending = {
            player.player_id
            for player in self._players
            if player.status == PlayerStatus.ACTIVE
        }
        self._raise_rights = set(self._pending)

        if len(active_seats) == 2:
            first_to_act = self.button_index
        else:
            first_to_act = self._next_seat(
                self.big_blind_index,
                lambda player: player.status == PlayerStatus.ACTIVE,
            )
        self.current_player = self._first_pending_from(first_to_act)
        self._auto_progress_if_no_betting_possible()
        self.assert_invariants()

    def legal_actions(self, player_id: str) -> LegalActions:
        self._require_running_hand()
        if player_id != self.current_player:
            raise IllegalAction(f"only current player {self.current_player!r} may act")
        player = self._player(player_id)
        if player.status == PlayerStatus.FOLDED:
            raise IllegalAction("a folded player cannot act")
        if player.status == PlayerStatus.ALL_IN:
            raise IllegalAction("an all-in player cannot act")
        if player.status != PlayerStatus.ACTIVE:
            raise IllegalAction("player is not active in this hand")

        to_call = max(0, self.current_bet - player.street_contribution)
        own_max = player.street_contribution + player.stack
        opponents = [
            other
            for other in self._players
            if other.player_id != player_id
            and other.status not in (PlayerStatus.FOLDED, PlayerStatus.OUT)
        ]
        contestable_to = max(
            (other.street_contribution + other.stack for other in opponents),
            default=player.street_contribution,
        )
        max_to = min(own_max, contestable_to)

        call_amount = min(to_call, player.stack) if to_call > 0 else None
        min_bet_to = max_bet_to = None
        min_raise_to = max_raise_to = None
        if player_id in self._raise_rights and player.stack > 0:
            if self.current_bet == 0 and max_to > 0:
                max_bet_to = max_to
                min_bet_to = min(self.config.big_blind, max_to)
            elif self.current_bet > 0 and max_to > self.current_bet:
                normal_minimum = (
                    self.config.big_blind
                    if self.current_bet < self.config.big_blind
                    else self.current_bet + self.last_full_raise_size
                )
                max_raise_to = max_to
                min_raise_to = min(normal_minimum, max_to)

        return LegalActions(
            player_id=player_id,
            can_fold=True,
            can_check=to_call == 0,
            call_amount=call_amount,
            min_bet_to=min_bet_to,
            max_bet_to=max_bet_to,
            min_raise_to=min_raise_to,
            max_raise_to=max_raise_to,
        )

    def step(self, action: Action, player_id: str | None = None) -> Transition:
        self._require_running_hand()
        actor_id = player_id if player_id is not None else self.current_player
        if actor_id is None or actor_id != self.current_player:
            raise IllegalAction(f"only current player {self.current_player!r} may act")
        player = self._player(actor_id)
        legal = self.legal_actions(actor_id)
        street_before = self.street
        board_before = len(self.board)
        pot_before = self.pot
        stack_before = player.stack
        contribution_before = player.street_contribution
        to_call_before = max(0, self.current_bet - contribution_before)
        amount_paid = 0
        target_to: int | None = None
        aggressive = False
        full_raise = False
        old_bet = self.current_bet

        if isinstance(action, Fold):
            player.status = PlayerStatus.FOLDED
            action_type = ActionType.FOLD
        elif isinstance(action, Check):
            if not legal.can_check:
                raise IllegalAction("cannot check while facing a bet")
            action_type = ActionType.CHECK
        elif isinstance(action, Call):
            if legal.call_amount is None:
                raise IllegalAction("cannot call when no chips are owed")
            amount_paid = self._commit(player, legal.call_amount)
            target_to = player.street_contribution
            action_type = ActionType.CALL
        elif isinstance(action, BetTo):
            self._validate_target(action.amount)
            if legal.min_bet_to is None or legal.max_bet_to is None:
                raise IllegalAction(
                    "betting is not legal; use RaiseTo when facing an existing wager"
                )
            if not legal.min_bet_to <= action.amount <= legal.max_bet_to:
                raise IllegalAction(
                    f"bet-to must be between {legal.min_bet_to} and {legal.max_bet_to}"
                )
            target_to = action.amount
            amount_paid = self._commit(player, target_to - contribution_before)
            self.current_bet = target_to
            aggressive = True
            full_raise = target_to >= self.config.big_blind
            action_type = ActionType.BET
        elif isinstance(action, RaiseTo):
            self._validate_target(action.amount)
            if legal.min_raise_to is None or legal.max_raise_to is None:
                raise IllegalAction(
                    "raising is not legal or betting has not been reopened"
                )
            if not legal.min_raise_to <= action.amount <= legal.max_raise_to:
                raise IllegalAction(
                    f"raise-to must be between {legal.min_raise_to} and {legal.max_raise_to}"
                )
            target_to = action.amount
            amount_paid = self._commit(player, target_to - contribution_before)
            self.current_bet = target_to
            aggressive = True
            if old_bet < self.config.big_blind:
                full_raise = target_to >= self.config.big_blind
            else:
                full_raise = target_to - old_bet >= self.last_full_raise_size
            action_type = ActionType.RAISE
        else:
            raise IllegalAction(f"unsupported action type: {type(action).__name__}")

        if aggressive:
            if target_to is None:
                raise AssertionError("aggressive action must have a target")
            if full_raise:
                if old_bet < self.config.big_blind:
                    self.last_full_raise_size = (
                        self.config.big_blind
                        if target_to == self.config.big_blind
                        else target_to - self.config.big_blind
                    )
                else:
                    self.last_full_raise_size = target_to - old_bet  # type: ignore[operator]
                others = {
                    other.player_id
                    for other in self._players
                    if other.player_id != actor_id
                    and other.status == PlayerStatus.ACTIVE
                }
                self._pending = set(others)
                self._raise_rights = set(others)
            else:
                self._pending.discard(actor_id)
                self._raise_rights.discard(actor_id)
                self._pending.update(
                    other.player_id
                    for other in self._players
                    if other.player_id != actor_id
                    and other.status == PlayerStatus.ACTIVE
                    and other.street_contribution < self.current_bet
                )
                # Multiple short all-ins can cumulatively reach a full raise for
                # a prior actor. At that point real no-limit rules reopen raising.
                self._raise_rights.update(
                    other.player_id
                    for other in self._players
                    if other.player_id != actor_id
                    and other.status == PlayerStatus.ACTIVE
                    and self.current_bet - other.street_contribution
                    >= self.last_full_raise_size
                )
        else:
            self._pending.discard(actor_id)
            if not isinstance(action, Check):
                self._raise_rights.discard(actor_id)

        if player.status != PlayerStatus.ACTIVE:
            self._pending.discard(actor_id)
            self._raise_rights.discard(actor_id)

        record = ActionRecord(
            sequence=len(self.history),
            street=street_before,
            player_id=actor_id,
            action_type=action_type,
            amount_paid=amount_paid,
            target_to=target_to,
            contribution_before=contribution_before,
            contribution_after=player.street_contribution,
            amount_to_call_before=to_call_before,
            pot_before=pot_before,
            pot_after=self.pot,
            stack_before=stack_before,
            stack_after=player.stack,
            caused_all_in=player.status == PlayerStatus.ALL_IN,
        )
        self.history.append(record)

        self._clean_action_sets()
        non_folded = self._non_folded_players()
        if len(non_folded) == 1:
            self._award_uncontested(non_folded[0])
        elif self._should_auto_runout():
            self._runout_and_showdown()
        elif not self._pending:
            self._close_betting_round()
        else:
            self.current_player = self._next_pending_after(player.seat)

        self.assert_invariants()
        return Transition(
            action_record=record,
            street_changed=self.street != street_before,
            cards_revealed=tuple(self.board[board_before:]),
            hand_terminated=self.is_terminal,
            next_player=self.current_player,
        )

    def observation_for(self, player_id: str) -> PlayerObservation:
        player = self._player(player_id)
        legal = None
        if not self.is_terminal and player_id == self.current_player:
            legal = self.legal_actions(player_id)
        return PlayerObservation(
            player_id=player_id,
            hole_cards=player.hole_cards,
            board=tuple(self.board),
            street=self.street,
            button_player=self._players[self.button_index].player_id,
            small_blind_player=self._players[self.small_blind_index].player_id,
            big_blind_player=self._players[self.big_blind_index].player_id,
            current_player=self.current_player,
            pot=self.pot,
            current_bet=self.current_bet,
            players=tuple(self._public_player(other) for other in self._players),
            history=tuple(self.history),
            legal_actions=legal,
            is_terminal=self.is_terminal,
        )

    @property
    def internal_state(self) -> InternalState:
        self._require_started_hand()
        return InternalState(
            players=tuple(
                InternalPlayerState(
                    player.player_id,
                    player.seat,
                    player.stack,
                    player.status,
                    player.street_contribution,
                    player.total_contribution,
                    player.hole_cards,
                )
                for player in self._players
            ),
            board=tuple(self.board),
            remaining_deck=tuple(self._deck[self._deck_position :]),
            street=self.street,
            button_player=self._players[self.button_index].player_id,
            small_blind_player=self._players[self.small_blind_index].player_id,
            big_blind_player=self._players[self.big_blind_index].player_id,
            current_player=self.current_player,
            current_bet=self.current_bet,
            last_full_raise_size=self.last_full_raise_size,
            pending_players=tuple(sorted(self._pending)),
            raise_rights=tuple(sorted(self._raise_rights)),
            history=tuple(self.history),
            is_terminal=self.is_terminal,
        )

    def assert_invariants(self) -> None:
        self._require_started_hand()
        committed = sum(player.total_contribution for player in self._players)
        stacks = sum(player.stack for player in self._players)
        if stacks + committed != self._hand_start_total:
            raise AssertionError(
                f"chip conservation failed: stacks={stacks}, committed={committed}, "
                f"expected={self._hand_start_total}"
            )

        located = [card for player in self._players for card in player.hole_cards]
        located.extend(self.board)
        located.extend(self._deck[self._deck_position :])
        if (
            len(located) != 52
            or len(set(located)) != 52
            or set(located) != set(full_deck())
        ):
            raise AssertionError("card conservation failed")
        for player in self._players:
            if player.status == PlayerStatus.ACTIVE and player.stack <= 0:
                raise AssertionError("active player must have chips")
            if (
                not self.is_terminal
                and player.status == PlayerStatus.ALL_IN
                and player.stack != 0
            ):
                raise AssertionError("all-in player must have zero chips")
            if (
                player.stack < 0
                or player.total_contribution < 0
                or player.street_contribution < 0
            ):
                raise AssertionError("chip amounts cannot be negative")

    def _post_blind(self, seat: int, requested: int, action_type: ActionType) -> None:
        player = self._players[seat]
        pot_before = self.pot
        stack_before = player.stack
        before = player.street_contribution
        paid = self._commit(player, min(requested, player.stack))
        self.history.append(
            ActionRecord(
                sequence=len(self.history),
                street=Street.PREFLOP,
                player_id=player.player_id,
                action_type=action_type,
                amount_paid=paid,
                target_to=player.street_contribution,
                contribution_before=before,
                contribution_after=player.street_contribution,
                amount_to_call_before=0,
                pot_before=pot_before,
                pot_after=self.pot,
                stack_before=stack_before,
                stack_after=player.stack,
                caused_all_in=player.status == PlayerStatus.ALL_IN,
            )
        )

    def _commit(self, player: PlayerState, amount: int) -> int:
        if amount < 0 or amount > player.stack:
            raise IllegalAction("chip commitment is outside the player's stack")
        player.stack -= amount
        player.street_contribution += amount
        player.total_contribution += amount
        if player.stack == 0:
            player.status = PlayerStatus.ALL_IN
        return amount

    def _close_betting_round(self) -> None:
        if self.street == Street.RIVER:
            self._showdown()
            return
        if self.street == Street.PREFLOP:
            self.street = Street.FLOP
            self.board.extend(self._deal(3))
        elif self.street == Street.FLOP:
            self.street = Street.TURN
            self.board.extend(self._deal(1))
        elif self.street == Street.TURN:
            self.street = Street.RIVER
            self.board.extend(self._deal(1))

        for player in self._players:
            player.street_contribution = 0
        self.current_bet = 0
        self.last_full_raise_size = self.config.big_blind
        self._pending = {
            player.player_id
            for player in self._players
            if player.status == PlayerStatus.ACTIVE
        }
        self._raise_rights = set(self._pending)
        first_seat = self._next_seat(
            self.button_index, lambda player: player.status == PlayerStatus.ACTIVE
        )
        self.current_player = self._first_pending_from(first_seat)
        self._auto_progress_if_no_betting_possible()

    def _auto_progress_if_no_betting_possible(self) -> None:
        non_folded = self._non_folded_players()
        if len(non_folded) == 1:
            self._award_uncontested(non_folded[0])
        elif self._should_auto_runout():
            self._runout_and_showdown()

    def _runout_and_showdown(self) -> None:
        if len(self.board) == 0:
            self.street = Street.FLOP
            self.board.extend(self._deal(3))
        if len(self.board) == 3:
            self.street = Street.TURN
            self.board.extend(self._deal(1))
        if len(self.board) == 4:
            self.street = Street.RIVER
            self.board.extend(self._deal(1))
        self._showdown()

    def _showdown(self) -> None:
        self.street = Street.SHOWDOWN
        eligible = {player.player_id for player in self._non_folded_players()}
        contributions = {
            player.player_id: player.total_contribution for player in self._players
        }
        layers = build_side_pots(contributions, eligible)
        payouts = {player.player_id: 0 for player in self._players}
        pot_results: list[PotResult] = []
        for layer in layers:
            if not layer.eligible_players:
                raise AssertionError("derived a pot with no eligible player")
            ranks = {
                player_id: evaluate_holdem(
                    self._player(player_id).hole_cards + tuple(self.board)
                )
                for player_id in layer.eligible_players
            }
            best = max(ranks.values())
            tied = {player_id for player_id, rank in ranks.items() if rank == best}
            ordered_winners = tuple(
                self._players[seat].player_id
                for seat in self._clockwise_seats_left_of_button()
                if self._players[seat].player_id in tied
            )
            base, odd = divmod(layer.amount, len(ordered_winners))
            layer_payouts = {winner: base for winner in ordered_winners}
            for winner in ordered_winners[:odd]:
                layer_payouts[winner] += 1
            for winner, amount in layer_payouts.items():
                payouts[winner] += amount
            pot_results.append(
                PotResult(
                    layer.amount, layer.eligible_players, ordered_winners, layer_payouts
                )
            )
        self._finish_hand("showdown", True, payouts, tuple(pot_results))

    def _award_uncontested(self, winner: PlayerState) -> None:
        amount = self.pot
        payouts = {player.player_id: 0 for player in self._players}
        payouts[winner.player_id] = amount
        pot_result = PotResult(
            amount, (winner.player_id,), (winner.player_id,), {winner.player_id: amount}
        )
        self._finish_hand("all_others_folded", False, payouts, (pot_result,))

    def _finish_hand(
        self,
        reason: str,
        showdown: bool,
        payouts: dict[str, int],
        pots: tuple[PotResult, ...],
    ) -> None:
        for player in self._players:
            player.stack += payouts[player.player_id]
            player.street_contribution = 0
            player.total_contribution = 0
        winners = tuple(
            player.player_id
            for player in self._players
            if payouts[player.player_id] > 0
        )
        self.current_player = None
        self.current_bet = 0
        self._pending.clear()
        self._raise_rights.clear()
        self.result = HandResult(
            reason,
            showdown,
            winners,
            dict(payouts),
            pots,
            {player.player_id: player.stack for player in self._players},
        )

    def _clean_action_sets(self) -> None:
        active = {
            player.player_id
            for player in self._players
            if player.status == PlayerStatus.ACTIVE
        }
        self._pending.intersection_update(active)
        self._raise_rights.intersection_update(active)

    def _deal_one(self) -> Card:
        if self._deck_position >= len(self._deck):
            raise RuntimeError("deck exhausted")
        card = self._deck[self._deck_position]
        self._deck_position += 1
        return card

    def _deal(self, count: int) -> tuple[Card, ...]:
        return tuple(self._deal_one() for _ in range(count))

    def _player(self, player_id: str) -> PlayerState:
        for player in self._players:
            if player.player_id == player_id:
                return player
        raise KeyError(f"unknown player {player_id!r}")

    def _public_player(self, player: PlayerState) -> PublicPlayerState:
        return PublicPlayerState(
            player.player_id,
            player.seat,
            player.stack,
            player.status,
            player.street_contribution,
            player.total_contribution,
        )

    def _is_in_hand(self, player: PlayerState) -> bool:
        return player.status != PlayerStatus.OUT

    def _next_seat(self, seat: int, predicate) -> int:
        for offset in range(1, len(self._players) + 1):
            candidate = (seat + offset) % len(self._players)
            if predicate(self._players[candidate]):
                return candidate
        raise RuntimeError("no eligible seat")

    def _seat_order_from(self, first_seat: int, predicate) -> tuple[int, ...]:
        return tuple(
            seat
            for offset in range(len(self._players))
            if predicate(
                self._players[seat := (first_seat + offset) % len(self._players)]
            )
        )

    def _first_pending_from(self, first_seat: int) -> str | None:
        for offset in range(len(self._players)):
            player = self._players[(first_seat + offset) % len(self._players)]
            if player.player_id in self._pending:
                return player.player_id
        return None

    def _next_pending_after(self, seat: int) -> str:
        player_id = self._first_pending_from((seat + 1) % len(self._players))
        if player_id is None:
            raise AssertionError("pending action set has no reachable player")
        return player_id

    def _clockwise_seats_left_of_button(self) -> tuple[int, ...]:
        return tuple(
            (self.button_index + offset) % len(self._players)
            for offset in range(1, len(self._players) + 1)
        )

    def _non_folded_players(self) -> list[PlayerState]:
        return [
            player
            for player in self._players
            if player.status not in (PlayerStatus.FOLDED, PlayerStatus.OUT)
        ]

    def _active_count(self) -> int:
        return sum(player.status == PlayerStatus.ACTIVE for player in self._players)

    def _should_auto_runout(self) -> bool:
        active = [
            player for player in self._players if player.status == PlayerStatus.ACTIVE
        ]
        if not active:
            return True
        if len(active) > 1:
            return False
        sole_player = active[0]
        return (
            sole_player.player_id not in self._pending
            or sole_player.street_contribution >= self.current_bet
        )

    @staticmethod
    def _validate_target(amount: int) -> None:
        if isinstance(amount, bool) or not isinstance(amount, int):
            raise IllegalAction("bet and raise targets must be integer chip amounts")
        if amount <= 0:
            raise IllegalAction("bet and raise targets must be positive")

    def _require_started_hand(self) -> None:
        if not self._hand_started:
            raise RuntimeError("start_hand() must be called first")

    def _require_running_hand(self) -> None:
        self._require_started_hand()
        if self.is_terminal:
            raise IllegalAction("the hand is already terminal")
