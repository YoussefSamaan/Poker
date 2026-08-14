from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json

from ..agents import StrategyProfile


@dataclass(frozen=True, slots=True)
class Participant:
    participant_id: str
    label: str
    profile: StrategyProfile

    @property
    def profile_fingerprint(self) -> str:
        payload = json.dumps(
            self.profile.to_dict(), sort_keys=True, separators=(",", ":")
        )
        return hashlib.sha256(payload.encode()).hexdigest()[:16]


@dataclass(frozen=True, slots=True)
class ScheduledHand:
    hand_index: int
    button: int
    participant_indices_by_seat: tuple[int, ...]
    duplicate_block_id: int | None
    duplicate_leg: int | None


def participants_from_profiles(
    profiles: tuple[StrategyProfile, ...],
) -> tuple[Participant, ...]:
    return tuple(
        Participant(f"participant_{index}", f"Player {index + 1}", profile)
        for index, profile in enumerate(profiles)
    )


def build_schedule(
    hands: int,
    player_count: int,
    duplicate: bool,
    rotate_assignments: bool = True,
) -> tuple[ScheduledHand, ...]:
    if hands < 1:
        raise ValueError("hands must be positive")
    if not 2 <= player_count <= 6:
        raise ValueError("player count must be between 2 and 6")
    if duplicate and hands % player_count:
        raise ValueError(
            f"duplicate mode requires hands divisible by player count ({player_count})"
        )
    scheduled = []
    if duplicate:
        for hand in range(hands):
            block, leg = divmod(hand, player_count)
            button = block % player_count
            assignment = tuple(
                (seat - leg) % player_count for seat in range(player_count)
            )
            scheduled.append(ScheduledHand(hand, button, assignment, block, leg))
    else:
        for hand in range(hands):
            within_block = hand % player_count
            block = hand // player_count
            button = within_block
            rotation = block % player_count if rotate_assignments else 0
            assignment = tuple(
                (seat + rotation) % player_count for seat in range(player_count)
            )
            scheduled.append(ScheduledHand(hand, button, assignment, None, None))
    return tuple(scheduled)
