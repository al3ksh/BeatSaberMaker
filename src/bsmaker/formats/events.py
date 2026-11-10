"""Turn a note list into a time-ordered sequence of (left, right) token events."""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from bsmaker.formats.model import Note
from bsmaker.formats.tokens import NONE_TOKEN, encode_token

# Mel frame rate: 22050 Hz sample rate / hop_length 256. Also defined in the audio module.
FRAMES_PER_SECOND = 22050 / 256


@dataclass(frozen=True)
class Event:
    time_s: float
    left: int
    right: int


def to_events(notes: Sequence[Note], merge_gap_s: float = 0.040) -> tuple[list[Event], int]:
    """Group notes into events; return (events, number of notes dropped by merging)."""
    groups: dict[int, list[Note]] = {}
    for n in sorted(notes, key=lambda n: (n.time_s, n.color, n.x, n.y)):
        groups.setdefault(round(n.time_s * FRAMES_PER_SECOND), []).append(n)

    dropped = 0
    events: list[Event] = []
    last_time: dict[int, float] = {}  # color -> time of last kept note
    for frame in sorted(groups):
        tokens = {0: NONE_TOKEN, 1: NONE_TOKEN}
        for n in groups[frame]:
            c = n.color
            too_close = c in last_time and n.time_s - last_time[c] < merge_gap_s
            if c not in tokens or tokens[c] != NONE_TOKEN or too_close:
                dropped += 1
                continue
            tokens[c] = encode_token(n.x, n.y, n.direction)
            last_time[c] = n.time_s
        if tokens[0] != NONE_TOKEN or tokens[1] != NONE_TOKEN:
            events.append(Event(frame / FRAMES_PER_SECOND, tokens[0], tokens[1]))
    return events, dropped
