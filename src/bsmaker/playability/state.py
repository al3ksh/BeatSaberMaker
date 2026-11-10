"""Per-hand simulation state."""
from __future__ import annotations

from dataclasses import dataclass

from bsmaker.formats.tokens import NONE_TOKEN


@dataclass(frozen=True)
class HandState:
    time_s: float | None
    x: int
    y: int
    angle: float | None


@dataclass(frozen=True)
class State:
    left: HandState
    right: HandState
    prev_time_s: float | None
    prev_left: int
    prev_right: int


def initial_state() -> State:
    h = HandState(None, 0, 0, None)
    return State(h, h, None, NONE_TOKEN, NONE_TOKEN)
