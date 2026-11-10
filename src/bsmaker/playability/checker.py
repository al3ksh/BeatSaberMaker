"""Hard/soft playability rules, shared by the generation mask and the report."""
from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

from bsmaker.formats.events import Event
from bsmaker.formats.tokens import NONE_TOKEN, VOCAB_SIZE, decode_token
from bsmaker.playability.config import PlayabilityConfig
from bsmaker.playability.geometry import angle_diff, direction_angle
from bsmaker.playability.state import HandState, State, initial_state

_DECODED = [(0, 0, 8)] + [decode_token(t) for t in range(1, VOCAB_SIZE)]
_ANGLE = [direction_angle(d) for (_, _, d) in _DECODED]

Rule = Callable[[State, float, int, int, int, PlayabilityConfig], "str | None"]


def _hand(state: State, hand: int) -> HandState:
    return state.left if hand == 0 else state.right


def _parity_delta(h: HandState, token: int) -> float | None:
    """Angle difference to the expected swing, or None if parity does not apply."""
    new = _ANGLE[token]
    if h.angle is None or new is None:
        return None
    return angle_diff(new, (h.angle + 180) % 360)


def _parity(state, time_s, hand, token, other, cfg):
    h = _hand(state, hand)
    d = _parity_delta(h, token)
    if d is None or h.time_s is None:
        return None
    gap = time_s - h.time_s
    if gap >= cfg.reset_gap_s:
        return None
    limit = cfg.lenient_angle_deg if gap >= cfg.slow_gap_s else cfg.strict_angle_deg
    return "parity" if d > limit else None


def _hand_speed(state, time_s, hand, token, other, cfg):
    h = _hand(state, hand)
    if h.time_s is None:
        return None
    x, y, _ = _DECODED[token]
    need = cfg.min_hand_gap_s + cfg.min_hand_gap_per_cell_s * max(abs(x - h.x), abs(y - h.y))
    return "hand_speed" if time_s - h.time_s < need else None


def _rare_token(state, time_s, hand, token, other, cfg):
    return "rare_token" if token in cfg.rare_tokens[hand] else None


def _pair(hand: int, token: int, other: int) -> tuple[int, int]:
    """Return (red, blue) tokens."""
    return (token, other) if hand == 0 else (other, token)


def _same_cell(state, time_s, hand, token, other, cfg):
    if other == NONE_TOKEN:
        return None
    r, b = _pair(hand, token, other)
    return "same_cell" if _DECODED[r][:2] == _DECODED[b][:2] else None


def _crossover(state, time_s, hand, token, other, cfg):
    if other == NONE_TOKEN or not cfg.forbid_crossover:
        return None
    r, b = _pair(hand, token, other)
    return "crossover" if _DECODED[r][0] > _DECODED[b][0] else None


def _handclap(state, time_s, hand, token, other, cfg):
    if other == NONE_TOKEN:
        return None
    r, b = _pair(hand, token, other)
    (xr, yr, _), (xb, yb, _) = _DECODED[r], _DECODED[b]
    if yr != yb or xb - xr != 1:
        return None
    return "handclap" if _ANGLE[r] in (0, 45, 315) and _ANGLE[b] in (135, 180, 225) else None


def _vision_block(state, time_s, hand, token, other, cfg):
    if state.prev_time_s is None or time_s - state.prev_time_s >= cfg.vision_gap_s:
        return None
    prev = state.prev_right if hand == 0 else state.prev_left
    if prev == NONE_TOKEN:
        return None
    px, py, _ = _DECODED[prev]
    return "vision_block" if py == 1 and px in (1, 2) and _DECODED[token][0] == px else None


HARD_RULES: tuple[Rule, ...] = (
    _parity, _hand_speed, _same_cell, _handclap, _crossover, _vision_block, _rare_token,
)


def _reset(state, time_s, hand, token, other, cfg):
    h = _hand(state, hand)
    d = _parity_delta(h, token)
    if d is None or h.time_s is None:
        return None
    return "reset" if time_s - h.time_s >= cfg.reset_gap_s and d > cfg.lenient_angle_deg else None


def _big_jump(state, time_s, hand, token, other, cfg):
    h = _hand(state, hand)
    if h.time_s is None:
        return None
    x, y, _ = _DECODED[token]
    return "big_jump" if max(abs(x - h.x), abs(y - h.y)) >= 3 and time_s - h.time_s < 0.20 else None


SOFT_RULES: tuple[Rule, ...] = (_reset, _big_jump)


def allowed(
    state: State, time_s: float, hand: int, cfg: PlayabilityConfig, other_token: int = NONE_TOKEN
) -> np.ndarray:
    """Boolean mask over all tokens that break no hard rule for `hand` at `time_s`."""
    mask = np.zeros(VOCAB_SIZE, dtype=bool)
    mask[NONE_TOKEN] = True
    for token in range(1, VOCAB_SIZE):
        mask[token] = all(
            rule(state, time_s, hand, token, other_token, cfg) is None for rule in HARD_RULES
        )
    return mask


def advance(state: State, time_s: float, left: int, right: int) -> State:
    """State after the event; a hand whose token is NONE_TOKEN is left untouched."""

    def step(h: HandState, token: int) -> HandState:
        if token == NONE_TOKEN:
            return h
        x, y, _ = _DECODED[token]
        angle = _ANGLE[token]
        if angle is None and h.angle is not None:
            angle = (h.angle + 180) % 360
        return HandState(time_s, x, y, angle)

    return State(step(state.left, left), step(state.right, right), time_s, left, right)


@dataclass(frozen=True)
class Violation:
    time_s: float
    hand: int
    rule: str


@dataclass(frozen=True)
class Report:
    n_notes: int
    hard: tuple[Violation, ...]
    soft: tuple[Violation, ...]

    @property
    def hard_rate(self) -> float:
        return len(self.hard) / max(self.n_notes, 1)


def check(events: Sequence[Event], cfg: PlayabilityConfig) -> Report:
    """Replay events through the same rules `allowed` uses."""
    state = initial_state()
    hard: list[Violation] = []
    soft: list[Violation] = []
    n_notes = 0
    for e in events:
        for hand, token in ((0, e.left), (1, e.right)):
            if token == NONE_TOKEN:
                continue
            n_notes += 1
            # Doubles are attributed to the right hand: only hand 1 sees the other token.
            other = e.left if hand == 1 else NONE_TOKEN
            for rule in HARD_RULES:
                name = rule(state, e.time_s, hand, token, other, cfg)
                if name:
                    hard.append(Violation(e.time_s, hand, name))
            for rule in SOFT_RULES:
                name = rule(state, e.time_s, hand, token, NONE_TOKEN, cfg)
                if name:
                    soft.append(Violation(e.time_s, hand, name))
        state = advance(state, e.time_s, e.left, e.right)
    return Report(n_notes, tuple(hard), tuple(soft))
