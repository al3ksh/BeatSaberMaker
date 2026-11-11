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
_X = [d[0] for d in _DECODED]
_Y = [d[1] for d in _DECODED]
_ANGLE = [direction_angle(d[2]) for d in _DECODED]

class _Ctx:
    """Per-call invariants shared by all rule functions (built once per allowed/check call)."""

    __slots__ = (
        "hand", "cfg", "last_t", "last_x", "last_y", "gap", "expected", "limit", "rare",
        "other", "ox", "oy", "oangle", "vision_x",
    )

    def __init__(self, state: State, time_s: float, hand: int, cfg: PlayabilityConfig, other: int):
        h = state.left if hand == 0 else state.right
        self.hand = hand
        self.cfg = cfg
        self.last_t = h.time_s
        self.last_x, self.last_y = h.x, h.y
        self.gap = None if h.time_s is None else time_s - h.time_s
        self.expected = None if h.angle is None else (h.angle + 180) % 360
        gap = self.gap
        # Parity limit in degrees: None = no limit (reset), else the allowed angle difference.
        if gap is None or gap >= cfg.reset_gap_s:
            self.limit = None
        elif gap >= cfg.slow_gap_s:
            self.limit = cfg.lenient_angle_deg
        else:
            self.limit = cfg.strict_angle_deg
        self.rare = cfg.rare_tokens[hand]
        self.other = other
        self.ox, self.oy, self.oangle = _X[other], _Y[other], _ANGLE[other]
        self.vision_x = -1
        if state.prev_time_s is not None and time_s - state.prev_time_s < cfg.vision_gap_s:
            prev = state.prev_right if hand == 0 else state.prev_left
            if prev != NONE_TOKEN and _Y[prev] == 1 and _X[prev] in (1, 2):
                self.vision_x = _X[prev]


Rule = Callable[[_Ctx, int], "str | None"]


def _rare_token(c, token):
    return "rare_token" if token in c.rare else None


def _hand_speed(c, token):
    if c.gap is None:
        return None
    cfg = c.cfg
    need = cfg.min_hand_gap_s + cfg.min_hand_gap_per_cell_s * max(
        abs(_X[token] - c.last_x), abs(_Y[token] - c.last_y)
    )
    return "hand_speed" if c.gap < need else None


def _parity_delta(c, token):
    new = _ANGLE[token]
    if c.expected is None or new is None:
        return None
    return angle_diff(new, c.expected)


def _parity(c, token):
    if c.limit is None:
        return None
    d = _parity_delta(c, token)
    return "parity" if d is not None and d > c.limit else None


def _vision_block(c, token):
    return "vision_block" if c.vision_x >= 0 and _X[token] == c.vision_x else None


def _same_cell(c, token):
    if c.other == NONE_TOKEN:
        return None
    return "same_cell" if _X[token] == c.ox and _Y[token] == c.oy else None


def _crossover(c, token):
    if c.other == NONE_TOKEN or not c.cfg.forbid_crossover:
        return None
    xr, xb = (_X[token], c.ox) if c.hand == 0 else (c.ox, _X[token])
    return "crossover" if xr > xb else None


def _handclap(c, token):
    if c.other == NONE_TOKEN or _Y[token] != c.oy:
        return None
    if c.hand == 0:
        xr, xb, ar, ab = _X[token], c.ox, _ANGLE[token], c.oangle
    else:
        xr, xb, ar, ab = c.ox, _X[token], c.oangle, _ANGLE[token]
    if xb - xr != 1:
        return None
    return "handclap" if ar in (0, 45, 315) and ab in (135, 180, 225) else None


HARD_RULES: tuple[Rule, ...] = (
    _rare_token, _hand_speed, _vision_block, _same_cell, _crossover, _handclap, _parity,
)


def _reset(c, token):
    if c.gap is None or c.gap < c.cfg.reset_gap_s:
        return None
    d = _parity_delta(c, token)
    return "reset" if d is not None and d > c.cfg.lenient_angle_deg else None


def _big_jump(c, token):
    if c.gap is None:
        return None
    far = max(abs(_X[token] - c.last_x), abs(_Y[token] - c.last_y)) >= 3
    return "big_jump" if far and c.gap < 0.20 else None


SOFT_RULES: tuple[Rule, ...] = (_reset, _big_jump)


def allowed(
    state: State, time_s: float, hand: int, cfg: PlayabilityConfig, other_token: int = NONE_TOKEN
) -> np.ndarray:
    """Boolean mask over all tokens that break no hard rule for `hand` at `time_s`.

    Only a profile without slider tails (`slider_gap_s == 0`, the mask profile) may drive
    generation: `check()` drops slider tails first, so the mask would disagree with it.
    """
    if cfg.slider_gap_s > 0:
        raise ValueError("allowed() needs a profile with slider_gap_s == 0 (the mask profile)")
    ctx = _Ctx(state, time_s, hand, cfg, other_token)
    mask = np.zeros(VOCAB_SIZE, dtype=bool)
    mask[NONE_TOKEN] = True
    for token in range(1, VOCAB_SIZE):
        for rule in HARD_RULES:
            if rule(ctx, token) is not None:
                break
        else:
            mask[token] = True
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


def drop_slider_tails(events: Sequence[Event], gap_s: float) -> list[Event]:
    """Remove slider tails: same-hand notes < `gap_s` after the hand's last kept note, with a
    compatible direction (either is a dot, or the angles differ by <= 45). `gap_s <= 0` keeps all."""
    if gap_s <= 0:
        return list(events)
    last: list[tuple[float, int] | None] = [None, None]
    out: list[Event] = []
    for e in events:
        tokens = [e.left, e.right]
        for hand in (0, 1):
            token = tokens[hand]
            if token == NONE_TOKEN:
                continue
            prev = last[hand]
            if prev is not None and e.time_s - prev[0] < gap_s:
                a, b = _ANGLE[prev[1]], _ANGLE[token]
                if a is None or b is None or angle_diff(a, b) <= 45:
                    tokens[hand] = NONE_TOKEN
                    continue
            last[hand] = (e.time_s, token)
        if tokens[0] != NONE_TOKEN or tokens[1] != NONE_TOKEN:
            out.append(Event(e.time_s, tokens[0], tokens[1]))
    return out


def check(events: Sequence[Event], cfg: PlayabilityConfig) -> Report:
    """Replay events through the same rules `allowed` uses, after dropping slider tails."""
    events = drop_slider_tails(events, cfg.slider_gap_s)
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
            ctx = _Ctx(state, e.time_s, hand, cfg, other)
            for rule in HARD_RULES:
                name = rule(ctx, token)
                if name:
                    hard.append(Violation(e.time_s, hand, name))
            sctx = _Ctx(state, e.time_s, hand, cfg, NONE_TOKEN)
            for rule in SOFT_RULES:
                name = rule(sctx, token)
                if name:
                    soft.append(Violation(e.time_s, hand, name))
        state = advance(state, e.time_s, e.left, e.right)
    return Report(n_notes, tuple(hard), tuple(soft))
