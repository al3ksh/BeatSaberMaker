"""Rule-based flow post-processing: swing-extension dot stacks and snake runs."""
from __future__ import annotations

import math
import random
from collections.abc import Sequence
from dataclasses import dataclass, replace

import numpy as np

from bsmaker.formats.events import to_events
from bsmaker.formats.model import Note
from bsmaker.playability.checker import check
from bsmaker.playability.config import PlayabilityConfig

RUN_SHARE = (0.0, 0.25, 0.5, 0.8)
STACK_CHANCE = (0.0, 0.15, 0.3, 0.5)

_MIN_STACK_SPACING_S = 0.045
_MAX_STACK_SPACING_S = 0.090
_MAX_STACK_SPAN_S = 0.120
_MIN_SAME_HAND_GAP_S = 0.35
_OTHER_HAND_MARGIN_S = 0.05
_CELL_CLEARANCE_S = 0.15
_ONSET_PERCENTILE = 70.0
_SLIDER_VALIDATION_GAP_S = 0.12
_WINDOW_PAD_S = 1.0

_DIRS: dict[int, tuple[int, int]] = {
    0: (0, 1),    # up
    1: (0, -1),   # down
    2: (-1, 0),   # left
    3: (1, 0),    # right
    4: (-1, 1),   # up-left
    5: (1, 1),    # up-right
    6: (-1, -1),  # down-left
    7: (1, -1),   # down-right
}


@dataclass(frozen=True)
class FlowReport:
    level: int
    runs_found: int = 0
    runs_rewritten: int = 0
    notes_rewritten: int = 0
    dot_stacks: int = 0
    dots_added: int = 0
    reverted: int = 0


def stack_spacing_s(bpm: float) -> float | None:
    """Return 1/8 beat (or 1/4 beat if < 45 ms) in seconds if within [45, 90] ms."""
    if not math.isfinite(bpm) or bpm <= 0.0:
        return None
    eighth = 60.0 / bpm / 8.0
    spacing = eighth if eighth >= _MIN_STACK_SPACING_S else 60.0 / bpm / 4.0
    if not (_MIN_STACK_SPACING_S <= spacing <= _MAX_STACK_SPACING_S):
        return None
    return spacing


def validation_cfg(mask_cfg: PlayabilityConfig) -> PlayabilityConfig:
    """Return a copy of `mask_cfg` that treats stack dots as slider tails."""
    return replace(mask_cfg, slider_gap_s=_SLIDER_VALIDATION_GAP_S)


def window_ok(notes: Sequence[Note], t0: float, t1: float, cfg: PlayabilityConfig) -> bool:
    """Check notes within [t0 - 1 s, t1 + 1 s] for hard playability violations or drops."""
    sub = [n for n in notes if t0 - _WINDOW_PAD_S <= n.time_s <= t1 + _WINDOW_PAD_S]
    events, dropped = to_events(sub)
    return dropped == 0 and not check(events, cfg).hard


def _ray_cells(x: int, y: int, direction: int) -> list[tuple[int, int]]:
    vec = _DIRS.get(direction)
    if vec is None:
        return []
    dx, dy = vec
    cells: list[tuple[int, int]] = []
    for step in (1, 2):
        cx, cy = x + step * dx, y + step * dy
        if 0 <= cx <= 3 and 0 <= cy <= 2:
            cells.append((cx, cy))
        else:
            break
    return cells


def _has_onset(env: np.ndarray, hop_s: float, threshold: float, t_s: float) -> bool:
    if hop_s <= 0.0:
        return False
    k = round(t_s / hop_s)
    if k < 0 or k >= len(env):
        return False
    lo = max(0, k - 1)
    hi = min(len(env), k + 2)
    return float(np.max(env[lo:hi])) >= threshold


def _other_hand_busy(working: Sequence[Note], hand: int, t0: float, t1: float) -> bool:
    return any(
        n.color != hand and t0 - 1e-9 <= n.time_s <= t1 + 1e-9
        for n in working
    )


def _cell_free(working: Sequence[Note], t_s: float, x: int, y: int) -> bool:
    return not any(
        n.x == x and n.y == y and abs(n.time_s - t_s) <= _CELL_CLEARANCE_S + 1e-9
        for n in working
    )


def add_dot_stacks(
    notes: list[Note],
    *,
    bpm: float,
    onset_env: np.ndarray,
    onset_hop_s: float,
    chance: float,
    rng: random.Random,
    mask_cfg: PlayabilityConfig,
) -> tuple[list[Note], int, int, int]:
    """Add swing-extension dot stacks after eligible arrows; return (notes, stacks, dots, reverted)."""
    if chance <= 0.0:
        return list(notes), 0, 0, 0

    ordered = sorted(notes, key=lambda n: (n.time_s, n.color, n.x, n.y))
    spacing = stack_spacing_s(bpm)
    env = np.asarray(onset_env, dtype=np.float32)
    if spacing is None or len(env) == 0 or not ordered:
        return ordered, 0, 0, 0

    threshold = float(np.percentile(env, _ONSET_PERCENTILE))
    vcfg = validation_cfg(mask_cfg)

    next_same: list[float | None] = [None] * len(ordered)
    last_idx: dict[int, int] = {}
    for idx, n in enumerate(ordered):
        prev_idx = last_idx.get(n.color)
        if prev_idx is not None:
            next_same[prev_idx] = n.time_s
        last_idx[n.color] = idx

    working = list(ordered)
    stacks = 0
    dots_added = 0
    reverted = 0

    for idx, n in enumerate(ordered):
        if n.direction == 8:
            continue
        cells = _ray_cells(n.x, n.y, n.direction)
        if not cells:
            continue

        nxt = next_same[idx]
        t1 = n.time_s + spacing
        if nxt is not None and nxt - n.time_s < max(_MIN_SAME_HAND_GAP_S, spacing + 0.2):
            continue
        if _other_hand_busy(working, n.color, n.time_s, t1 + _OTHER_HAND_MARGIN_S):
            continue
        if not _has_onset(env, onset_hop_s, threshold, t1):
            continue
        if not _cell_free(working, t1, cells[0][0], cells[0][1]):
            continue

        if rng.random() >= chance:
            continue

        n_dots = 1
        span2 = 2.0 * spacing
        if len(cells) >= 2 and span2 <= _MAX_STACK_SPAN_S + 1e-9:
            t2 = n.time_s + span2
            gap2_ok = nxt is None or nxt - n.time_s >= max(_MIN_SAME_HAND_GAP_S, span2 + 0.2)
            if (
                gap2_ok
                and not _other_hand_busy(working, n.color, n.time_s, t2 + _OTHER_HAND_MARGIN_S)
                and _has_onset(env, onset_hop_s, threshold, t2)
                and _cell_free(working, t2, cells[1][0], cells[1][1])
                and rng.random() < 0.5
            ):
                n_dots = 2

        cand = [
            Note(
                time_s=n.time_s + (k + 1) * spacing,
                beat=(n.time_s + (k + 1) * spacing) * bpm / 60.0,
                x=cells[k][0],
                y=cells[k][1],
                color=n.color,
                direction=8,
            )
            for k in range(n_dots)
        ]
        trial = working + cand
        if window_ok(trial, n.time_s, n.time_s + n_dots * spacing, vcfg):
            working = trial
            stacks += 1
            dots_added += n_dots
        else:
            reverted += 1

    working.sort(key=lambda n: (n.time_s, n.color, n.x, n.y))
    return working, stacks, dots_added, reverted
