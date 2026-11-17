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
from bsmaker.playability.geometry import angle_diff, direction_angle

RUN_SHARE = (0.0, 0.25, 0.5, 0.8)
STACK_CHANCE = (0.0, 0.15, 0.3, 0.5)
SLITHER = "slither"
ROTATION = "rotation"

_MIN_STACK_SPACING_S = 0.045
_MAX_STACK_SPACING_S = 0.090
_MAX_STACK_SPAN_S = 0.120
_MIN_SAME_HAND_GAP_S = 0.35
_OTHER_HAND_MARGIN_S = 0.05
_CELL_CLEARANCE_S = 0.15
_ONSET_PERCENTILE = 70.0
_SLIDER_VALIDATION_GAP_S = 0.12
_WINDOW_PAD_S = 1.0

_MIN_RUN_LEN = 8
_DOUBLE_GAP_S = 0.040
_MIN_RUN_GAP_S = 0.10
_MAX_RUN_GAP_S = 0.30
_MAX_RUN_HAND_GAP_S = 0.60
_ROTATION_PROB = 0.30
_ROTATION_CYCLE: tuple[tuple[int, int, int], ...] = (
    (0, 1, 4),  # L UL
    (1, 0, 6),  # R DL
    (2, 0, 7),  # L DR
    (3, 1, 5),  # R UR
)

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


def find_runs(notes: Sequence[Note]) -> list[tuple[int, int]]:
    """Return half-open [i, j) ranges of >= 8 alternating single-note events (0.1-0.3 s gaps)."""
    n = len(notes)
    if n < _MIN_RUN_LEN:
        return []
    is_double = [False] * n
    for k in range(n - 1):
        if notes[k + 1].time_s - notes[k].time_s < _DOUBLE_GAP_S - 1e-9:
            is_double[k] = True
            is_double[k + 1] = True

    runs: list[tuple[int, int]] = []
    i = 0
    while i < n:
        if is_double[i]:
            i += 1
            continue
        j = i + 1
        while j < n:
            if is_double[j] or notes[j].color == notes[j - 1].color:
                break
            dt = notes[j].time_s - notes[j - 1].time_s
            if not (_MIN_RUN_GAP_S - 1e-9 <= dt <= _MAX_RUN_GAP_S + 1e-9):
                break
            if j - i >= 2 and notes[j].time_s - notes[j - 2].time_s > _MAX_RUN_HAND_GAP_S + 1e-9:
                break
            j += 1
        if j - i >= _MIN_RUN_LEN:
            runs.append((i, j))
        i = j
    return runs


def _prev_hand_angles(notes: Sequence[Note], end: int) -> dict[int, float | None]:
    angles: dict[int, float | None] = {0: None, 1: None}
    for n in notes[:end]:
        ang = direction_angle(n.direction)
        if ang is not None:
            angles[n.color] = ang
        elif angles[n.color] is not None:
            angles[n.color] = (angles[n.color] + 180.0) % 360.0
    return angles


def _best_up_down(prev_angle: float | None) -> int:
    if prev_angle is None:
        return 1
    expected = (prev_angle + 180.0) % 360.0
    return min((1, 0), key=lambda d: angle_diff(direction_angle(d), expected))


def rewrite_run(
    notes: Sequence[Note],
    i: int,
    j: int,
    template: str,
    rng: random.Random,
) -> list[Note]:
    """Return a copy of `notes` with indices i..j-1 rewritten to a snake template."""
    if not (0 <= i <= j <= len(notes)):
        raise ValueError(f"invalid run range [{i}, {j}) for {len(notes)} notes")
    out = list(notes)
    if i == j:
        return out

    prev_angles = _prev_hand_angles(notes, i)
    if template == SLITHER:
        dirs = {h: _best_up_down(prev_angles[h]) for h in (0, 1)}
        lx, rx = 1, 2
        for k in range(i, j):
            n = notes[k]
            d = dirs[n.color]
            dirs[n.color] = 1 - d
            step = rng.choice((-1, 0, 1))
            if n.color == 0:
                lx = max(0, min(2, rx, lx + step))
                nx = lx
            else:
                rx = max(1, lx, min(3, rx + step))
                nx = rx
            out[k] = replace(n, x=nx, y=0, direction=d)
        return out

    if template == ROTATION:
        h0 = notes[i].color
        cands = (0, 2) if h0 == 0 else (1, 3)
        prev_ang = prev_angles[h0]
        if prev_ang is None:
            offset = cands[0]
        else:
            expected = (prev_ang + 180.0) % 360.0
            offset = min(
                cands,
                key=lambda off: angle_diff(direction_angle(_ROTATION_CYCLE[off][2]), expected),
            )
        for step, k in enumerate(range(i, j)):
            x, y, d = _ROTATION_CYCLE[(offset + step) % 4]
            out[k] = replace(notes[k], x=x, y=y, direction=d)
        return out

    raise ValueError(f"unknown snake template: {template!r}")


def apply_snakes(
    notes: list[Note],
    *,
    share: float,
    rng: random.Random,
    mask_cfg: PlayabilityConfig,
) -> tuple[list[Note], int, int, int, int]:
    """Rewrite a share of eligible runs into snakes; return (notes, found, rewritten, n_notes, reverted)."""
    ordered = sorted(notes, key=lambda n: (n.time_s, n.color, n.x, n.y))
    runs = find_runs(ordered)
    if share <= 0.0 or not runs:
        return (list(notes) if share <= 0.0 else ordered), len(runs), 0, 0, 0

    vcfg = validation_cfg(mask_cfg)
    working = ordered
    runs_rewritten = 0
    notes_rewritten = 0
    reverted = 0

    for i, j in runs:
        if rng.random() >= share:
            continue
        chosen = ROTATION if rng.random() < _ROTATION_PROB else SLITHER
        trials = (ROTATION, SLITHER) if chosen == ROTATION else (SLITHER,)
        t0, t1 = working[i].time_s, working[j - 1].time_s
        accepted = False
        for tmpl in trials:
            cand = rewrite_run(working, i, j, tmpl, rng)
            if window_ok(cand, t0, t1, vcfg):
                working = cand
                runs_rewritten += 1
                notes_rewritten += j - i
                accepted = True
                break
        if not accepted:
            reverted += 1

    return working, len(runs), runs_rewritten, notes_rewritten, reverted


def auto_flow_level(styles: Sequence[str]) -> int:
    """Default flow level: 2 when 'tech' is in styles, otherwise 1."""
    return 2 if "tech" in styles else 1


def apply_flow(
    notes: list[Note],
    *,
    bpm: float,
    onset_env: np.ndarray,
    onset_hop_s: float,
    level: int,
    seed: int,
    mask_cfg: PlayabilityConfig,
) -> tuple[list[Note], FlowReport]:
    """Apply snake rewrites then dot stacks for the given flow level (0-3)."""
    if level not in (0, 1, 2, 3):
        raise ValueError(f"flow level must be 0, 1, 2, or 3, got {level}")
    if level == 0:
        return list(notes), FlowReport(level=0)

    rng = random.Random(seed)
    after_snakes, runs_found, runs_rewritten, notes_rewritten, snake_reverted = apply_snakes(
        notes, share=RUN_SHARE[level], rng=rng, mask_cfg=mask_cfg,
    )
    after_stacks, dot_stacks, dots_added, stack_reverted = add_dot_stacks(
        after_snakes, bpm=bpm, onset_env=onset_env, onset_hop_s=onset_hop_s,
        chance=STACK_CHANCE[level], rng=rng, mask_cfg=mask_cfg,
    )
    report = FlowReport(
        level=level,
        runs_found=runs_found,
        runs_rewritten=runs_rewritten,
        notes_rewritten=notes_rewritten,
        dot_stacks=dot_stacks,
        dots_added=dots_added,
        reverted=snake_reverted + stack_reverted,
    )
    return after_stacks, report


