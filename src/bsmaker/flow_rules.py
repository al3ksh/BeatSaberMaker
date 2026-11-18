"""Rule-based flow post-processing: swing-extension dot stacks and snake runs."""
from __future__ import annotations

import bisect
import math
import random
from collections.abc import Collection, Sequence
from dataclasses import dataclass, replace

import numpy as np

from bsmaker.formats.events import to_events
from bsmaker.formats.model import Note
from bsmaker.playability.checker import Violation, check, check_exact
from bsmaker.playability.config import PlayabilityConfig
from bsmaker.playability.geometry import angle_diff, direction_angle

RUN_SHARE = (0.0, 0.25, 0.5, 0.8)
STACK_CHANCE = (0.0, 0.25, 0.5, 0.8)
SLITHER = "slither"
ROTATION = "rotation"

# Dot stacks. Values measured on reference tech maps (MAYDAY 6b8c, Astronomia e2ba).
_STACK_BEAT_FRACTIONS = (1 / 16, 1 / 12, 1 / 8, 1 / 6, 1 / 4)
_MIN_STACK_SPACING_S = 0.045
_MAX_STACK_SPACING_S = 0.090
_MAX_STACK_SPAN_S = 0.110
_MIN_SAME_HAND_GAP_S = 0.20
_NEXT_AFTER_LAST_DOT_S = 0.13
_DOUBLE_TOL_S = 0.01
_OTHER_HAND_MARGIN_S = 0.05
_CELL_CLEARANCE_S = 0.15
_ONSET_PERCENTILE = 50.0
_SECOND_DOT_CHANCE = 0.5
# No stacks inside weave runs (tuned after the GHOST end-to-end run: stacks mid-stream cut the
# longest weave from 29 to 17 notes). Same weave definition as compare_tech.py.
_WEAVE_MIN_LEN = 8
_WEAVE_MIN_GAP_S = 0.05
_WEAVE_MAX_GAP_S = 0.22

# Loose-dot repair: a dot is a swing extension when it lies on the ray of the same hand's
# previous arrow (or extension dot) within this time (same definition as compare_tech.py).
_EXTENSION_WINDOW_S = 0.35
_UP_DOWN = (0, 1)

# Validation.
_SLIDER_VALIDATION_GAP_S = 0.12
_WINDOW_PAD_S = 1.0
# An arrow at least this long after the hand's previous note is kept by every profile (slider
# tails need < 0.12 s on frame times, which can be up to one 11.6 ms frame shorter). After such
# an arrow the checker's state for that hand no longer depends on earlier notes.
_SYNC_GAP_S = 0.14
_SYNC_EDGE_S = 0.05
_SYNC_TAIL_S = 0.20
_RANGE_MARGIN_S = 0.02
_EPS = 1e-9

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

Key = tuple[float, int]


@dataclass(frozen=True)
class FlowReport:
    level: int
    runs_found: int = 0
    runs_rewritten: int = 0
    notes_rewritten: int = 0
    dot_stacks: int = 0
    dots_added: int = 0
    dots_repaired: int = 0
    reverted: int = 0


def _order(n: Note) -> tuple[float, int, int, int]:
    return (n.time_s, n.color, n.x, n.y)


def _key(n: Note) -> Key:
    return (n.time_s, n.color)


def _time(n: Note) -> float:
    return n.time_s


def stack_spacing_s(bpm: float) -> float | None:
    """Smallest of 1/16, 1/12, 1/8, 1/6, 1/4 beat (seconds) that is >= 45 ms; None if > 90 ms."""
    if not math.isfinite(bpm) or bpm <= 0.0:
        return None
    beat = 60.0 / bpm
    for frac in _STACK_BEAT_FRACTIONS:
        spacing = beat * frac
        if spacing >= _MIN_STACK_SPACING_S - _EPS:
            return spacing if spacing <= _MAX_STACK_SPACING_S + _EPS else None
    return None


def validation_cfg(mask_cfg: PlayabilityConfig) -> PlayabilityConfig:
    """Return a copy of `mask_cfg` that treats stack dots as slider tails."""
    return replace(mask_cfg, slider_gap_s=_SLIDER_VALIDATION_GAP_S)


def _vcfg_hard(notes: Sequence[Note], mask_cfg: PlayabilityConfig) -> frozenset[Violation]:
    events, _ = to_events(notes)
    return frozenset(check(events, validation_cfg(mask_cfg)).hard)


def _hard_ok(
    notes: Sequence[Note],
    mask_cfg: PlayabilityConfig,
    added: Collection[Key],
    base_hard: Collection[Violation],
    lo: float,
    hi: float,
) -> bool:
    """The two flow guarantees for violations timed in [lo, hi].

    (a) Without the flow-added dots, the notes pass `mask_cfg` like the base map must (frame
        check and exact-time check, nothing merged).
    (b) With them, under the slider-tail profile, every hard violation already existed in the
        base map (`base_hard`), and nothing is merged.
    """
    def in_range(v: Violation) -> bool:
        return lo <= v.time_s <= hi

    kept = [n for n in notes if _key(n) not in added] if added else list(notes)
    events, dropped = to_events(kept)
    if dropped or any(in_range(v) for v in check(events, mask_cfg).hard):
        return False
    if any(in_range(v) for v in check_exact(kept, mask_cfg).hard):
        return False
    events, dropped = to_events(notes)
    if dropped:
        return False
    return all(v in base_hard for v in check(events, validation_cfg(mask_cfg)).hard if in_range(v))


def final_flow_ok(
    pre_notes: Sequence[Note], final_notes: Sequence[Note], mask_cfg: PlayabilityConfig,
) -> bool:
    """Whole-map check after the flow layer (the notes flow adds are dots at new times)."""
    pre = sorted(pre_notes, key=_order)
    final = sorted(final_notes, key=_order)
    pre_keys = {_key(n) for n in pre}
    added = frozenset(_key(n) for n in final if _key(n) not in pre_keys)
    return _hard_ok(final, mask_cfg, added, _vcfg_hard(pre, mask_cfg), -math.inf, math.inf)


def _sync_before(notes: Sequence[Note], t: float) -> float:
    """Earliest of each hand's last state-resetting arrow before t (-inf if a hand has none)."""
    i = bisect.bisect_left(notes, t, key=_time) - 1
    cand: dict[int, Note | None] = {0: None, 1: None}
    found: dict[int, float] = {}
    while i >= 0 and len(found) < 2:
        n = notes[i]
        h = n.color
        i -= 1
        if h in found:
            continue
        c = cand[h]
        if c is not None and c.direction != 8 and c.time_s - n.time_s >= _SYNC_GAP_S - _EPS:
            found[h] = c.time_s
        else:
            cand[h] = n
    for h in (0, 1):
        if h not in found:
            c = cand[h]
            if c is None:
                found[h] = math.inf  # no note of this hand before t: nothing to sync
            elif c.direction != 8:
                found[h] = c.time_s  # the hand's first note
            else:
                found[h] = -math.inf
    return min(found.values())


def _sync_after(notes: Sequence[Note], t: float) -> float:
    """Latest of each hand's first state-resetting arrow after t (inf if a hand has none)."""
    i = bisect.bisect_right(notes, t, key=_time)
    prev: dict[int, float | None] = {0: None, 1: None}
    k = i - 1
    while k >= 0 and (prev[0] is None or prev[1] is None):
        if prev[notes[k].color] is None:
            prev[notes[k].color] = notes[k].time_s
        k -= 1
    found: dict[int, float] = {}
    seen = {0: False, 1: False}
    while i < len(notes) and len(found) < 2:
        n = notes[i]
        h = n.color
        i += 1
        if h in found:
            continue
        seen[h] = True
        p = prev[h]
        if n.direction != 8 and (p is None or n.time_s - p >= _SYNC_GAP_S - _EPS):
            found[h] = n.time_s
        prev[h] = n.time_s
    for h in (0, 1):
        if h not in found:
            found[h] = math.inf if seen[h] else -math.inf
    return max(found.values())


def window_ok(
    notes: Sequence[Note],
    t0: float,
    t1: float,
    mask_cfg: PlayabilityConfig,
    *,
    added: Collection[Key] = frozenset(),
    base_hard: Collection[Violation] = frozenset(),
) -> bool:
    """Check a change to the time-sorted `notes` within [t0, t1] with the final-check rules.

    The checked span reaches at least 1 s around the change, and further out to an arrow of
    each hand after which the checker state no longer depends on the change (dots carry
    parity). The checker is causal, so inside that span the cropped check equals the whole-map
    check: a change that passes here cannot fail `final_flow_ok`.
    """
    lo = min(t0 - _WINDOW_PAD_S, _sync_before(notes, t0 - _SYNC_EDGE_S) - 1e-3)
    hi = max(t1 + _WINDOW_PAD_S, _sync_after(notes, t1 + _SYNC_EDGE_S) + _SYNC_TAIL_S)
    a = bisect.bisect_left(notes, lo, key=_time)
    b = bisect.bisect_right(notes, hi, key=_time)
    return _hard_ok(notes[a:b], mask_cfg, added, base_hard,
                    t0 - _RANGE_MARGIN_S, hi - _RANGE_MARGIN_S)


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


def _between(working: Sequence[Note], t0: float, t1: float) -> Sequence[Note]:
    a = bisect.bisect_left(working, t0 - _EPS, key=_time)
    b = bisect.bisect_right(working, t1 + _EPS, key=_time)
    return working[a:b]


def _other_hand_free(working: Sequence[Note], hand: int, t: float, t_last: float) -> bool:
    """No other-hand note in (t + 0.01, t_last + 0.05]; a double at t itself is allowed."""
    return not any(
        n.color != hand and n.time_s > t + _DOUBLE_TOL_S + _EPS
        for n in _between(working, t, t_last + _OTHER_HAND_MARGIN_S)
    )


def _cell_free(working: Sequence[Note], t_s: float, x: int, y: int) -> bool:
    return not any(
        n.x == x and n.y == y
        for n in _between(working, t_s - _CELL_CLEARANCE_S, t_s + _CELL_CLEARANCE_S)
    )


def _vision_ok(working: Sequence[Note], hand: int, t_s: float, x: int, y: int, gap: float) -> bool:
    """The dot neither blocks nor is blocked by the other hand (vision rule, both ways).

    Stack dots are slider tails for the validation profile, so the checker never sees them.
    """
    for n in _between(working, t_s - gap, t_s + gap):
        if n.color == hand or n.x != x:
            continue
        if n.time_s > t_s and y == 1 and x in (1, 2):
            return False
        if n.time_s < t_s and n.y == 1 and n.x in (1, 2):
            return False
    return True


def _weave_inner(notes: Sequence[Note]) -> set[int]:
    """Indices of notes inside a weave run of >= 8 notes, except each run's last note.

    Weave as compare_tech.py measures it: consecutive notes alternate hands 0.05-0.22 s apart
    and each hand's swing flips (>= 135 deg). A stack dot there would break the alternation.
    """
    inner: set[int] = set()
    start = 0
    last: dict[int, Note | None] = {0: None, 1: None}

    def close(end: int) -> None:
        if end - start >= _WEAVE_MIN_LEN:
            inner.update(range(start, end - 1))

    for i in range(1, len(notes)):
        p, n = notes[i - 1], notes[i]
        prev_same = last[n.color]
        ok = n.color != p.color and _WEAVE_MIN_GAP_S < n.time_s - p.time_s <= _WEAVE_MAX_GAP_S + _EPS
        if ok and prev_same is not None:
            a, b = direction_angle(prev_same.direction), direction_angle(n.direction)
            if a is not None and b is not None:
                ok = angle_diff(a, b) >= 135.0
        last[n.color] = n
        if not ok:
            close(i)
            start = i
    close(len(notes))
    return inner


def _gap_ok(nxt: float | None, t: float, span: float) -> bool:
    if nxt is None:
        return True
    return nxt - t >= max(_MIN_SAME_HAND_GAP_S, span + _NEXT_AFTER_LAST_DOT_S) - _EPS


def add_dot_stacks(
    notes: list[Note],
    *,
    bpm: float,
    onset_env: np.ndarray,
    onset_hop_s: float,
    chance: float,
    rng: random.Random,
    mask_cfg: PlayabilityConfig,
    base_hard: Collection[Violation] | None = None,
    phrase=None,
) -> tuple[list[Note], int, int, int]:
    """Add swing-extension dot stacks after eligible arrows; return (notes, stacks, dots, reverted).

    `base_hard` is the validation-profile violation set of the map before the flow layer
    (default: computed from `notes`). `phrase` (a bsmaker.phrase.PatternLinks) makes a repeated
    bar stack where its earlier twin stacked (F7).
    """
    if chance <= 0.0:
        return list(notes), 0, 0, 0

    ordered = sorted(notes, key=_order)
    spacing = stack_spacing_s(bpm)
    env = np.asarray(onset_env, dtype=np.float32)
    if spacing is None or len(env) == 0 or not ordered:
        return ordered, 0, 0, 0

    threshold = float(np.percentile(env, _ONSET_PERCENTILE))
    if base_hard is None:
        base_hard = _vcfg_hard(ordered, mask_cfg)
    vision_gap = mask_cfg.vision_gap_s

    next_same: list[float | None] = [None] * len(ordered)
    last_idx: dict[int, int] = {}
    for idx, n in enumerate(ordered):
        prev_idx = last_idx.get(n.color)
        if prev_idx is not None:
            next_same[prev_idx] = n.time_s
        last_idx[n.color] = idx

    in_weave = _weave_inner(ordered)
    working = list(ordered)
    added: set[Key] = set()
    stacks = 0
    dots_added = 0
    reverted = 0

    def dot_ok(t: float, cell: tuple[int, int], hand: int, t_arrow: float) -> bool:
        return (
            _other_hand_free(working, hand, t_arrow, t)
            and _has_onset(env, onset_hop_s, threshold, t)
            and _cell_free(working, t, cell[0], cell[1])
            and _vision_ok(working, hand, t, cell[0], cell[1], vision_gap)
        )

    for idx, n in enumerate(ordered):
        if n.direction == 8 or idx in in_weave:
            continue
        cells = _ray_cells(n.x, n.y, n.direction)
        if not cells:
            continue

        nxt = next_same[idx]
        if not _gap_ok(nxt, n.time_s, spacing):
            continue
        if not dot_ok(n.time_s + spacing, cells[0], n.color, n.time_s):
            continue

        skip = rng.random() >= chance
        want = phrase.stack_wanted(n.time_s, n.color) if phrase is not None else None  # F7 hook
        if want is not None:
            skip = want == 0
        if skip:
            continue

        n_dots = 1
        span2 = 2.0 * spacing
        if (
            len(cells) >= 2
            and span2 <= _MAX_STACK_SPAN_S + _EPS
            and _gap_ok(nxt, n.time_s, span2)
            and dot_ok(n.time_s + span2, cells[1], n.color, n.time_s)
            and (rng.random() < _SECOND_DOT_CHANCE if want is None else want >= 2)
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
        cand_keys = {_key(c) for c in cand}
        trial = sorted(working + cand, key=_order)
        if window_ok(trial, n.time_s, cand[-1].time_s, mask_cfg,
                     added=added | cand_keys, base_hard=base_hard):
            working = trial
            added |= cand_keys
            stacks += 1
            dots_added += n_dots
            if phrase is not None:  # F7 phrase repetition hook
                phrase.stack_record(n.time_s, n.color, n_dots)
        else:
            reverted += 1

    return working, stacks, dots_added, reverted


def _on_ray(line: tuple[int, int, int, int, float], n: Note) -> bool:
    if n.time_s - line[4] > _EXTENSION_WINDOW_S + _EPS:
        return False
    x, y, dx, dy, _ = line
    rx, ry = n.x - x, n.y - y
    return rx * dy == ry * dx and rx * dx + ry * dy > 0


def _repair_candidates(expected: float) -> list[int]:
    """Arrow directions within 45 deg of the parity-expected angle: up/down first, then closest."""
    cands = [d for d in range(8) if angle_diff(direction_angle(d), expected) <= 45.0 + _EPS]
    return sorted(
        cands, key=lambda d: (d not in _UP_DOWN, angle_diff(direction_angle(d), expected), d),
    )


def repair_loose_dots(
    notes: list[Note],
    *,
    mask_cfg: PlayabilityConfig,
    base_hard: Collection[Violation] | None = None,
) -> tuple[list[Note], int, int]:
    """Turn dots that are not swing extensions into parity-expected arrows.

    A dot is an extension when it lies on the cut ray of the same hand's previous arrow (or
    extension dot) within 0.35 s, as in compare_tech.py. Every other dot whose hand has a
    previous swing becomes an arrow cutting opposite to that swing (a direction within 45 deg,
    straight up/down first); each candidate is validated like any flow change and the dot
    stays if none passes. Returns (notes, repaired, reverted).
    """
    working = sorted(notes, key=_order)
    if base_hard is None:
        base_hard = _vcfg_hard(working, mask_cfg)
    angle: dict[int, float | None] = {0: None, 1: None}
    line: dict[int, tuple[int, int, int, int, float] | None] = {0: None, 1: None}
    repaired = 0
    reverted = 0
    for i in range(len(working)):
        n = working[i]
        h = n.color
        if n.direction != 8:
            angle[h] = direction_angle(n.direction)
            line[h] = (n.x, n.y, *_DIRS[n.direction], n.time_s)
            continue
        cur = line[h]
        prev_angle = angle[h]
        if prev_angle is not None:
            angle[h] = (prev_angle + 180.0) % 360.0  # a dot flips the swing, like advance()
        if cur is not None and _on_ray(cur, n):
            line[h] = (cur[0], cur[1], cur[2], cur[3], n.time_s)
            continue
        line[h] = None
        if prev_angle is None:
            continue
        expected = (prev_angle + 180.0) % 360.0
        for d in _repair_candidates(expected):
            working[i] = replace(n, direction=d)
            if window_ok(working, n.time_s, n.time_s, mask_cfg, base_hard=base_hard):
                repaired += 1
                angle[h] = direction_angle(d)
                line[h] = (n.x, n.y, *_DIRS[d], n.time_s)
                break
        else:
            working[i] = n
            reverted += 1
    return working, repaired, reverted


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
    base_hard: Collection[Violation] | None = None,
) -> tuple[list[Note], int, int, int, int]:
    """Rewrite a share of eligible runs into snakes; return (notes, found, rewritten, n_notes, reverted)."""
    ordered = sorted(notes, key=_order)
    runs = find_runs(ordered)
    if share <= 0.0 or not runs:
        return (list(notes) if share <= 0.0 else ordered), len(runs), 0, 0, 0

    if base_hard is None:
        base_hard = _vcfg_hard(ordered, mask_cfg)
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
            if window_ok(cand, t0, t1, mask_cfg, base_hard=base_hard):
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
    phrase=None,
) -> tuple[list[Note], FlowReport]:
    """Repair loose dots, then snake rewrites, then dot stacks, for flow level 0-3.

    `phrase` (F7, a bsmaker.phrase.PatternLinks or None) is passed to `add_dot_stacks`.
    """
    if level not in (0, 1, 2, 3):
        raise ValueError(f"flow level must be 0, 1, 2, or 3, got {level}")
    if level == 0:
        return list(notes), FlowReport(level=0)

    rng = random.Random(seed)
    base_hard = _vcfg_hard(sorted(notes, key=_order), mask_cfg)
    repaired_notes, dots_repaired, repair_reverted = repair_loose_dots(
        notes, mask_cfg=mask_cfg, base_hard=base_hard,
    )
    after_snakes, runs_found, runs_rewritten, notes_rewritten, snake_reverted = apply_snakes(
        repaired_notes, share=RUN_SHARE[level], rng=rng, mask_cfg=mask_cfg, base_hard=base_hard,
    )
    after_stacks, dot_stacks, dots_added, stack_reverted = add_dot_stacks(
        after_snakes, bpm=bpm, onset_env=onset_env, onset_hop_s=onset_hop_s,
        chance=STACK_CHANCE[level], rng=rng, mask_cfg=mask_cfg, base_hard=base_hard,
        phrase=phrase,
    )
    report = FlowReport(
        level=level,
        runs_found=runs_found,
        runs_rewritten=runs_rewritten,
        notes_rewritten=notes_rewritten,
        dot_stacks=dot_stacks,
        dots_added=dots_added,
        dots_repaired=dots_repaired,
        reverted=repair_reverted + snake_reverted + stack_reverted,
    )
    return after_stacks, report
