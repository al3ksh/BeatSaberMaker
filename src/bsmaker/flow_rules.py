"""Rule-based flow post-processing: swing-extension dot stacks and snake runs."""
from __future__ import annotations

import bisect
import math
import random
from collections.abc import Collection, Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import numpy as np

from bsmaker.formats.events import to_events
from bsmaker.formats.model import Note
from bsmaker.playability.checker import Violation, check, check_exact
from bsmaker.playability.config import PlayabilityConfig
from bsmaker.playability.geometry import angle_diff, direction_angle

if TYPE_CHECKING:
    from bsmaker.audio.bar_similarity import BarRepeats

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

# Stack profiles (patterns.txt F5). "tech" keeps the dot-tail stacks above. "window" (flow
# level 1, the balanced default) makes same-time arrow stacks: the reference stack swing
# share is 0.051 with no dot-tail sliders (target 4-6% of swings).
WINDOW = "window"
TECH = "tech"
_WINDOW_STACK_SHARE = 0.05
# A stack is one swing through two cells: a calm swing (>= 0.20 s to the hand's neighbours,
# as for dot stacks), no other-hand note within 0.10 s except an exact double, and both
# cells on the hand's own half (left x <= 1, right x >= 2) so stacks never cross.
_WINDOW_OTHER_HAND_S = 0.10
_OWN_SIDE = {0: (0, 1), 1: (2, 3)}
# Standalone dots (window profile, patterns.txt G6/F5). Reference: dot-only swings are 4.4% of
# swings (dot share 0.054), 91% of them after a same-hand rest of >= 1 beat, and 98% are played
# as the flip of the previous swing (next arrow within 90 deg of the arrow before the dot:
# 0 deg 54%, 45 deg 37%, 90 deg 7%; .superpowers/fixes/dots-scratch/ref_dots.py, 329 dots),
# which is what the checker assumes for a dot. Alex: no loose dots inside dense patterns.
# So a dot may sit (or a model dot may stay) on a single swing that is
# - >= 1 beat and >= slow_gap_s after the same hand's previous arrow (a gap the checker treats
#   as fast is no rest; this also guards double-time BPM grids),
# - >= 0.20 s before the hand's next note (not a stack or slider head),
# - not part of a double (reference: 3.6% of dot swings are in a double),
# - calm: the nearest other-time note (either hand) on at least one side is > 0.30 s away;
#   both sides <= 0.30 s is a stream of > 3.3 notes/s (same bound as _MAX_RUN_GAP_S).
# Target 4.5% of swings, counting the dots the map already has, so the map ends near the
# reference 4.4% and never above it, and at most 25% of the calm rest spots (reference
# difficulties that use standalone dots put them on 10-23% of these spots, median 5.5%): a
# dense song with few calm spots gets fewer dots, not a dot on every other calm swing. Swings that would be a reset (< 90 deg from the previous
# swing) or a big jump (>= 3 cells from the previous swing's end to this swing's start) go
# first: there a dot also removes an awkward swing.
_FREE_DOT_REST_BEATS = 1.0
_FREE_DOT_JUMP_CELLS = 3
_FREE_DOT_TARGET_SHARE = 0.045
_FREE_DOT_SPOT_SHARE = 0.25
_FREE_DOT_DENSE_GAP_S = 0.30
_SWING_MERGE_S = 0.09  # same-hand notes this close are one swing (pattern analysis)
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
# Matches the reference windmill (rotation) swing share of about 0.10 (patterns.txt F2).
_ROTATION_PROB = 0.30
# SLITHER = the reference diagonal sweep (patterns.txt F2; motifs "10DL 31UR 10DL" in 21/23
# reference difficulties, "12U 10D 01UL", "31UR 20DL 22U"). Each hand keeps its anchor pair:
# down cuts on its inner bottom cell (straight down, or down-inward after an outward-up
# diagonal), up cuts alternately outward-diagonal on its outer middle cell (the reference's
# 27%/26% anchor cells) or straight up above the down cut. Up cuts never sit on row 0.
_SWEEP_DOWN_CELL = {0: (1, 0), 1: (2, 0)}
_SWEEP_UP_CELL = {0: (0, 1), 1: (3, 1)}
_SWEEP_DOWN_DIAG = {0: 7, 1: 6}  # DR (left), DL (right)
_SWEEP_UP_DIAG = {0: 4, 1: 5}  # UL (left), UR (right)
# Full-height (row delta 2) straight ups only with a same-hand gap >= 0.25 s; shorter gaps
# (0.15-0.25 s, and below 0.15 s) put the up cut on row 1 (patterns.txt F2 thresholds).
_SWEEP_FULL_HEIGHT_GAP_S = 0.25
# Lift (patterns.txt F3): a same-hand straight up right after a straight down, both on row 0,
# moves up. Gap thresholds from the reference: >= 0.30 s full height (y=2 above the down),
# 0.18-0.30 s one row (y=1); an inner-column down with >= 0.25 s takes the outward diagonal
# (left UL at (0,1), right UR at (3,1)). The gap is the shorter of the gaps to the hand's
# previous and next note, so the hand can also get back down in time.
_LIFT_MIN_GAP_S = 0.18
_LIFT_FULL_GAP_S = 0.30
_LIFT_DIAG_GAP_S = 0.25
_LIFT_INNER_X = {0: 1, 1: 2}
_LIFT_DIAG = {0: (4, 0, 1), 1: (5, 3, 1)}  # (direction, x, y): UL left, UR right
# Per-map stop targets = reference IQR edges (patterns.txt F3): up cuts on the bottom row
# <= 0.20 and flat (same-row) U/D flips <= 0.30. The F3 row-0 target (<= 0.50) is not a stop
# condition: row 0 is mostly down cuts, and chasing it on the F1 balanced maps lifted every
# map down to the floor and pushed flat flips to 0.12-0.20 (reference median 0.23).
# Lifting also stops at the reference medians (up cuts on row 0 0.152, flat flips 0.225),
# so the rule cannot overshoot the reference.
_LIFT_STOP_UP_BOTTOM = 0.20
_LIFT_STOP_FLAT_FLIP = 0.30
_LIFT_FLOOR_UP_BOTTOM = 0.15
_LIFT_FLOOR_FLAT_FLIP = 0.225
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
    ups_lifted: int = 0
    arrow_stacks: int = 0
    free_dots: int = 0
    bars_replayed: int = 0


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
    for view in _stack_views(kept):
        events, dropped = to_events(view)
        if dropped or any(in_range(v) for v in check(events, mask_cfg).hard):
            return False
        if any(in_range(v) for v in check_exact(view, mask_cfg).hard):
            return False
    vcfg = validation_cfg(mask_cfg)
    for view in _stack_views(notes):
        events, dropped = to_events(view)
        if dropped:
            return False
        if not all(v in base_hard for v in check(events, vcfg).hard if in_range(v)):
            return False
    return True


def _stack_views(notes: Sequence[Note]) -> list[list[Note]]:
    """The notes as the checker sees them, once per way of reading same-time arrow stacks.

    A same-time arrow stack (one hand, identical time, same direction) is one swing through
    several cells. to_events keeps the first note of the stack (by x, y) and drops the rest,
    so the checker sees one cell. Every hard rule must hold with the stack read as its first
    cell and as its last cell. Any other same-time group stays and fails as merged notes.
    """
    groups: dict[Key, list[int]] = {}
    for i, n in enumerate(notes):
        groups.setdefault(_key(n), []).append(i)
    first: set[int] = set()
    last: set[int] = set()
    for idx in groups.values():
        if len(idx) < 2:
            continue
        members = sorted(idx, key=lambda i: _order(notes[i]))
        dirs = {notes[i].direction for i in members}
        if len(dirs) != 1 or 8 in dirs:
            continue
        first.update(members[1:])
        last.update(members[:-1])
    if not first:
        return [list(notes)]
    return [[n for i, n in enumerate(notes) if i not in drop] for drop in (first, last)]


def _added_dot_tails(pre_notes: Sequence[Note], final_notes: Sequence[Note]) -> frozenset[Key]:
    """Keys of flow-added TECH dot-tail sliders (new dots < 0.12 s after a same-hand kept note)."""
    pre_keys = {_key(n) for n in pre_notes}
    added: set[Key] = set()
    last_t: dict[int, float] = {}
    for n in final_notes:
        prev_t = last_t.get(n.color)
        if (
            n.direction == 8
            and _key(n) not in pre_keys
            and prev_t is not None
            and n.time_s - prev_t < _SLIDER_VALIDATION_GAP_S - _EPS
        ):
            added.add(_key(n))
        else:
            last_t[n.color] = n.time_s
    return frozenset(added)


def final_flow_ok(
    pre_notes: Sequence[Note], final_notes: Sequence[Note], mask_cfg: PlayabilityConfig,
) -> bool:
    """Whole-map check after the flow layer (only TECH dot-tail sliders are omitted from `kept`)."""
    pre = sorted(pre_notes, key=_order)
    final = sorted(final_notes, key=_order)
    added = _added_dot_tails(pre, final)
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
    profile: str = TECH,
) -> tuple[list[Note], int, int, int]:
    """Add stacks to eligible arrows; return (notes, stacks, dots, reverted).

    Profile "tech": swing-extension dot stacks after the arrow (slider tails). Profile
    "window": same-time arrow stacks on about 5% of swings (`chance` > 0 switches them on;
    dots is then 0). `base_hard` is the validation-profile violation set of the map before
    the flow layer (default: computed from `notes`).
    """
    if profile not in (TECH, WINDOW):
        raise ValueError(f"unknown stack profile: {profile!r}")
    if chance <= 0.0:
        return list(notes), 0, 0, 0
    if profile == WINDOW:
        out, stacks, reverted = _add_arrow_stacks(
            notes, onset_env=onset_env, onset_hop_s=onset_hop_s, rng=rng, mask_cfg=mask_cfg,
            base_hard=base_hard,
        )
        return out, stacks, 0, reverted

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

        if rng.random() >= chance:
            continue

        n_dots = 1
        span2 = 2.0 * spacing
        if (
            len(cells) >= 2
            and span2 <= _MAX_STACK_SPAN_S + _EPS
            and _gap_ok(nxt, n.time_s, span2)
            and dot_ok(n.time_s + span2, cells[1], n.color, n.time_s)
            and rng.random() < _SECOND_DOT_CHANCE
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
        else:
            reverted += 1

    return working, stacks, dots_added, reverted


def _swing_count(notes: Sequence[Note]) -> int:
    """Swings as pattern analysis counts them: same-hand notes within 0.09 s are one swing."""
    count = 0
    last_t: dict[int, float] = {}
    for n in notes:
        t_prev = last_t.get(n.color)
        last_t[n.color] = n.time_s
        if t_prev is None or n.time_s - t_prev > _SWING_MERGE_S + _EPS:
            count += 1
    return count


def _own_side(hand: int, x: int) -> bool:
    return x in _OWN_SIDE[hand]


def _calm_swing(notes: Sequence[Note], i: int, prev: Sequence[int | None],
                nxt: Sequence[int | None], gap: float) -> bool:
    t = notes[i].time_s
    p, q = prev[i], nxt[i]
    return ((p is None or t - notes[p].time_s >= gap - _EPS)
            and (q is None or notes[q].time_s - t >= gap - _EPS))


def _other_hand_clear(working: Sequence[Note], hand: int, t: float) -> bool:
    """No other-hand note within 0.10 s, except an exact double on that hand's own half."""
    for m in _between(working, t - _WINDOW_OTHER_HAND_S, t + _WINDOW_OTHER_HAND_S):
        if m.color == hand:
            continue
        if abs(m.time_s - t) > _DOUBLE_TOL_S or not _own_side(m.color, m.x):
            return False
    return True


def _add_arrow_stacks(
    notes: Sequence[Note],
    *,
    onset_env: np.ndarray,
    onset_hop_s: float,
    rng: random.Random,
    mask_cfg: PlayabilityConfig,
    base_hard: Collection[Violation] | None,
) -> tuple[list[Note], int, int]:
    """Window profile: same-time arrow stacks on ~5% of swings. Returns (notes, stacks, reverted).

    The second note copies the arrow at the same time, one cell along the cut line behind it
    (or ahead of it when behind leaves the grid or the hand's half). Candidates are calm,
    loud (onset >= p50) swings outside weave runs, tried in random order.
    """
    ordered = sorted(notes, key=_order)
    env = np.asarray(onset_env, dtype=np.float32)
    if not ordered or len(env) == 0:
        return ordered, 0, 0
    budget = round(_WINDOW_STACK_SHARE * _swing_count(ordered))
    if budget <= 0:
        return ordered, 0, 0
    if base_hard is None:
        base_hard = _vcfg_hard(ordered, mask_cfg)
    threshold = float(np.percentile(env, _ONSET_PERCENTILE))
    vision_gap = mask_cfg.vision_gap_s
    prev, nxt = _same_hand_neighbours(ordered)
    in_weave = _weave_inner(ordered)
    cands = [
        i for i, n in enumerate(ordered)
        if n.direction in _DIRS and i not in in_weave and _own_side(n.color, n.x)
        and _calm_swing(ordered, i, prev, nxt, _MIN_SAME_HAND_GAP_S)
        and _has_onset(env, onset_hop_s, threshold, n.time_s)
    ]
    rng.shuffle(cands)

    working = ordered
    stacks = 0
    reverted = 0
    for i in cands:
        if stacks >= budget:
            break
        n = ordered[i]
        if not _other_hand_clear(working, n.color, n.time_s):
            continue
        dx, dy = _DIRS[n.direction]
        tried = False
        for cx, cy in ((n.x + dx, n.y + dy), (n.x - dx, n.y - dy)):
            if not (0 <= cy <= 2 and _own_side(n.color, cx)):
                continue
            if not (_cell_free(working, n.time_s, cx, cy)
                    and _vision_ok(working, n.color, n.time_s, cx, cy, vision_gap)):
                continue
            tried = True
            trial = sorted([*working, replace(n, x=cx, y=cy)], key=_order)
            if window_ok(trial, n.time_s, n.time_s, mask_cfg, base_hard=base_hard):
                working = trial
                stacks += 1
                break
        else:
            reverted += tried
    return working, stacks, reverted


def _rest_dot_spots(
    notes: Sequence[Note],
    prev: Sequence[int | None],
    nxt: Sequence[int | None],
    *,
    bpm: float,
    mask_cfg: PlayabilityConfig,
) -> list[bool]:
    """Per time-sorted note: a calm single swing after a same-hand rest (see _FREE_DOT_*).

    The note's own direction and its previous note's direction are not checked here.
    """
    spots = [False] * len(notes)
    if not notes or not math.isfinite(bpm) or bpm <= 0.0:
        return spots
    rest = max(_FREE_DOT_REST_BEATS * 60.0 / bpm, mask_cfg.slow_gap_s)
    times = [n.time_s for n in notes]
    for i, n in enumerate(notes):
        p, q = prev[i], nxt[i]
        t = n.time_s
        if p is None or t - notes[p].time_s < rest - _EPS:
            continue
        if q is not None and notes[q].time_s - t < _MIN_SAME_HAND_GAP_S - _EPS:
            continue
        a = bisect.bisect_left(times, t - _DOUBLE_TOL_S - _EPS)
        b = bisect.bisect_right(times, t + _DOUBLE_TOL_S + _EPS)
        if any(notes[k].color != n.color for k in range(a, b)):
            continue  # part of a double
        gap_before = t - times[a - 1] if a > 0 else math.inf
        gap_after = times[b] - t if b < len(times) else math.inf
        spots[i] = max(gap_before, gap_after) > _FREE_DOT_DENSE_GAP_S + _EPS
    return spots


def _dot_target(notes: Sequence[Note], spots: int) -> int:
    """Dots a window-profile map should end with: 4.5% of swings, <= 25% of `spots`."""
    return min(round(_FREE_DOT_TARGET_SHARE * _swing_count(notes)),
               math.floor(_FREE_DOT_SPOT_SHARE * spots))


def add_free_dots(
    notes: list[Note],
    *,
    bpm: float,
    rng: random.Random,
    mask_cfg: PlayabilityConfig,
    base_hard: Collection[Violation] | None = None,
) -> tuple[list[Note], int, int]:
    """Window profile: turn swings after a rest into dots. Returns (notes, dots, reverted).

    Candidates are single arrows on a calm rest spot (`_rest_dot_spots`) whose same-hand
    neighbours are arrows (no dot runs). Dots are added until the map has 4.5% of its swings
    as dots, counting the dots it already has; resets and big jumps go first, each group in
    random order. Each change is validated like any flow change.
    """
    ordered = sorted(notes, key=_order)
    if not ordered or not math.isfinite(bpm) or bpm <= 0.0:
        return ordered, 0, 0
    prev, nxt = _same_hand_neighbours(ordered)
    spots = _rest_dot_spots(ordered, prev, nxt, bpm=bpm, mask_cfg=mask_cfg)
    budget = _dot_target(ordered, sum(spots)) - sum(1 for n in ordered if n.direction == 8)
    if budget <= 0:
        return ordered, 0, 0
    if base_hard is None:
        base_hard = _vcfg_hard(ordered, mask_cfg)
    awkward: list[int] = []
    plain: list[int] = []
    for i, n in enumerate(ordered):
        p = prev[i]
        if not spots[i] or n.direction == 8 or p is None or ordered[p].direction == 8:
            continue
        if nxt[i] is not None and ordered[nxt[i]].direction == 8:
            continue
        q = ordered[p]
        reset = angle_diff(direction_angle(n.direction), direction_angle(q.direction)) < 90.0
        (qx, qy), (nx, ny) = _DIRS[q.direction], _DIRS[n.direction]
        travel = max(abs((n.x - nx) - (q.x + qx)), abs((n.y - ny) - (q.y + qy)))
        (awkward if reset or travel >= _FREE_DOT_JUMP_CELLS else plain).append(i)
    rng.shuffle(awkward)
    rng.shuffle(plain)

    working = list(ordered)
    dots = 0
    reverted = 0
    for i in awkward + plain:
        if dots >= budget:
            break
        p, q = prev[i], nxt[i]
        if working[p].direction == 8 or (q is not None and working[q].direction == 8):
            continue  # a neighbour became a dot
        n = working[i]
        working[i] = replace(n, direction=8)
        if window_ok(working, n.time_s, n.time_s, mask_cfg, base_hard=base_hard):
            dots += 1
        else:
            working[i] = n
            reverted += 1
    return working, dots, reverted


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
    bpm: float | None = None,
) -> tuple[list[Note], int, int]:
    """Turn dots that are not swing extensions into parity-expected arrows.

    A dot is an extension when it lies on the cut ray of the same hand's previous arrow (or
    extension dot) within 0.35 s, as in compare_tech.py. With `bpm` (window profile), a dot
    on a calm rest spot after an arrow (`_rest_dot_spots`) also stays, up to 4.5% of swings
    (longest rests first): the reference's standalone dot. Every other dot whose hand has a
    previous swing becomes an arrow cutting opposite to that swing (a direction within 45 deg,
    straight up/down first); each candidate is validated like any flow change and the dot
    stays if none passes. Returns (notes, repaired, reverted).
    """
    working = sorted(notes, key=_order)
    if base_hard is None:
        base_hard = _vcfg_hard(working, mask_cfg)
    keep: set[int] = set()
    if bpm is not None:
        prev, nxt = _same_hand_neighbours(working)
        spots = _rest_dot_spots(working, prev, nxt, bpm=bpm, mask_cfg=mask_cfg)
        rest_dots = [
            i for i, n in enumerate(working)
            if n.direction == 8 and spots[i] and working[prev[i]].direction != 8
        ]
        rest_dots.sort(key=lambda i: (working[prev[i]].time_s - working[i].time_s, i))
        keep = set(rest_dots[:_dot_target(working, sum(spots))])
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
        if prev_angle is None or i in keep:
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


def _same_hand_neighbours(notes: Sequence[Note]) -> tuple[list[int | None], list[int | None]]:
    prev: list[int | None] = [None] * len(notes)
    nxt: list[int | None] = [None] * len(notes)
    last: dict[int, int] = {}
    for i, n in enumerate(notes):
        p = last.get(n.color)
        if p is not None:
            prev[i] = p
            nxt[p] = i
        last[n.color] = i
    return prev, nxt


def _lift_metrics(notes: Sequence[Note]) -> tuple[float, float]:
    """(up cuts on row 0, flat U/D flips) as the F3 stop targets count them.

    Same-hand notes within 0.09 s form one swing, counted once by its first note.
    """
    ups = ups_bottom = flips = flat = 0
    head: dict[int, Note] = {}
    last_t: dict[int, float] = {}
    for n in notes:
        t_prev = last_t.get(n.color)
        last_t[n.color] = n.time_s
        if t_prev is not None and n.time_s - t_prev <= _SWING_MERGE_S + _EPS:
            continue
        p = head.get(n.color)
        head[n.color] = n
        if n.direction in (0, 4, 5):
            ups += 1
            ups_bottom += n.y == 0
        if p is not None and {p.direction, n.direction} == {0, 1}:
            flips += 1
            flat += p.y == n.y
    return ups_bottom / max(ups, 1), flat / max(flips, 1)


def _lift_done(notes: Sequence[Note]) -> bool:
    up_bottom, flat_flip = _lift_metrics(notes)
    if up_bottom <= _LIFT_FLOOR_UP_BOTTOM + _EPS or flat_flip <= _LIFT_FLOOR_FLAT_FLIP + _EPS:
        return True
    return up_bottom <= _LIFT_STOP_UP_BOTTOM + _EPS and flat_flip <= _LIFT_STOP_FLAT_FLIP + _EPS


def _lift_targets(hand: int, down_x: int, gap: float) -> list[tuple[int, int, int]]:
    """(direction, x, y) moves for a flat up cut, preferred first (patterns.txt F3)."""
    targets: list[tuple[int, int, int]] = []
    if down_x == _LIFT_INNER_X[hand] and gap >= _LIFT_DIAG_GAP_S - _EPS:
        targets.append(_LIFT_DIAG[hand])
    if gap >= _LIFT_FULL_GAP_S - _EPS:
        targets.append((0, down_x, 2))
    targets.append((0, down_x, 1))
    return targets


def lift_flat_ups(
    notes: list[Note],
    *,
    mask_cfg: PlayabilityConfig,
    rng: random.Random,
    base_hard: Collection[Violation] | None = None,
) -> tuple[list[Note], int, int]:
    """Move bottom-row up cuts that follow a bottom-row down cut up (patterns.txt F3).

    Candidates are single straight ups on row 0 right after the same hand's single straight
    down on row 0, with >= 0.18 s to the hand's previous and next note. They are tried in
    random order until the map meets the F3 stop targets. Each move is validated like any
    flow change. Returns (notes, lifted, reverted).
    """
    working = sorted(notes, key=_order)
    if not working or _lift_done(working):
        return working, 0, 0
    if base_hard is None:
        base_hard = _vcfg_hard(working, mask_cfg)
    prev, nxt = _same_hand_neighbours(working)

    def single(i: int, other: int | None) -> bool:
        return other is None or abs(working[other].time_s - working[i].time_s) > _SWING_MERGE_S

    cands: list[tuple[int, int, float]] = []
    for i, n in enumerate(working):
        p = prev[i]
        if n.direction != 0 or n.y != 0 or p is None:
            continue
        d = working[p]
        if d.direction != 1 or d.y != 0 or not single(p, prev[p]):
            continue
        gap = n.time_s - d.time_s
        if nxt[i] is not None:
            gap = min(gap, working[nxt[i]].time_s - n.time_s)
        if gap >= _LIFT_MIN_GAP_S - _EPS:
            cands.append((i, d.x, gap))
    rng.shuffle(cands)

    vision_gap = mask_cfg.vision_gap_s
    lifted = 0
    reverted = 0
    for i, down_x, gap in cands:
        if _lift_done(working):
            break
        n = working[i]
        for d, x, y in _lift_targets(n.color, down_x, gap):
            if not (_cell_free(working, n.time_s, x, y)
                    and _vision_ok(working, n.color, n.time_s, x, y, vision_gap)):
                continue
            working[i] = replace(n, x=x, y=y, direction=d)
            if window_ok(working, n.time_s, n.time_s, mask_cfg, base_hard=base_hard):
                lifted += 1
                break
            working[i] = n
        else:
            reverted += 1
    return working, lifted, reverted


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


def _prev_same_hand_time(notes: Sequence[Note], k: int) -> float | None:
    h = notes[k].color
    for m in range(k - 1, -1, -1):
        if notes[m].color == h:
            return notes[m].time_s
    return None


def _rewrite_sweep(
    notes: Sequence[Note],
    out: list[Note],
    i: int,
    j: int,
    prev_angles: dict[int, float | None],
    rng: random.Random,
) -> list[Note]:
    """SLITHER: the reference diagonal sweep, each hand on its anchor pair (see _SWEEP_*)."""
    angle = dict(prev_angles)
    down_next = {h: _best_up_down(prev_angles[h]) == 1 for h in (0, 1)}
    up_diag = {h: rng.random() < 0.5 for h in (0, 1)}
    last_up_diag: dict[int, bool | None] = {0: None, 1: None}
    first = {0: True, 1: True}
    for k in range(i, j):
        n = notes[k]
        h = n.color
        prev_t = _prev_same_hand_time(notes, k)
        gap = math.inf if prev_t is None else n.time_s - prev_t
        dx, dy = _SWEEP_DOWN_CELL[h]
        if down_next[h]:
            diag = (_SWEEP_DOWN_DIAG[h], dx, dy)
            vert = (1, dx, dy)
            cands = [diag, vert] if last_up_diag[h] else [vert, diag]
        else:
            uy = 2 if gap >= _SWEEP_FULL_HEIGHT_GAP_S - _EPS else 1
            diag = (_SWEEP_UP_DIAG[h], *_SWEEP_UP_CELL[h])
            vert = (0, dx, uy)
            cands = [diag, vert] if up_diag[h] else [vert, diag]
        prev = angle[h]
        if prev is not None:
            expected = (prev + 180.0) % 360.0
            ok = [c for c in cands if angle_diff(direction_angle(c[0]), expected) <= 45.0 + _EPS]
            if first[h]:
                # Entering the run: the cut closest to the hand's parity (ties keep preference).
                ok = sorted(ok or cands, key=lambda c: angle_diff(direction_angle(c[0]), expected))
            cands = ok or cands
        d, x, y = cands[0]
        if not down_next[h]:
            last_up_diag[h] = d != 0
            up_diag[h] = not last_up_diag[h]
        out[k] = replace(n, x=x, y=y, direction=d)
        angle[h] = direction_angle(d)
        down_next[h] = not down_next[h]
        first[h] = False
    return out


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
        return _rewrite_sweep(notes, out, i, j, prev_angles, rng)

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


def stack_profile(level: int) -> str:
    """Stack profile for a flow level: 1 (balanced default) window, 2-3 (tech) tech."""
    return WINDOW if level == 1 else TECH


def apply_flow(
    notes: list[Note],
    *,
    bpm: float,
    onset_env: np.ndarray,
    onset_hop_s: float,
    level: int,
    seed: int,
    mask_cfg: PlayabilityConfig,
    repeats: BarRepeats | None = None,
) -> tuple[list[Note], FlowReport]:
    """Flow level 0-3: repair loose dots, lift flat up cuts, snake runs, stacks, then replay.

    Level 1 uses the window stack profile (a few standalone dots after rests, then same-time
    arrow stacks); levels 2-3 use the tech profile (dot-tail stacks).
    """
    if level not in (0, 1, 2, 3):
        raise ValueError(f"flow level must be 0, 1, 2, or 3, got {level}")
    if level == 0:
        return list(notes), FlowReport(level=0)

    rng = random.Random(seed)
    base_hard = _vcfg_hard(sorted(notes, key=_order), mask_cfg)
    profile = stack_profile(level)
    repaired_notes, dots_repaired, repair_reverted = repair_loose_dots(
        notes, mask_cfg=mask_cfg, base_hard=base_hard,
        bpm=bpm if profile == WINDOW else None,
    )
    lifted_notes, ups_lifted, lift_reverted = lift_flat_ups(
        repaired_notes, mask_cfg=mask_cfg, rng=rng, base_hard=base_hard,
    )
    after_snakes, runs_found, runs_rewritten, notes_rewritten, snake_reverted = apply_snakes(
        lifted_notes, share=RUN_SHARE[level], rng=rng, mask_cfg=mask_cfg, base_hard=base_hard,
    )
    free_dots = free_reverted = 0
    if profile == WINDOW:
        after_snakes, free_dots, free_reverted = add_free_dots(
            after_snakes, bpm=bpm, rng=rng, mask_cfg=mask_cfg, base_hard=base_hard,
        )
    after_stacks, stacks, dots_added, stack_reverted = add_dot_stacks(
        after_snakes, bpm=bpm, onset_env=onset_env, onset_hop_s=onset_hop_s,
        chance=STACK_CHANCE[level], rng=rng, mask_cfg=mask_cfg, base_hard=base_hard,
        profile=profile,
    )
    bars_replayed = 0
    if repeats is not None:
        from bsmaker.phrase import replay_patterns

        added = _added_dot_tails(after_snakes, after_stacks) if profile == TECH else frozenset()
        after_stacks, bars_replayed = replay_patterns(
            after_stacks,
            repeats=repeats,
            bpm=bpm,
            mask_cfg=mask_cfg,
            rng=rng,
            added=added,
            base_hard=base_hard,
        )
    report = FlowReport(
        level=level,
        runs_found=runs_found,
        runs_rewritten=runs_rewritten,
        notes_rewritten=notes_rewritten,
        dot_stacks=stacks if profile == TECH else 0,
        dots_added=dots_added,
        dots_repaired=dots_repaired,
        reverted=repair_reverted + lift_reverted + snake_reverted + free_reverted + stack_reverted,
        ups_lifted=ups_lifted,
        arrow_stacks=stacks if profile == WINDOW else 0,
        free_dots=free_dots,
        bars_replayed=bars_replayed,
    )
    return after_stacks, report
