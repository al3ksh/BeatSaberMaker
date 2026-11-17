"""Rule-based gameplay obstacles: rare, gentle side-step and duck walls.

Candidates are found deterministically with sorted times, prefix counts and bisect (O(n log n)); the
seed only picks among them. A sequence is one wall. Consecutive side-steps alternate sides when the
candidates allow it (ducks do not change the alternation).
"""
from __future__ import annotations

import random
from bisect import bisect_left, bisect_right
from collections.abc import Sequence
from itertools import accumulate

from bsmaker.formats.model import Arc, Chain, Note, Obstacle
from bsmaker.playability.layers import NoteIndex, ornament_points, wall_ok, wall_on_ornament

START_MARGIN_S = 10.0
END_MARGIN_S = 5.0
SIDE_WINDOW_S = 0.2
_SPACING_S = {2: 45.0, 3: 30.0}


def _side_candidates(times: list[float], xs: list[int], beat_s: float, lo: float, hi: float
                     ) -> tuple[list[tuple[float, int]], list[tuple[float, int]]]:
    """(start, beats) candidates for a wall on the left columns / on the right columns."""
    left_prefix = [0, *accumulate(1 if x <= 1 else 0 for x in xs)]
    left: list[tuple[float, int]] = []
    right: list[tuple[float, int]] = []
    seen: set[tuple[float, int]] = set()
    for s in times:
        for beats in (1, 2):
            if (s, beats) in seen:
                continue
            seen.add((s, beats))
            end = s + beats * beat_s
            if s < lo or end > hi:
                continue
            a = bisect_left(times, s - SIDE_WINDOW_S)
            b = bisect_right(times, end + SIDE_WINDOW_S)
            n_left = left_prefix[b] - left_prefix[a]
            n_right = (b - a) - n_left
            if n_left == 0 and n_right >= 2:
                left.append((s, beats))  # wall on columns 0-1, every note in columns 2-3
            elif n_right == 0 and n_left >= 2:
                right.append((s, beats))  # wall on columns 2-3, every note in columns 0-1
    return left, right


def _duck_candidates(times: list[float], beat_s: float, duration_s: float, lo: float, hi: float
                     ) -> list[tuple[float, int]]:
    ducks: list[tuple[float, int]] = []
    bounds = [0.0, *times, duration_s]
    for g0, g1 in zip(bounds, bounds[1:]):
        if g1 - g0 >= 2 * beat_s:
            s = (g0 + g1) / 2 - beat_s / 2
            if s >= lo and s + beat_s <= hi:
                ducks.append((s, 1))
    return ducks


def gameplay_obstacles(notes: Sequence[Note], bpm: float, duration_s: float, level: int,
                       seed: int, arcs: Sequence[Arc] = (), chains: Sequence[Chain] = ()
                       ) -> list[Obstacle]:
    """At most one wall per 45 s (30 s at level 3), none in the first 10 s or the last 5 s."""
    if level < 2 or bpm <= 0 or duration_s <= 0:
        return []
    spacing = _SPACING_S.get(level, _SPACING_S[3])
    max_n = int(duration_s // spacing)
    lo, hi = START_MARGIN_S, duration_s - END_MARGIN_S
    if max_n < 1 or hi <= lo:
        return []
    beat_s = 60.0 / bpm
    idx = NoteIndex(notes)
    points = ornament_points(arcs=arcs, chains=chains)
    times = idx.times
    xs = [n.x for n in idx.notes]
    left, right = _side_candidates(times, xs, beat_s, lo, hi)
    ducks = _duck_candidates(times, beat_s, duration_s, lo, hi)

    def make(kind: str, s: float, beats: int) -> Obstacle:
        dur = beats * beat_s
        x, w, y, h = {"left": (0, 2, 0, 5), "right": (2, 2, 0, 5), "duck": (0, 4, 2, 3)}[kind]
        return Obstacle(s, s * bpm / 60.0, dur, float(beats), x, y, w, h)

    pools: dict[str, list[tuple[float, int]]] = {"left": [], "right": [], "duck": []}
    for kind, cands in (("left", left), ("right", right), ("duck", ducks)):
        for s, beats in cands:
            wall = make(kind, s, beats)
            if wall_ok(wall, idx) and not wall_on_ornament(wall, points):
                pools[kind].append((s, beats))
        pools[kind].sort()

    rng = random.Random(seed)
    next_side = rng.choice(("left", "right"))
    bound = lo
    out: list[Obstacle] = []

    def pick(kinds: list[str]) -> tuple[str, float, int] | None:
        ranges = []
        for k in kinds:
            starts = pools[k]
            a = bisect_left(starts, (bound, 0))
            ranges.append((k, a, a))
        firsts = [pools[k][a][0] for k, a, _ in ranges if a < len(pools[k])]
        if not firsts:
            return None
        limit = min(firsts) + spacing
        ranges = [(k, a, bisect_left(pools[k], (limit, 0))) for k, a, _ in ranges]
        total = sum(b - a for _, a, b in ranges)
        if total == 0:
            return None
        r = rng.randrange(total)
        for k, a, b in ranges:
            if r < b - a:
                return (k, *pools[k][a + r])
            r -= b - a
        return None

    while len(out) < max_n:
        other = "right" if next_side == "left" else "left"
        chosen = pick([next_side, "duck"]) or pick([other])
        if chosen is None:
            break
        kind, s, beats = chosen
        out.append(make(kind, s, beats))
        if kind != "duck":
            next_side = "right" if kind == "left" else "left"
        bound = s + spacing
    return out
