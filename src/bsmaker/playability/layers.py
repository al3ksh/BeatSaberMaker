"""Hard playability rules for layer objects (walls, bombs, chains, arcs).

Every rule takes either a note sequence or a prebuilt `NoteIndex`; lookups by time are O(log n)
once the index exists, so masks can be evaluated per sampled event.
"""
from __future__ import annotations

from bisect import bisect_left, bisect_right
from collections.abc import Sequence

from bsmaker.formats.model import Arc, Chain, Layers, Note, Obstacle
from bsmaker.playability.checker import Violation

WALL_MARGIN_S = 0.1
BOMB_WINDOW_S = 0.4
CHAIN_MIN_GAP_S = 0.25
_EPS = 1e-6
LAYER_RULE_NAMES = ("wall_on_note", "bomb_in_swing", "chain_conflict", "arc_overrun")

# Cut direction vectors (dx, dy); direction 8 (dot) has none.
_DIRECTION_VECTORS: dict[int, tuple[int, int]] = {
    0: (0, 1), 1: (0, -1), 2: (-1, 0), 3: (1, 0),
    4: (-1, 1), 5: (1, 1), 6: (-1, -1), 7: (1, -1),
}

Cell = tuple[int, int]


class NoteIndex:
    """Notes sorted by time, plus per-colour sorted times, for bisect lookups."""

    def __init__(self, notes: Sequence[Note]) -> None:
        self.notes = sorted(notes, key=lambda n: n.time_s)
        self.times = [n.time_s for n in self.notes]
        self.hand_times: dict[int, list[float]] = {}
        for n in self.notes:
            self.hand_times.setdefault(n.color, []).append(n.time_s)

    def window(self, t0: float, t1: float) -> list[Note]:
        """Notes with t0 <= time <= t1 (inclusive, with a small tolerance)."""
        lo = bisect_left(self.times, t0 - _EPS)
        hi = bisect_right(self.times, t1 + _EPS)
        return self.notes[lo:hi]

    def next_time(self, color: int, after_s: float) -> float | None:
        """Time of the colour's first note strictly after `after_s`, or None."""
        ts = self.hand_times.get(color, [])
        i = bisect_right(ts, after_s + _EPS)
        return ts[i] if i < len(ts) else None


def _index(notes: "Sequence[Note] | NoteIndex") -> NoteIndex:
    return notes if isinstance(notes, NoteIndex) else NoteIndex(notes)


def _in_grid(x: int, y: int) -> bool:
    return 0 <= x < 4 and 0 <= y < 3


def wall_cells(ob: Obstacle) -> set[Cell]:
    """Grid cells (x 0-3, y 0-2) covered by the wall; cells outside the grid are dropped."""
    return {(x, y)
            for x in range(ob.x, ob.x + max(ob.w, 0))
            for y in range(ob.y, ob.y + max(ob.h, 0))
            if _in_grid(x, y)}


def wall_ok(ob: Obstacle, notes: "Sequence[Note] | NoteIndex") -> bool:
    """No note in a covered cell from 0.1 s before the wall to 0.1 s after its end."""
    cells = wall_cells(ob)
    if not cells:
        return True
    end = ob.time_s + max(ob.duration_s, 0.0)
    return not any((n.x, n.y) in cells
                   for n in _index(notes).window(ob.time_s - WALL_MARGIN_S, end + WALL_MARGIN_S))


def bomb_forbidden_cells(notes: "Sequence[Note] | NoteIndex", time_s: float) -> set[Cell]:
    """Cells a bomb at `time_s` must avoid: cells of notes within 0.4 s and their neighbours
    along the cut direction, both ways (clipped to the grid; a dot has no neighbours)."""
    out: set[Cell] = set()
    for n in _index(notes).window(time_s - BOMB_WINDOW_S, time_s + BOMB_WINDOW_S):
        out.add((n.x, n.y))
        vec = _DIRECTION_VECTORS.get(n.direction)
        if vec is not None:
            for s in (1, -1):
                c = (n.x + s * vec[0], n.y + s * vec[1])
                if _in_grid(*c):
                    out.add(c)
    return {c for c in out if _in_grid(*c)}


def bomb_mask_ok(mask: int, notes: "Sequence[Note] | NoteIndex", time_s: float) -> bool:
    """True if no set bit (bit y * 4 + x) of the 12-bit mask is a forbidden cell."""
    return not any(mask >> (y * 4 + x) & 1 for x, y in bomb_forbidden_cells(notes, time_s))


def _line_cells(x0: int, y0: int, x1: int, y1: int) -> set[Cell]:
    steps = max(abs(x1 - x0), abs(y1 - y0))
    if steps == 0:
        return {(x0, y0)}
    return {(round(x0 + (x1 - x0) * i / steps), round(y0 + (y1 - y0) * i / steps))
            for i in range(steps + 1)}


def chain_ok(chain: Chain, notes: "Sequence[Note] | NoteIndex") -> bool:
    """Same hand's next note >= 0.25 s after the chain head, and the head-to-tail cells hold no
    other note (either hand) from the head until that next note (exclusive)."""
    idx = _index(notes)
    nxt = idx.next_time(chain.color, chain.time_s)
    if nxt is not None and nxt - chain.time_s < CHAIN_MIN_GAP_S - _EPS:
        return False
    cells = {c for c in _line_cells(chain.x, chain.y, chain.tx, chain.ty) if _in_grid(*c)}
    end = nxt if nxt is not None else float("inf")
    hi = len(idx.times) if nxt is None else bisect_left(idx.times, nxt - _EPS)
    for n in idx.notes[bisect_left(idx.times, chain.time_s - _EPS):hi]:
        if (n.x, n.y) not in cells or n.time_s >= end - _EPS:
            continue
        own_head = (abs(n.time_s - chain.time_s) <= _EPS and n.color == chain.color
                    and (n.x, n.y) == (chain.x, chain.y))
        if not own_head:
            return False
    return True


def arc_ok(arc: Arc, notes: "Sequence[Note] | NoteIndex") -> bool:
    """An arc never ends after the same hand's next note."""
    nxt = _index(notes).next_time(arc.color, arc.time_s)
    return nxt is None or arc.tail_time_s <= nxt + _EPS


def check_layers(notes: Sequence[Note], layers: Layers) -> list[Violation]:
    """All layer violations, sorted by time. `hand` is the colour for arcs and chains, -1 else."""
    idx = NoteIndex(notes)
    out: list[Violation] = []
    for w in layers.obstacles:
        if not wall_ok(w, idx):
            out.append(Violation(w.time_s, -1, "wall_on_note"))
    for b in layers.bombs:
        if _in_grid(b.x, b.y) and not bomb_mask_ok(1 << (b.y * 4 + b.x), idx, b.time_s):
            out.append(Violation(b.time_s, -1, "bomb_in_swing"))
    for c in layers.chains:
        if not chain_ok(c, idx):
            out.append(Violation(c.time_s, c.color, "chain_conflict"))
    for a in layers.arcs:
        if not arc_ok(a, idx):
            out.append(Violation(a.time_s, a.color, "arc_overrun"))
    out.sort(key=lambda v: v.time_s)
    return out
