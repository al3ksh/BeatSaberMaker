"""Musical memory: onset-grid phrase reuse, hand-presence links, and post-flow pattern replay."""
from __future__ import annotations

import bisect
import math
import random
from collections.abc import Collection, Iterable, Sequence
from dataclasses import replace

import numpy as np
import torch

from bsmaker.audio.bar_similarity import BarRepeats
from bsmaker.audio.beats import SNAP_MAX_SHIFT_S
from bsmaker.audio.mel import FRAMES_PER_SECOND
from bsmaker.formats.model import Note
from bsmaker.formats.tokens import NONE_TOKEN, decode_token, encode_token
from bsmaker.models.timing.decode import TimingCurve
from bsmaker.playability.checker import Violation
from bsmaker.playability.config import PlayabilityConfig
from bsmaker.playability.geometry import angle_diff, direction_angle

RESNAP_TOL_FRAMES = round(SNAP_MAX_SHIFT_S * FRAMES_PER_SECOND)
# Support threshold on the weighted timing curve (same ~0.5x pick threshold as F7 / patterns.txt F7).
RESNAP_FRACTION = 0.45
# Soft grace for at most 1 weaker offset in a dense bar (>= 6 events).
RESNAP_SOFT_FRACTION = 0.25
RESNAP_SOFT_MIN_EVENTS = 6
# Minimum bar events to copy or overwrite: pat_analyze_f1.py:416 only counts bars with >= 3 notes.
MIN_BAR_EVENTS = 3
# Keep copied bar density close to target bar's own event count (guards NPS & density CV).
RESNAP_MAX_COUNT_CHANGE = 0.35
# Minimum gap to adjacent bar's events: 3 mel frames, matching decode.py pick_events min_gap_frames=3.
MIN_GAP_S = 3.0 / FRAMES_PER_SECOND
# Offset equality tolerance for linking bars whose rhythms already/now match (2 ms).
OFFSET_TOL_S = 0.002

# Logit nudge on NONE_TOKEN only so repeating bars echo L/R/double presence without distorting any
# (x, y, dir) distribution (ref with-hand rhythm repeat 0.316 vs ours 0.033; different pattern 0.872;
# Eval #1 at 2.0 gave 0.093 hand repeat, tuned to 4.5 -> ~0.21-0.24).
HAND_PRESENCE_BONUS = 4.5
# Cap share of active bars with hand-presence links so with-hand repeat approaches ~0.25-0.28 without
# exceeding reference median 0.316.
MAX_HAND_LINK_SHARE = 0.36
# Never encourage a double when already in a run of >= 2 consecutive doubles (protects double rate).
HAND_LINK_MAX_DOUBLE_RUN = 2

# Target share of active bars (>= 3 notes) replayed verbatim or mirrored.
# Reference bar_pattern_reuse_share median 0.047 [IQR 0.015-0.169]; budget slightly above median
# to account for ~15-20% boundary-parity rejections, hard-capped so loop songs never overshoot.
PATTERN_REPLAY_TARGET_SHARE = 0.055
PATTERN_REPLAY_MAX_SHARE = 0.070
# Share of replayed bars that prefer left-right mirror (x -> 3-x, color -> 1-color, mirrored dir).
# Reference audio-repeat mirror match 0.071 vs exact match 0.066 (~52% mirror).
MIRROR_SHARE = 0.45
# Never replay more than 2 consecutive bars (creates memorable 1-2 bar hooks, never copy-paste loops).
MAX_CONSECUTIVE_REPLAY_BARS = 2
# A replayed bar never becomes a replay source (ref pattern reuse 0.047).
MAX_REPLAY_DEPTH = 1

# Left-right direction mirror: U/D/dot unchanged; L<->R, UL<->UR, DL<->DR swapped.
_MIRROR_DIR: tuple[int, ...] = (0, 1, 3, 2, 5, 4, 7, 6, 8)


def _time_key(time_s: float) -> int:
    return round(float(time_s) * 1_000_000)


class HandRhythmLinks:
    """Direction-free hand-presence memory that adjusts only `logits[NONE_TOKEN]`."""

    def __init__(
        self,
        pairs: Iterable[tuple[float, float]] = (),
        bonus: float = HAND_PRESENCE_BONUS,
        max_double_run: int = HAND_LINK_MAX_DOUBLE_RUN,
    ) -> None:
        self.bonus = float(bonus)
        self.max_double_run = int(max_double_run)
        self._links: dict[int, int] = {_time_key(t_tgt): _time_key(t_src) for t_tgt, t_src in pairs}
        self._history: dict[int, tuple[bool, bool]] = {}

    def __len__(self) -> int:
        return len(self._links)

    def record(self, time_s: float, has_left: bool, has_right: bool) -> None:
        """Record which hand(s) played at kept event `time_s`."""
        self._history[_time_key(time_s)] = (bool(has_left), bool(has_right))

    def lookup(self, time_s: float) -> tuple[bool, bool] | None:
        """Return `(src_has_left, src_has_right)` for a linked target event, or `None`."""
        src_key = self._links.get(_time_key(time_s))
        if src_key is None:
            return None
        return self._history.get(src_key)

    def bias_left(self, logits: torch.Tensor, time_s: float, mask_l: np.ndarray) -> torch.Tensor:
        """Bias left-hand `NONE_TOKEN` toward matching the linked source bar's left-hand presence."""
        src = self.lookup(time_s)
        if src is None:
            return logits
        src_has_l, _ = src
        if not src_has_l and bool(mask_l[NONE_TOKEN]):
            out = logits.clone()
            out[NONE_TOKEN] += self.bonus
            return out
        if src_has_l and bool(mask_l[1:].any()):
            out = logits.clone()
            out[NONE_TOKEN] -= self.bonus
            return out
        return logits

    def bias_right(
        self,
        logits: torch.Tensor,
        time_s: float,
        mask_r: np.ndarray,
        left: int,
        run: int,
    ) -> torch.Tensor:
        """Bias right-hand `NONE_TOKEN` toward matching the linked source bar's right-hand presence."""
        src = self.lookup(time_s)
        if src is None:
            return logits
        _, src_has_r = src
        if not src_has_r and left != NONE_TOKEN and bool(mask_r[NONE_TOKEN]):
            out = logits.clone()
            out[NONE_TOKEN] += self.bonus
            return out
        if src_has_r and bool(mask_r[1:].any()) and (left == NONE_TOKEN or run < self.max_double_run):
            out = logits.clone()
            out[NONE_TOKEN] -= self.bonus
            return out
        return logits


def _cand_supported(cand: np.ndarray, curve: np.ndarray, thr: float) -> bool:
    hard_thr = RESNAP_FRACTION * thr
    soft_thr = RESNAP_SOFT_FRACTION * thr
    weak = 0
    n_curve = len(curve)
    for t in cand.tolist():
        f = int(round(t * FRAMES_PER_SECOND))
        if f < 0 or f >= n_curve:
            return False
        lo = max(0, f - RESNAP_TOL_FRAMES)
        hi = min(n_curve, f + RESNAP_TOL_FRAMES + 1)
        sup = float(curve[lo:hi].max()) if hi > lo else 0.0
        if sup < soft_thr:
            return False
        if sup < hard_thr:
            weak += 1
            if weak > 1 or len(cand) < RESNAP_SOFT_MIN_EVENTS:
                return False
    return True


def repeat_rhythm(
    planned: np.ndarray,
    timing_curve: TimingCurve | None,
    repeats: BarRepeats,
) -> np.ndarray:
    """All-or-nothing onset-grid phrase reuse on map bars `[k*bar_s, (k+1)*bar_s)`."""
    working = np.sort(np.asarray(planned, dtype=np.float64)).copy()
    if (
        len(working) == 0
        or timing_curve is None
        or not np.isfinite(repeats.bar_s)
        or repeats.bar_s <= 0
        or len(repeats.source) == 0
    ):
        return working

    curve = np.asarray(timing_curve.curve, dtype=np.float32)
    thr = float(timing_curve.threshold)
    if len(curve) == 0 or not np.isfinite(thr) or thr <= 0.0:
        return working

    bar_s = float(repeats.bar_s)
    bar_hi_eps = np.nextafter(bar_s, -np.inf)
    for j in range(len(repeats.source)):
        i = int(repeats.source[j])
        if i < 0 or i >= j:
            continue
        i0 = int(np.searchsorted(working, i * bar_s, side="left"))
        i1 = int(np.searchsorted(working, (i + 1) * bar_s, side="left"))
        a = working[i0:i1]

        j0 = int(np.searchsorted(working, j * bar_s, side="left"))
        j1 = int(np.searchsorted(working, (j + 1) * bar_s, side="left"))
        b = working[j0:j1]

        if len(a) < MIN_BAR_EVENTS or len(b) < MIN_BAR_EVENTS:
            continue
        if abs(len(a) - len(b)) > max(1.0, RESNAP_MAX_COUNT_CHANGE * len(b)):
            continue

        cand = np.clip(a + (j - i) * bar_s, j * bar_s, j * bar_s + bar_hi_eps)
        prev_t = float(working[j0 - 1]) if j0 > 0 else -np.inf
        next_t = float(working[j1]) if j1 < len(working) else np.inf
        if cand[0] - prev_t < MIN_GAP_S or next_t - cand[-1] < MIN_GAP_S:
            continue
        if not _cand_supported(cand, curve, thr):
            continue

        working = np.concatenate([working[:j0], cand, working[j1:]])
    return working


def _group_by_bar(times_s: np.ndarray, bar_s: float) -> dict[int, np.ndarray]:
    bars: dict[int, np.ndarray] = {}
    if len(times_s) == 0:
        return bars
    bar_idx = np.floor(times_s / bar_s).astype(np.int64)
    uniq, starts, counts = np.unique(bar_idx, return_index=True, return_counts=True)
    for k, s, c in zip(uniq.tolist(), starts.tolist(), counts.tolist()):
        if k >= 0:
            bars[k] = times_s[s:s + c]
    return bars


def plan_rhythm(
    planned: np.ndarray,
    timing_curve: TimingCurve | None,
    repeats: BarRepeats | None,
) -> tuple[np.ndarray, HandRhythmLinks | None]:
    """Reuse onset rhythms on matching bars and build direction-free hand-presence links."""
    if repeats is None or not np.isfinite(repeats.bar_s) or repeats.bar_s <= 0 or len(repeats.source) == 0:
        return np.sort(np.asarray(planned, dtype=np.float64)).copy(), None

    updated = repeat_rhythm(planned, timing_curve, repeats)
    bar_s = float(repeats.bar_s)
    bar_events = _group_by_bar(updated, bar_s)
    n_active = sum(1 for ev in bar_events.values() if len(ev) >= MIN_BAR_EVENTS)
    max_linked_bars = int(math.floor(MAX_HAND_LINK_SHARE * n_active))
    if max_linked_bars <= 0:
        return updated, None

    pairs: list[tuple[float, float]] = []
    linked_bars = 0
    for j in range(len(repeats.source)):
        if linked_bars >= max_linked_bars:
            break
        i = int(repeats.source[j])
        if i < 0 or i >= j:
            continue
        a = bar_events.get(i)
        b = bar_events.get(j)
        if a is None or b is None or len(a) < MIN_BAR_EVENTS or len(b) != len(a):
            continue
        if not np.all(np.abs((b - j * bar_s) - (a - i * bar_s)) <= OFFSET_TOL_S):
            continue
        pairs.extend((float(tb), float(ta)) for tb, ta in zip(b.tolist(), a.tolist()))
        linked_bars += 1

    return updated, (HandRhythmLinks(pairs) if pairs else None)


def mirror_token(token: int) -> int:
    """Horizontally mirror a single note token (`x -> 3-x`, `dir -> _MIRROR_DIR[dir]`)."""
    if token == NONE_TOKEN:
        return NONE_TOKEN
    x, y, direction = decode_token(token)
    return encode_token(3 - x, y, _MIRROR_DIR[direction])


def mirror_note(note: Note) -> Note:
    """Horizontally mirror a `Note` (`color -> 1-color`, `x -> 3-x`, `dir -> _MIRROR_DIR[dir]`)."""
    return replace(
        note,
        color=1 - note.color,
        x=3 - note.x,
        direction=_MIRROR_DIR[note.direction],
    )


def _non_slider_swings(
    notes: Sequence[Note],
    added: Collection[tuple[float, int]],
) -> dict[int, list[Note]]:
    swings: dict[int, list[Note]] = {0: [], 1: []}
    for n in notes:
        if (n.time_s, n.color) in added:
            continue
        seq = swings[n.color]
        if not seq or n.time_s - seq[-1].time_s > 1e-9:
            seq.append(n)
    return swings


def _bar_onset_offsets(
    bar_notes: Sequence[Note],
    bar_start_s: float,
    added: Collection[tuple[float, int]],
) -> list[float]:
    offsets = sorted(
        n.time_s - bar_start_s
        for n in bar_notes
        if (n.time_s, n.color) not in added
    )
    out: list[float] = []
    for off in offsets:
        if not out or off - out[-1] > OFFSET_TOL_S:
            out.append(off)
    return out


def _loose_dot_mask(
    working: Sequence[Note],
    *,
    bpm: float,
    mask_cfg: PlayabilityConfig,
    added: Collection[tuple[float, int]],
) -> list[bool]:
    from bsmaker import flow_rules

    prev, nxt = flow_rules._same_hand_neighbours(working)
    spots = flow_rules._rest_dot_spots(working, prev, nxt, bpm=bpm, mask_cfg=mask_cfg)
    loose = [False] * len(working)
    line: dict[int, tuple[int, int, int, int, float] | None] = {0: None, 1: None}
    for idx, n in enumerate(working):
        h = n.color
        if n.direction != 8:
            line[h] = (n.x, n.y, *flow_rules._DIRS[n.direction], n.time_s)
            continue
        cur = line[h]
        if (n.time_s, h) in added or (cur is not None and flow_rules._on_ray(cur, n)):
            if cur is not None:
                line[h] = (cur[0], cur[1], cur[2], cur[3], n.time_s)
            continue
        line[h] = None
        p, q = prev[idx], nxt[idx]
        valid_rest = (
            spots[idx]
            and p is not None
            and working[p].direction != 8
            and (q is None or working[q].direction != 8)
        )
        valid_first = (
            p is None
            and (
                q is None
                or (
                    working[q].time_s - n.time_s >= flow_rules._MIN_SAME_HAND_GAP_S - 1e-9
                    and working[q].direction != 8
                )
            )
        )
        if not (valid_rest or valid_first):
            loose[idx] = True
    return loose


def _source_bar_clean(
    working: Sequence[Note],
    lo_idx: int,
    hi_idx: int,
    *,
    is_loose: Sequence[bool],
    added: Collection[tuple[float, int]],
) -> bool:
    if any(is_loose[idx] for idx in range(lo_idx, hi_idx)):
        return False
    swings = _non_slider_swings(working[lo_idx:hi_idx], added)
    for seq in swings.values():
        ang: float | None = None
        for idx, n in enumerate(seq):
            if idx > 0:
                p = seq[idx - 1]
                if {p.direction, n.direction} == {0, 1} and p.y == 0 and n.y == 0:
                    return False
            if n.direction == 8:
                if ang is not None:
                    ang = (ang + 180.0) % 360.0
                continue
            new_ang = direction_angle(n.direction)
            if ang is not None and new_ang is not None and angle_diff(new_ang, ang) < 90.0:
                return False
            ang = new_ang
    return True


def _advance_angle(ang: float | None, seq: Sequence[Note]) -> float | None:
    for n in seq:
        if n.direction == 8:
            if ang is not None:
                ang = (ang + 180.0) % 360.0
        else:
            ang = direction_angle(n.direction)
    return ang


def _first_arrow_resets(ang: float | None, seq: Sequence[Note]) -> bool:
    for n in seq:
        if n.direction == 8:
            if ang is not None:
                ang = (ang + 180.0) % 360.0
            continue
        new_ang = direction_angle(n.direction)
        return ang is not None and new_ang is not None and angle_diff(new_ang, ang) < 90.0
    return False


def _is_flat_row0_flip(a: Note, b: Note) -> bool:
    return {a.direction, b.direction} == {0, 1} and a.y == 0 and b.y == 0


def _boundary_issues(
    before_h: Sequence[Note],
    mid_h: Sequence[Note],
    after_h: Sequence[Note],
) -> tuple[bool, bool, bool, bool]:
    ang_before = _advance_angle(None, before_h)
    if not mid_h:
        if before_h and after_h:
            flip = _is_flat_row0_flip(before_h[-1], after_h[0])
            reset = _first_arrow_resets(ang_before, after_h)
            return flip, reset, flip, reset
        return False, False, False, False

    entry_flip = _is_flat_row0_flip(before_h[-1], mid_h[0]) if before_h else False
    entry_reset = _first_arrow_resets(ang_before, mid_h) if before_h else False
    ang_mid = _advance_angle(ang_before, mid_h)
    exit_flip = _is_flat_row0_flip(mid_h[-1], after_h[0]) if after_h else False
    exit_reset = _first_arrow_resets(ang_mid, after_h) if after_h else False
    return entry_flip, entry_reset, exit_flip, exit_reset


def _boundary_transitions_ok(
    before: Sequence[Note],
    orig_mid: Sequence[Note],
    cand_mid: Sequence[Note],
    after: Sequence[Note],
    *,
    bpm: float,
    mask_cfg: PlayabilityConfig,
    orig_added: Collection[tuple[float, int]],
    trial_added: Collection[tuple[float, int]],
) -> bool:
    from bsmaker import flow_rules

    rest_need = max(flow_rules._FREE_DOT_REST_BEATS * 60.0 / bpm, mask_cfg.slow_gap_s)
    before_swings = _non_slider_swings(before, orig_added)
    orig_swings = _non_slider_swings(orig_mid, orig_added)
    cand_swings = _non_slider_swings(cand_mid, trial_added)
    after_swings = _non_slider_swings(after, orig_added)

    before_any: dict[int, Note | None] = {0: None, 1: None}
    for n in before:
        before_any[n.color] = n
    after_any: dict[int, Note | None] = {0: None, 1: None}
    for n in after:
        if after_any[n.color] is None:
            after_any[n.color] = n

    for h in (0, 1):
        c_seq = cand_swings[h]
        if c_seq and c_seq[0].direction == 8:
            p_any = before_any[h]
            if p_any is not None:
                if c_seq[0].time_s - p_any.time_s < rest_need - 1e-9 or p_any.direction == 8:
                    return False
        if c_seq and c_seq[-1].direction == 8:
            q_any = after_any[h]
            if q_any is not None and q_any.direction == 8:
                return False

        orig_issues = _boundary_issues(before_swings[h], orig_swings[h], after_swings[h])
        cand_issues = _boundary_issues(before_swings[h], c_seq, after_swings[h])
        if any(c and not o for c, o in zip(cand_issues, orig_issues)):
            return False
    return True


def _consecutive_run_exceeds(depth: dict[int, int], bar_idx: int) -> bool:
    run = 1
    k = bar_idx - 1
    while depth.get(k, 0) > 0:
        run += 1
        k -= 1
    k = bar_idx + 1
    while depth.get(k, 0) > 0:
        run += 1
        k += 1
    return run > MAX_CONSECUTIVE_REPLAY_BARS


def _bar_index(time_s: float, bar_s: float) -> int:
    """Return `k` matching `k * bar_s <= time_s < (k + 1) * bar_s` under float64."""
    k = int(math.floor(time_s / bar_s))
    while k > 0 and time_s < k * bar_s:
        k -= 1
    while time_s >= (k + 1) * bar_s:
        k += 1
    return k


def _cross_bar_tech_bars(
    working: Sequence[Note],
    bar_s: float,
    orig_added: Collection[tuple[float, int]],
) -> tuple[set[int], set[int]]:
    """Return `(child_bars, owner_bars)` for cross-bar TECH dot-tail sliders."""
    from bsmaker import flow_rules

    child_bars: set[int] = set()
    owner_bars: set[int] = set()
    if not orig_added:
        return child_bars, owner_bars
    last_kept: dict[int, tuple[int, float]] = {}
    for n in working:
        k = _bar_index(n.time_s, bar_s)
        h = n.color
        if flow_rules._key(n) in orig_added:
            prev = last_kept.get(h)
            if (
                prev is None
                or n.time_s - prev[1] >= flow_rules._SLIDER_VALIDATION_GAP_S - flow_rules._EPS
                or prev[0] != k
            ):
                if k >= 0:
                    child_bars.add(k)
                if prev is not None and prev[0] >= 0 and prev[0] != k:
                    owner_bars.add(prev[0])
        else:
            last_kept[h] = (k, n.time_s)
    return child_bars, owner_bars


def _valid_added_dot_tails(
    notes: Sequence[Note],
    cand_added: Collection[tuple[float, int]],
) -> frozenset[tuple[float, int]]:
    """Keep only `cand_added` dots < 0.12 s after a same-hand kept note (matches `_added_dot_tails`)."""
    from bsmaker import flow_rules

    if not cand_added:
        return frozenset()
    cand_set = frozenset(cand_added)
    valid: set[tuple[float, int]] = set()
    last_t: dict[int, float] = {}
    for n in notes:
        k = flow_rules._key(n)
        prev_t = last_t.get(n.color)
        if (
            n.direction == 8
            and k in cand_set
            and prev_t is not None
            and n.time_s - prev_t < flow_rules._SLIDER_VALIDATION_GAP_S - flow_rules._EPS
        ):
            valid.add(k)
        else:
            last_t[n.color] = n.time_s
    return frozenset(valid)


def replay_patterns(
    notes: Sequence[Note],
    *,
    repeats: BarRepeats | None,
    bpm: float,
    mask_cfg: PlayabilityConfig,
    rng: random.Random | None = None,
    seed: int = 0,
    added: Collection[tuple[float, int]] = frozenset(),
    base_hard: Collection[Violation] | None = None,
) -> tuple[list[Note], int]:
    """Post-flow atomic, `window_ok`-validated exact/mirrored bar pattern replay."""
    from bsmaker import flow_rules

    working = sorted(notes, key=flow_rules._order)
    if (
        not working
        or repeats is None
        or not np.isfinite(repeats.bar_s)
        or repeats.bar_s <= 0.0
        or len(repeats.source) == 0
        or not math.isfinite(bpm)
        or bpm <= 0.0
    ):
        return working, 0

    bar_s = float(repeats.bar_s)
    bar_bounds: dict[int, tuple[int, int]] = {}
    for idx, n in enumerate(working):
        k = _bar_index(n.time_s, bar_s)
        if k >= 0:
            lo_i, _ = bar_bounds.get(k, (idx, idx + 1))
            bar_bounds[k] = (lo_i, idx + 1)

    n_active = sum(1 for lo_i, hi_i in bar_bounds.values() if hi_i - lo_i >= MIN_BAR_EVENTS)
    budget = min(
        round(PATTERN_REPLAY_TARGET_SHARE * n_active),
        math.floor(PATTERN_REPLAY_MAX_SHARE * n_active),
    )
    if budget <= 0:
        return working, 0

    orig_added = frozenset(added)
    cross_bar_children, cross_bar_owners = _cross_bar_tech_bars(working, bar_s, orig_added)
    is_loose = _loose_dot_mask(working, bpm=bpm, mask_cfg=mask_cfg, added=orig_added)
    clean_cache: dict[int, bool] = {}
    orig_bars: dict[int, list[Note]] = {
        k: working[lo_i:hi_i] for k, (lo_i, hi_i) in bar_bounds.items()
    }

    cands: list[tuple[float, int, int]] = []
    n_src = len(repeats.source)
    for j in range(n_src):
        if j in cross_bar_owners:
            continue
        i = int(repeats.source[j])
        if i < 0 or i >= j or i in cross_bar_children:
            continue
        src_notes = orig_bars.get(i)
        tgt_notes = orig_bars.get(j)
        if src_notes is None or tgt_notes is None:
            continue
        if len(src_notes) < MIN_BAR_EVENTS or len(tgt_notes) < MIN_BAR_EVENTS:
            continue
        off_i = _bar_onset_offsets(src_notes, i * bar_s, orig_added)
        off_j = _bar_onset_offsets(tgt_notes, j * bar_s, orig_added)
        if len(off_i) != len(off_j) or any(abs(a - b) > OFFSET_TOL_S for a, b in zip(off_i, off_j)):
            continue
        if i not in clean_cache:
            lo_i, hi_i = bar_bounds[i]
            clean_cache[i] = _source_bar_clean(
                working, lo_i, hi_i, is_loose=is_loose, added=orig_added,
            )
        if not clean_cache[i]:
            continue
        sim_val = float(repeats.sim[j, i]) if j < len(repeats.sim) and i < repeats.sim.shape[1] else 0.0
        ps = int(repeats.phrase_start[j]) if j < len(repeats.phrase_start) else -1
        bonus = 0.02 if ps >= 0 and j in (ps, ps + 1) else 0.0
        cands.append((sim_val + bonus, j, i))

    if not cands:
        return working, 0

    cands.sort(key=lambda item: (-item[0], item[1], item[2]))
    if base_hard is None:
        base_hard = flow_rules._vcfg_hard(working, mask_cfg)
    if rng is None:
        rng = random.Random(seed)

    working_added = orig_added
    depth: dict[int, int] = {}
    used_as_source: set[int] = set()
    replayed = 0
    for _, j, i in cands:
        if replayed >= budget:
            break
        if depth.get(i, 0) >= MAX_REPLAY_DEPTH or depth.get(j, 0) > 0 or j in used_as_source:
            continue
        if _consecutive_run_exceeds(depth, j):
            continue

        j_lo = j * bar_s
        j_hi = (j + 1) * bar_s
        j_hi_eps = math.nextafter(j_hi, -math.inf)
        a = bisect.bisect_left(working, j_lo, key=flow_rules._time)
        b = bisect.bisect_left(working, j_hi, key=flow_rules._time)
        tgt_slice = working[a:b]
        if not tgt_slice:
            continue

        src_notes = orig_bars[i]
        dt = (j - i) * bar_s
        dbeat = 4.0 * (j - i)
        bar_j_keys = {flow_rules._key(n) for n in tgt_slice}
        prefer_mirror = rng.random() < MIRROR_SHARE

        for mirror in (prefer_mirror, not prefer_mirror):
            cand: list[Note] = []
            cand_added: set[tuple[float, int]] = set()
            for n in src_notes:
                t_new = min(max(n.time_s + dt, j_lo), j_hi_eps)
                shifted = replace(n, time_s=t_new, beat=n.beat + dbeat)
                out_n = mirror_note(shifted) if mirror else shifted
                cand.append(out_n)
                if flow_rules._key(n) in orig_added:
                    cand_added.add(flow_rules._key(out_n))
            cand.sort(key=flow_rules._order)
            trial = working[:a] + cand + working[b:]
            trial_added = _valid_added_dot_tails(trial, (working_added - bar_j_keys) | cand_added)

            if not _boundary_transitions_ok(
                working[:a], tgt_slice, cand, working[b:],
                bpm=bpm, mask_cfg=mask_cfg, orig_added=working_added, trial_added=trial_added,
            ):
                continue

            t0 = min(cand[0].time_s, tgt_slice[0].time_s)
            t1 = max(cand[-1].time_s, tgt_slice[-1].time_s)
            if flow_rules.window_ok(trial, t0, t1, mask_cfg, added=trial_added, base_hard=base_hard):
                working = trial
                working_added = trial_added
                depth[j] = 1
                used_as_source.add(i)
                replayed += 1
                break

    return working, replayed
