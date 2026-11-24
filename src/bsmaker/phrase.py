"""Musical memory: onset-grid phrase reuse and direction-free hand-presence links."""
from __future__ import annotations

import math
from collections.abc import Iterable

import numpy as np
import torch

from bsmaker.audio.bar_similarity import BarRepeats
from bsmaker.audio.beats import SNAP_MAX_SHIFT_S
from bsmaker.audio.mel import FRAMES_PER_SECOND
from bsmaker.formats.tokens import NONE_TOKEN
from bsmaker.models.timing.decode import TimingCurve

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
