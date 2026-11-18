"""Phrase repetition: when the music repeats, repeat the rhythm and (sometimes) the pattern.

Reference maps (23 Expert diffs, .superpowers/refmaps/patterns.txt G4) repeat a 2-bar rhythm in
31% of bar pairs (ours 2%) and reuse a whole bar pattern in 4.7% of bars (ours 0); on bars whose
audio repeats, their notes match the earlier bar 3x more often than on random bar pairs (ours
1.35x). Two steps, both driven by `bsmaker.audio.bar_similarity.BarRepeats`:

- `repeat_rhythm` (timing): a bar whose audio repeats an earlier bar takes over the earlier
  bar's beat offsets wherever the timing curve supports them.
- `PatternLinks` (sampling): on such a bar, when the rhythm matches, each event's logits favour
  the token the earlier bar played at the same offset (or its left-right mirror). The
  playability mask is applied afterwards, so nothing unplayable gets in. The flow layer also
  stacks dots on a repeated bar where the earlier bar got them (`PatternLinks.stack_wanted`).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from bsmaker.audio.bar_similarity import BarRepeats
from bsmaker.audio.mel import FRAMES_PER_SECOND
from bsmaker.formats.tokens import NONE_TOKEN, VOCAB_SIZE, decode_token, encode_token

# Timing: a copied offset is kept where the timing curve (locally normalised, as in pick_events)
# within +-RESNAP_TOL_FRAMES reaches this share of the pick threshold (F7 in patterns.txt). The
# tolerance is the snap cap (beats.SNAP_MAX_SHIFT_S = 45 ms, about 4 frames): a curve peak that
# far from a grid point would itself have been snapped onto it.
RESNAP_FRACTION = 0.5
RESNAP_TOL_FRAMES = 4
# A bar is re-snapped only if at least this share of the earlier bar's events survive the curve
# check; otherwise the bar keeps its own rhythm (a half-copied bar is neither a repeat nor the
# timing model's choice).
RESNAP_MIN_KEEP = 0.75
# ... and only if the bar's event count changes by at most this share (at least 1 event): a
# copy chain over a riff would otherwise carry its sparsest bar forward and thin the map.
RESNAP_MAX_COUNT_CHANGE = 0.5
# Bars with fewer events (rests, breaks) are neither copied from nor overwritten.
MIN_BAR_EVENTS = 2
# Events closer than this (s) to a neighbour outside the bar are dropped (3 mel frames, the
# pick_events minimum gap).
MIN_GAP_S = 3 / FRAMES_PER_SECOND
# Events put back after re-snapping keep at least this many beats from every other event.
REFILL_GAP_BEATS = 0.25
# Two bars have the same rhythm when their offsets agree within this many seconds.
OFFSET_TOL_S = 0.002

# Sampling: logit bonus per event for the earlier bar's token. F7 suggested +1.5 to start; at
# 1.5 whole bars were reused in 2% of bars (reference 4.7%), at 3.0 in about 4%.
PATTERN_BONUS = 3.0
# Share of linked bars that copy the left-right mirror instead (reference audio-repeat bars:
# mirror match 0.071, identical 0.066, so mirrors are common; identical stays the majority).
MIRROR_SHARE = 1 / 3
# A bar that copies a pattern is never itself a pattern source, so a riff of similar bars gives
# A A B B ... and not one bar looped (reference bar pattern reuse is only 4.7%).
MAX_LINK_DEPTH = 1

_MIRROR_DIR = {0: 0, 1: 1, 2: 3, 3: 2, 4: 5, 5: 4, 6: 7, 7: 6, 8: 8}


def mirror_token(token: int) -> int:
    """The same note mirrored left-right (x -> 3 - x, direction mirrored); NONE stays NONE."""
    if token == NONE_TOKEN:
        return NONE_TOKEN
    x, y, d = decode_token(token)
    return encode_token(3 - x, y, _MIRROR_DIR[d])


def _bar_slices(times: np.ndarray, bar_s: float, n_bars: int) -> np.ndarray:
    """Start index into sorted `times` of every bar, plus the end: `[n_bars + 1]`."""
    edges = np.arange(n_bars + 1) * bar_s
    return np.searchsorted(times, edges - 1e-9, side="left")


def _curve_ok(curve: np.ndarray, t: float, need: float, fps: float) -> bool:
    f = int(round(t * fps))
    if f < 0 or f >= len(curve):
        return False
    lo, hi = max(0, f - RESNAP_TOL_FRAMES), min(len(curve), f + RESNAP_TOL_FRAMES + 1)
    return float(curve[lo:hi].max()) >= need


def repeat_rhythm(times: np.ndarray, curve: np.ndarray, threshold: float, repeats: BarRepeats,
                  fps: float, touched: set[int] | None = None) -> tuple[np.ndarray, int]:
    """Event times (sorted) with repeated bars re-snapped to their earlier twin's offsets.

    Bars are handled in time order, so a bar copied from a bar that was itself re-snapped gets
    the final rhythm. `curve` is the per-frame timing curve the events were picked from and
    `threshold` its pick threshold. Returns (times, number of re-snapped bars); `touched`, when
    given, receives every re-snapped (or already matching) bar and its source.
    """
    t = np.sort(np.asarray(times, dtype=np.float64))
    bar_s = repeats.bar_s
    n_bars = repeats.n_bars
    need = RESNAP_FRACTION * threshold
    n_done = 0
    for j in range(n_bars):
        i = int(repeats.source[j])
        if i < 0 or i >= j:
            continue
        sl = _bar_slices(t, bar_s, n_bars)
        a = t[sl[i]:sl[i + 1]]
        b = t[sl[j]:sl[j + 1]]
        if len(a) < MIN_BAR_EVENTS or len(b) < MIN_BAR_EVENTS:
            continue
        cand = a + (j - i) * bar_s
        prev_t = t[sl[j] - 1] if sl[j] > 0 else -np.inf
        next_t = t[sl[j + 1]] if sl[j + 1] < len(t) else np.inf
        keep = [c for c in cand
                if c - prev_t >= MIN_GAP_S and next_t - c >= MIN_GAP_S
                and _curve_ok(curve, c, need, fps)]
        if len(keep) < max(MIN_BAR_EVENTS, RESNAP_MIN_KEEP * len(a)):
            continue
        if abs(len(keep) - len(b)) > max(1.0, RESNAP_MAX_COUNT_CHANGE * len(b)):
            continue
        if len(keep) == len(b) and np.allclose(keep, b, atol=1e-6):
            if touched is not None:  # already a repeat: keep it intact as well
                touched.update((i, j))
            continue
        t = np.concatenate([t[:sl[j]], np.asarray(keep), t[sl[j + 1]:]])
        n_done += 1
        if touched is not None:
            touched.update((i, j))
    return t, n_done


def refill(times: np.ndarray, curve: np.ndarray, fps: float, n_missing: int, bar_s: float,
           frozen: set[int], snap) -> np.ndarray:
    """Add up to `n_missing` events back at the strongest curve peaks outside the `frozen` bars.

    Re-snapping thins the map a little (copied offsets the curve does not support are dropped);
    the lost events go back where the timing model most wanted one and no repeat is disturbed.
    A new event is snapped with `snap` (times -> grid times) and kept at least REFILL_GAP_BEATS
    beats (a quarter of a bar's beat) from every event.
    """
    t = np.sort(np.asarray(times, dtype=np.float64))
    if n_missing <= 0 or len(curve) < 3:
        return t
    c = np.asarray(curve, dtype=np.float64)
    peaks = np.nonzero((c[1:-1] >= c[:-2]) & (c[1:-1] >= c[2:]) & (c[1:-1] > 0))[0] + 1
    peaks = peaks[np.argsort(-c[peaks], kind="stable")]
    gap = REFILL_GAP_BEATS * bar_s / 4.0
    added: list[float] = []
    for f in peaks:
        if len(added) >= n_missing:
            break
        ts = float(snap(np.array([f / fps]))[0])
        if int(np.floor(ts / bar_s + 1e-9)) in frozen:
            continue
        k = np.searchsorted(t, ts)
        if (k > 0 and ts - t[k - 1] < gap) or (k < len(t) and t[k] - ts < gap):
            continue
        t = np.insert(t, k, ts)
        added.append(ts)
    return t


@dataclass
class PatternLinks:
    """Per planned event: which earlier event's tokens to favour, and whether mirrored.

    `link[key]` maps an event time key (ms) to (source time key, mirror). The sampler calls
    `bias` before picking each hand's token and `record` after the event is decided.
    """
    link: dict[int, tuple[int, bool]]
    bonus: float = PATTERN_BONUS
    played: dict[int, tuple[int, int]] = field(default_factory=dict)
    n_bars: int = 0
    # Every bar with the earlier bar's rhythm (pattern-linked or not): the flow layer adds a dot
    # stack on it exactly where the earlier bar got one, so the stack dots repeat too.
    rhythm_link: dict[int, tuple[int, bool]] = field(default_factory=dict)
    stacks: dict[tuple[int, int], int] = field(default_factory=dict)

    @staticmethod
    def key(t: float) -> int:
        return int(round(t * 1000.0))

    def _source_tokens(self, t: float) -> tuple[tuple[int, int], bool] | None:
        lk = self.link.get(self.key(t))
        if lk is None:
            return None
        src, mirror = lk
        for k in (src, src - 1, src + 1):
            if k in self.played:
                return self.played[k], mirror
        return None

    def wanted(self, t: float, hand: int) -> int | None:
        """The token this hand should favour at time `t`, or None."""
        found = self._source_tokens(t)
        if found is None:
            return None
        (left, right), mirror = found
        if mirror:
            return mirror_token(right if hand == 0 else left)
        return left if hand == 0 else right

    def bias(self, logits: torch.Tensor, t: float, hand: int) -> torch.Tensor:
        tok = self.wanted(t, hand)
        if tok is None or not 0 <= tok < VOCAB_SIZE or self.bonus == 0.0:
            return logits
        out = logits.clone()
        out[tok] += self.bonus
        return out

    def record(self, t: float, left: int, right: int) -> None:
        self.played[self.key(t)] = (int(left), int(right))

    def stack_wanted(self, t: float, hand: int) -> int | None:
        """Dots the earlier bar stacked after this hand's arrow at `t` (0: none), or None."""
        lk = self.rhythm_link.get(self.key(t))
        if lk is None:
            return None
        src, mirror = lk
        src_hand = 1 - hand if mirror else hand
        return max(self.stacks.get((k, src_hand), 0) for k in (src - 1, src, src + 1))

    def stack_record(self, t: float, hand: int, n_dots: int) -> None:
        if n_dots > 0:
            self.stacks[(self.key(t), int(hand))] = int(n_dots)


def pattern_links(times: np.ndarray, repeats: BarRepeats, seed: int,
                  bonus: float = PATTERN_BONUS) -> PatternLinks:
    """Links for every bar whose audio repeats an earlier bar and whose rhythm matches it.

    A bar is linked only when its event offsets equal the source bar's (within OFFSET_TOL_S) and
    the source is not itself a copy deeper than MAX_LINK_DEPTH. Which linked bars mirror is
    drawn from `seed`.
    """
    t = np.sort(np.asarray(times, dtype=np.float64))
    bar_s = repeats.bar_s
    n_bars = repeats.n_bars
    sl = _bar_slices(t, bar_s, n_bars)
    rng = np.random.default_rng(seed)
    depth = np.zeros(n_bars, dtype=np.int64)
    link: dict[int, tuple[int, bool]] = {}
    rhythm_link: dict[int, tuple[int, bool]] = {}
    n_linked = 0
    for j in range(n_bars):
        i = int(repeats.source[j])
        if i < 0 or i >= j:
            continue
        a = t[sl[i]:sl[i + 1]]
        b = t[sl[j]:sl[j + 1]]
        if len(a) < MIN_BAR_EVENTS or len(a) != len(b):
            continue
        shift = (j - i) * bar_s
        if not np.all(np.abs(b - (a + shift)) <= OFFSET_TOL_S):
            continue
        keys = [(PatternLinks.key(ta), PatternLinks.key(tb)) for ta, tb in zip(a, b)]
        if depth[i] >= MAX_LINK_DEPTH:
            for ka, kb in keys:
                rhythm_link[kb] = (ka, False)
            continue
        mirror = bool(rng.random() < MIRROR_SHARE)
        for ka, kb in keys:
            link[kb] = rhythm_link[kb] = (ka, mirror)
        depth[j] = depth[i] + 1
        n_linked += 1
    return PatternLinks(link, bonus=bonus, n_bars=n_linked, rhythm_link=rhythm_link)


def plan_phrases(planned: np.ndarray, probs: np.ndarray, repeats: BarRepeats | None,
                 seed: int, snap=None) -> tuple[np.ndarray, PatternLinks | None, int]:
    """The generate-side entry point: re-snap repeated bars, then link their patterns.

    `planned` are the snapped event times, `probs` the raw timing-model frame probabilities they
    were picked from. The pick curve and threshold are rebuilt the way `pick_events` builds them.
    `snap` (times -> grid-snapped times), when given, puts the events re-snapping dropped back
    outside the repeated bars (`refill`), so the density stays the timing model's.
    Returns (times, links or None, number of re-snapped bars).
    """
    from bsmaker.models.timing.decode import MIN_FIRST_EVENT_S, local_normalise, threshold_for_count

    times = np.sort(np.asarray(planned, dtype=np.float64))
    if repeats is None or len(times) < 2 * MIN_BAR_EVENTS or len(probs) == 0:
        return times, None, 0
    p = np.array(probs, dtype=np.float32)
    p[:int(round(MIN_FIRST_EVENT_S * FRAMES_PER_SECOND))] = 0.0
    curve = local_normalise(p)
    thr = threshold_for_count(curve, len(times))
    touched: set[int] = set()
    n_before = len(times)
    times, n_resnapped = repeat_rhythm(times, curve, thr, repeats, FRAMES_PER_SECOND, touched)
    if snap is not None:
        times = refill(times, curve, FRAMES_PER_SECOND, n_before - len(times), repeats.bar_s,
                       touched, snap)
    return times, pattern_links(times, repeats, seed), n_resnapped
