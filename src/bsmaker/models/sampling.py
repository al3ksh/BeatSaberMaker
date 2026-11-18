"""Quality- and density-based sampling weights for training difficulties."""
from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from bsmaker.data.events_cache import Entry, EventCache
    from bsmaker.data.pattern_metrics import PatternMetrics

EXCLUDED_TAGS = ("comedy-meme", "challenge", "ai")
MIN_TRAIN_SCORE = 0.70
MISSING_SCORE_WEIGHT = 0.6
# pattern factor (opt-in, `--weighting quality+pattern` of the pattern model)
PATTERN_RANKS = ("Hard", "Expert", "ExpertPlus")
REF_LIKE_FACTOR = 1.5
FLAT_UNKNOWN_FACTOR = 0.5
FLAT_MIN_BOTTOM_UP = 0.35


def entry_flags(score: float | None, tags: Sequence[str]) -> tuple[str, ...]:
    """Reasons a difficulty is left out of training sampling (a missing score is not low)."""
    flags: list[str] = []
    if score is not None and score < MIN_TRAIN_SCORE:
        flags.append("low_score")
    if any(t in EXCLUDED_TAGS for t in tags):
        flags.append("joke_tag")
    return tuple(flags)


def quality_weight(score: float | None) -> float:
    if score is None:
        return MISSING_SCORE_WEIGHT
    return 0.25 + 0.75 * float(np.clip((score - 0.80) / 0.15, 0.0, 1.0))


def entry_weights(entries: Sequence[Entry], stats: dict,
                  factors: np.ndarray | None = None) -> np.ndarray:
    """Per-entry sampling weight: quality x density (x `factors`); exactly 0.0 for flagged entries."""
    n = len(entries)
    if factors is not None and len(factors) != n:
        raise ValueError(f"{len(factors)} factors for {n} entries")
    if n == 0:
        return np.zeros(0, dtype=np.float64)
    edges = np.asarray(stats["nps_bins"], dtype=np.float64)
    n_bins = len(edges) - 1
    nps = np.array([e.nps for e in entries], dtype=np.float64)
    bins = np.clip(np.searchsorted(edges, nps, side="right") - 1, 0, n_bins - 1)
    counts = np.bincount(bins, minlength=n_bins)
    median = float(np.median(counts[counts > 0]))
    density = np.clip((median / counts[bins]) ** 0.5, 0.5, 3.0)
    quality = np.array([quality_weight(e.score) for e in entries], dtype=np.float64)
    w = quality * density
    if factors is not None:
        w = w * np.asarray(factors, dtype=np.float64)
    w[[bool(e.flags) for e in entries]] = 0.0
    return w


def pattern_factor(entry: Entry, metrics: PatternMetrics | None) -> float:
    """Weight multiplier from the pattern shape (Hard and above only).

    1.5 for reference-like patterns, 0.5 for untagged ("unknown" style) maps whose up cuts sit
    mostly on the bottom row, else 1.0.
    """
    from bsmaker.data.pattern_metrics import is_ref_like

    if metrics is None or entry.rank not in PATTERN_RANKS:
        return 1.0
    if is_ref_like(metrics):
        return REF_LIKE_FACTOR
    if "unknown" in entry.styles and metrics.bottom_up_share > FLAT_MIN_BOTTOM_UP:
        return FLAT_UNKNOWN_FACTOR
    return 1.0


def pattern_factors(cache: EventCache, entries: Sequence[Entry]) -> np.ndarray:
    """`pattern_factor` of every entry, with metrics computed from the cached token streams."""
    from bsmaker.data.pattern_metrics import compute_pattern_metrics

    out = np.ones(len(entries), dtype=np.float64)
    for k, e in enumerate(entries):
        if e.rank in PATTERN_RANKS:
            _, left, right = cache.events(e.i)
            out[k] = pattern_factor(e, compute_pattern_metrics(left, right))
    return out
