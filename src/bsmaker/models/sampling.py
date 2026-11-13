"""Quality- and density-based sampling weights for training difficulties."""
from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from bsmaker.data.events_cache import Entry

EXCLUDED_TAGS = ("comedy-meme", "challenge", "ai")
MIN_TRAIN_SCORE = 0.70
MISSING_SCORE_WEIGHT = 0.6


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


def entry_weights(entries: Sequence[Entry], stats: dict) -> np.ndarray:
    """Per-entry sampling weight: quality x density; exactly 0.0 for flagged entries."""
    n = len(entries)
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
    w[[bool(e.flags) for e in entries]] = 0.0
    return w
