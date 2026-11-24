"""Per-difficulty pattern-shape metrics on model token streams.

Definitions match the reference-map analysis:
- diag_share: diagonal cuts / all notes
- bottom_up_share: straight up cuts on the bottom row / all straight up cuts
- top_row_share: top-row notes / all notes
- flat_flip_share: of a hand's consecutive notes that flip straight up <-> straight down,
  the share that stays on the same row
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from bsmaker.formats.tokens import NONE_TOKEN, VOCAB_SIZE, decode_token

_ROW = np.full(VOCAB_SIZE, -1, dtype=np.int8)
_DIR = np.full(VOCAB_SIZE, -1, dtype=np.int8)
for _t in range(1, VOCAB_SIZE):
    _, _ROW[_t], _DIR[_t] = decode_token(_t)
del _t

UP, DOWN = 0, 1
DIAGONALS = (4, 5, 6, 7)
TOP_ROW, BOTTOM_ROW = 2, 0


@dataclass(frozen=True)
class PatternMetrics:
    diag_share: float
    bottom_up_share: float
    top_row_share: float
    flat_flip_share: float


def compute_pattern_metrics(left: np.ndarray, right: np.ndarray) -> PatternMetrics:
    """Metrics for one difficulty's left/right token streams (0.0 where a share is undefined)."""
    n = n_diag = n_top = n_up = n_up_bottom = n_flips = n_flat = 0
    for hand in (left, right):
        tok = np.asarray(hand, dtype=np.int64)
        tok = tok[tok != NONE_TOKEN]
        if len(tok) == 0:
            continue
        row, d = _ROW[tok], _DIR[tok]
        n += len(tok)
        n_diag += int(np.isin(d, DIAGONALS).sum())
        n_top += int((row == TOP_ROW).sum())
        up = d == UP
        n_up += int(up.sum())
        n_up_bottom += int((up & (row == BOTTOM_ROW)).sum())
        vert = (d == UP) | (d == DOWN)
        flip = vert[:-1] & vert[1:] & (d[:-1] != d[1:])
        n_flips += int(flip.sum())
        n_flat += int((flip & (row[:-1] == row[1:])).sum())
    if n == 0:
        return PatternMetrics(0.0, 0.0, 0.0, 0.0)
    return PatternMetrics(diag_share=n_diag / n,
                          bottom_up_share=n_up_bottom / n_up if n_up else 0.0,
                          top_row_share=n_top / n,
                          flat_flip_share=n_flat / n_flips if n_flips else 0.0)


REF_MIN_DIAG = 0.25
REF_MAX_BOTTOM_UP = 0.25
REF_MIN_TOP_ROW = 0.15
REF_MAX_FLAT_FLIPS = 0.35


def is_ref_like(m: PatternMetrics) -> bool:
    """Patterns close to hand-made reference maps (thresholds from the reference analysis)."""
    return (m.diag_share >= REF_MIN_DIAG and m.bottom_up_share <= REF_MAX_BOTTOM_UP
            and m.top_row_share >= REF_MIN_TOP_ROW and m.flat_flip_share <= REF_MAX_FLAT_FLIPS)
