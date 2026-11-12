"""Onset metrics for the timing model."""
from __future__ import annotations

from collections.abc import Sequence


def onset_f1(pred_s: Sequence[float], true_s: Sequence[float],
             tol_s: float = 0.025) -> tuple[float, float, float]:
    """(precision, recall, F1) with one-to-one greedy matching in time order within `tol_s`."""
    pred = sorted(float(x) for x in pred_s)
    true = sorted(float(x) for x in true_s)
    i = j = hits = 0
    while i < len(pred) and j < len(true):
        d = pred[i] - true[j]
        if abs(d) <= tol_s + 1e-9:
            hits += 1
            i += 1
            j += 1
        elif d < 0:
            i += 1
        else:
            j += 1
    precision = hits / len(pred) if pred else 0.0
    recall = hits / len(true) if true else 0.0
    f1 = 2 * precision * recall / (precision + recall) if hits else 0.0
    return precision, recall, f1
