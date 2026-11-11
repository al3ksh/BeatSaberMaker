"""Beat grid detection and snapping of note times to the grid."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from bsmaker.audio.decode import SAMPLE_RATE

MIN_STEADY_BEATS = 16
MAX_STEADY_RESIDUAL_S = 0.020


@dataclass(frozen=True)
class Grid:
    bpm: float
    offset_s: float
    steady: bool


def fit_grid(beat_times_s: np.ndarray) -> Grid:
    """Least-squares line through beat index -> time."""
    t = np.asarray(beat_times_s, dtype=np.float64).ravel()
    if len(t) < 2:
        return Grid(120.0, 0.0, False)
    k = np.arange(len(t), dtype=np.float64)
    slope, intercept = np.polyfit(k, t, 1)
    fallback_interval = float(np.median(np.diff(t)))
    if not (slope > 0 and np.isfinite(slope) and np.isfinite(intercept)):
        if fallback_interval > 0:
            return Grid(60.0 / fallback_interval, 0.0, False)
        return Grid(120.0, 0.0, False)
    residual = np.abs(t - (intercept + k * slope))
    steady = len(t) >= MIN_STEADY_BEATS and float(np.median(residual)) < MAX_STEADY_RESIDUAL_S
    if not steady:
        return Grid(60.0 / fallback_interval if fallback_interval > 0 else 120.0, 0.0, False)
    return Grid(60.0 / float(slope), float(intercept % slope), True)


def detect_grid(audio: np.ndarray) -> Grid:
    import librosa

    _, beats = librosa.beat.beat_track(y=audio, sr=SAMPLE_RATE, units="time")
    beats = np.asarray(beats, dtype=np.float64)
    if len(beats) < 4:
        return Grid(120.0, 0.0, False)
    return fit_grid(beats)


def snap_times(times_s: np.ndarray, grid: Grid, max_shift_s: float = 0.025) -> np.ndarray:
    times = np.asarray(times_s, dtype=np.float64)
    if len(times) > 1 and np.any(np.diff(times) <= 0):
        raise ValueError("times_s must be strictly increasing")
    if not grid.steady or len(times) == 0:
        return times.copy()
    beat = 60.0 / grid.bpm
    out = times.copy()
    prev = -np.inf
    for i, t in enumerate(times):
        best = t
        best_shift = np.inf
        for div in (16, 12):
            step = beat / div
            cand = grid.offset_s + np.round((t - grid.offset_s) / step) * step
            if abs(cand - t) < best_shift:
                best, best_shift = cand, abs(cand - t)
        value = best if best_shift <= max_shift_s else t
        if value <= prev:
            value = t
        out[i] = value
        prev = value
    return out


def to_beats(times_s: np.ndarray, bpm: float) -> np.ndarray:
    return np.asarray(times_s, dtype=np.float64) * bpm / 60.0
