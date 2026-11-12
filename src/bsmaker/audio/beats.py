"""Beat grid detection and snapping of note times to the grid."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from bsmaker.audio.decode import SAMPLE_RATE

MIN_STEADY_BEATS = 16
MAX_STEADY_RESIDUAL_S = 0.020

ENV_HOP = 256
ENV_FPS = SAMPLE_RATE / ENV_HOP
MIN_AUDIO_S = 30.0
BPM_SEARCH_REL = 0.03  # search seed tempo +-3%
BPM_STEP_REL = 0.0001  # in steps of 0.01% of the seed
COARSE_PHASE_S = 0.010
FINE_PHASE_S = 0.001
SEG_PHASE_S = 0.005
N_SEGMENTS = 4
MIN_CONTRAST = 1.5
MAX_SEGMENT_DEV_S = 0.030
# Lag of the onset-strength envelope behind the true beat: the median
# measured on 80 validation songs.
ENV_LAG_S = 0.030

# Snapping of predicted events (tuned on 80 val songs: F1@10 ms 0.458 -> 0.629, F1@25 ms -0.003).
SNAP_DIVISIONS = (4, 3)  # per beat, coarse first; human notes: 71-82% on 1/4, ~2-3% on 1/3
SNAP_MAX_SHIFT_S = 0.045
SNAP_MAX_STEP_FRACTION = 0.45  # never move more than this share of the smallest step
SNAP_FINER_MARGIN_S = 0.008  # the 1/3 point must be this much closer than the 1/4 point
ALIGN_SIGMA_S = 0.015  # width of the event-to-grid score
ALIGN_STEP_S = 0.001
ALIGN_MIN_EVENTS = 8


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
    fallback_interval = float(np.median(np.diff(t)))
    if not fallback_interval > 0:
        return Grid(120.0, 0.0, False)
    # beat index from the time, so a missed beat does not shift later indices
    steps = np.maximum(1.0, np.round(np.diff(t) / fallback_interval))
    k = np.concatenate(([0.0], np.cumsum(steps)))
    slope, intercept = np.polyfit(k, t, 1)
    if not (slope > 0 and np.isfinite(slope) and np.isfinite(intercept)):
        if fallback_interval > 0:
            return Grid(60.0 / fallback_interval, 0.0, False)
        return Grid(120.0, 0.0, False)
    residual = np.abs(t - (intercept + k * slope))
    steady = len(t) >= MIN_STEADY_BEATS and float(np.median(residual)) < MAX_STEADY_RESIDUAL_S
    if not steady:
        return Grid(60.0 / fallback_interval if fallback_interval > 0 else 120.0, 0.0, False)
    return Grid(60.0 / float(slope), float(intercept % slope), True)


def _comb_scores(env: np.ndarray, bpms: np.ndarray, phases: np.ndarray, n_frames: int) -> np.ndarray:
    """Mean envelope at phase + k*P over the song, for one BPM, all phases. Shape (len(phases),)."""
    period = 60.0 / bpms
    return _comb_scores_ks(env, period, phases, np.arange(int(n_frames / ENV_FPS / period) - 1))


def _comb_scores_ks(env: np.ndarray, period: float, phases: np.ndarray, ks: np.ndarray) -> np.ndarray:
    if len(ks) == 0:
        return np.full(len(phases), -np.inf)
    pos = (phases[:, None] + ks[None, :] * period) * ENV_FPS
    return np.interp(pos.ravel(), np.arange(len(env)), env).reshape(pos.shape).mean(axis=1)


def _circ(d: float, period: float) -> float:
    return (d + period / 2) % period - period / 2


def _unsteady(bpm: float | None = None) -> Grid:
    if bpm is None or not np.isfinite(bpm) or bpm <= 0:
        return Grid(120.0, 0.0, False)
    return Grid(float(bpm), 0.0, False)


def detect_grid(audio: np.ndarray) -> Grid:
    """Comb-filter fit of one constant BPM and phase to the onset envelope."""
    import librosa

    audio = np.asarray(audio, dtype=np.float32)
    if len(audio) < MIN_AUDIO_S * SAMPLE_RATE:
        return _unsteady()
    tempo, beats = librosa.beat.beat_track(y=audio, sr=SAMPLE_RATE, units="time")
    seed = float(np.atleast_1d(tempo)[0])
    if len(beats) < 4 or not (np.isfinite(seed) and seed > 0):
        return _unsteady()
    env = librosa.onset.onset_strength(y=audio, sr=SAMPLE_RATE, hop_length=ENV_HOP)
    kernel = np.exp(-0.5 * (np.arange(-4, 5) / 1.0) ** 2)
    env = np.convolve(env, kernel / kernel.sum(), mode="same")
    env = env - env.min()
    mean = float(env.mean())
    if not mean > 1e-9:
        return _unsteady(seed)

    n = len(env)
    rel = np.arange(-BPM_SEARCH_REL, BPM_SEARCH_REL + 1e-12, BPM_STEP_REL)
    best_score, best_bpm = -np.inf, seed
    for bpm in seed * (1 + rel):
        period = 60.0 / bpm
        ph = np.arange(0.0, period, COARSE_PHASE_S)
        s = _comb_scores(env, bpm, ph, n).max()
        if s > best_score:
            best_score, best_bpm = float(s), float(bpm)
    period = 60.0 / best_bpm
    phases = np.arange(0.0, period, FINE_PHASE_S)
    scores = _comb_scores(env, best_bpm, phases, n)
    phase = float(phases[int(np.argmax(scores))])
    contrast = float(scores.max()) / mean

    ks_all = np.arange(int(n / ENV_FPS / period) - 1)
    segdev = 0.0
    seg_phases = np.arange(0.0, period, SEG_PHASE_S)
    for q in range(N_SEGMENTS):
        ks = ks_all[q * len(ks_all) // N_SEGMENTS:(q + 1) * len(ks_all) // N_SEGMENTS]
        sp = float(seg_phases[int(np.argmax(_comb_scores_ks(env, period, seg_phases, ks)))])
        segdev = max(segdev, abs(_circ(_circ(sp - phase, period), period / 2)))

    offset = (phase - ENV_LAG_S) % period
    if not all(np.isfinite([best_bpm, contrast, segdev, offset])):
        return _unsteady(seed)
    if contrast >= MIN_CONTRAST and segdev <= MAX_SEGMENT_DEV_S:
        return Grid(best_bpm, float(offset), True)
    return _unsteady(best_bpm)


def align_grid(grid: Grid, event_times_s: np.ndarray) -> Grid:
    """Same BPM, offset re-fitted to where the events sit.

    The offset (mod 1/4 beat) maximises the summed Gaussian closeness of the events to the
    1/4-beat points; of the four beat origins that leave, the one with most events near
    1/3-beat points is kept. Unsteady grids and fewer than `ALIGN_MIN_EVENTS` events: unchanged.
    """
    t = np.asarray(event_times_s, dtype=np.float64).ravel()
    if not grid.steady or len(t) < ALIGN_MIN_EVENTS:
        return grid
    beat = 60.0 / grid.bpm
    quarter = beat / 4
    cand = np.arange(0.0, quarter, ALIGN_STEP_S)
    d = (t[None, :] - cand[:, None] + quarter / 2) % quarter - quarter / 2
    o4 = float(cand[int(np.argmax(np.exp(-d ** 2 / (2 * ALIGN_SIGMA_S ** 2)).sum(axis=1)))])
    origins = o4 + quarter * np.arange(4)
    third = beat / 3
    d3 = (t[None, :] - origins[:, None] + third / 2) % third - third / 2
    best = float(origins[int(np.argmax(np.exp(-d3 ** 2 / (2 * ALIGN_SIGMA_S ** 2)).sum(axis=1)))])
    return Grid(grid.bpm, best % beat, True)


def snap_times(times_s: np.ndarray, grid: Grid, max_shift_s: float = SNAP_MAX_SHIFT_S) -> np.ndarray:
    """Snap to the 1/4- or 1/3-beat grid points; the 1/4 point unless the 1/3 one is clearly closer.

    Strictly increasing input in, strictly increasing output out: a snap is taken only if it
    lands between the previous output and the next original time.
    """
    times = np.asarray(times_s, dtype=np.float64)
    if len(times) > 1 and np.any(np.diff(times) <= 0):
        raise ValueError("times_s must be strictly increasing")
    if not grid.steady or len(times) == 0:
        return times.copy()
    beat = 60.0 / grid.bpm
    cap = min(max_shift_s, SNAP_MAX_STEP_FRACTION * beat / max(SNAP_DIVISIONS))
    out = times.copy()
    prev = -np.inf
    n = len(times)
    for i, t in enumerate(times):
        best = t
        best_shift = np.inf
        for div in SNAP_DIVISIONS:  # coarse first
            step = beat / div
            cand = grid.offset_s + np.round((t - grid.offset_s) / step) * step
            shift = abs(cand - t)
            if best_shift == np.inf or shift + SNAP_FINER_MARGIN_S < best_shift:
                best, best_shift = cand, shift
        upper = times[i + 1] if i + 1 < n else np.inf
        if best_shift <= cap and prev < best < upper:
            out[i] = best
        prev = out[i]
    return out


def to_beats(times_s: np.ndarray, bpm: float) -> np.ndarray:
    return np.asarray(times_s, dtype=np.float64) * bpm / 60.0
