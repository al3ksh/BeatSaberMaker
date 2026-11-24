"""Hierarchical audio phrase similarity for musical bar-repeat detection."""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from bsmaker.audio.decode import SAMPLE_RATE

FEATURE_HOP = 512
BEATS_PER_BAR = 4

# pat_analyze_f1.py:589 defines audio-repeat bars above the song's 95th similarity percentile.
REPEAT_QUANTILE = 0.95
# Absolute cosine floor so through-composed songs without real repeats do not force spurious copies
# (chroma+MFCC unit-half cosine in [-1, 1]; real repeats in ref/eval songs exceed 0.82).
MIN_BAR_SIM = 0.82
# Cap quantile threshold on loop songs where >5% of bar pairs are near-identical (avoids STFT
# frame-alignment jitter in 0.9999+ pairs rejecting true loop repeats).
MAX_REPEAT_THR = 0.95
# Per-bar floor within a matched phrase: every bar in an 8/4/2-bar phrase must stay near `thr`.
PHRASE_BAR_SLACK = 0.05
# Exclude near-silent bars from comparison (same 30 dB below p95 guard as F7; avoids chroma noise).
QUIET_DB = 30.0
# Absolute loudness floor so all-silent / -80 dB clips never register audible matches.
MIN_AUDIBLE_DB = -60.0
# Do not copy a rhythm across a section boundary (pat_analyze_f1.py:660 defines boundaries at >= 3 dB).
MAX_REPEAT_LOUD_DIFF_DB = 3.5
# Phrase lengths tried longest-first (human mappers repeat 4- and 8-bar musical phrases; 2-bar fallback).
PHRASE_LENGTHS = (8, 4, 2)
# Anti-monotony caps (ref hand-free repeat 0.597, with-hand 0.316, 2-bar 0.308; F7 SENTINO ALE hit
# 0.549 pattern reuse from uncapped loop copying):
MAX_RHYTHM_REPEAT_SHARE = 0.48   # at most 48% of audible bars get a repeat source (tuned from 0.40 after Eval #1)
MAX_CONSECUTIVE_REPEAT_BARS = 8  # at most one 8-bar phrase repeated before a fresh break
COOLDOWN_BARS = 2                # minimum fresh bars after hitting MAX_CONSECUTIVE_REPEAT_BARS
MAX_COPIES_PER_SOURCE = 3        # cap copies of any single canonical source bar (on 3rd copy, vary tail)
VARY_EVERY_NTH_REPEAT = 3        # on every Nth repeat of a phrase, leave the final bar fresh (turnaround)


@dataclass(frozen=True)
class BarRepeats:
    """Detected hierarchical bar-repeat mapping aligned to map bars `[k*bar_s, (k+1)*bar_s)`."""

    bar_s: float
    source: np.ndarray        # int64[n_bars]: canonical source bar index or -1
    sim: np.ndarray           # float32[n_bars, n_bars]: pairwise bar similarity
    phrase_len: np.ndarray    # int64[n_bars]: matched phrase length (8, 4, 2) or 0
    phrase_start: np.ndarray  # int64[n_bars]: start bar index of matched phrase or -1


def features_from_frames(
    chroma: np.ndarray,
    mfcc: np.ndarray,
    rms: np.ndarray,
    hop_s: float,
    bar_s: float,
    offset_s: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Aggregate frame-level chroma, MFCC, and RMS into per-bar `(features, loud_db)`."""
    nfr = min(chroma.shape[1], mfcc.shape[1], len(rms))
    if (
        nfr <= 0
        or not np.isfinite(hop_s)
        or hop_s <= 0
        or not np.isfinite(bar_s)
        or bar_s <= 0
        or not np.isfinite(offset_s)
    ):
        return np.zeros((0, 24), dtype=np.float32), np.zeros(0, dtype=np.float32)
    off = max(0.0, float(offset_s))
    dur_s = nfr * hop_s
    if dur_s <= off:
        return np.zeros((0, 24), dtype=np.float32), np.zeros(0, dtype=np.float32)

    n_bars = max(0, int(math.floor((dur_s - off + hop_s) / bar_s)))
    feats = np.zeros((n_bars, 24), dtype=np.float32)
    loud = np.full(n_bars, np.nan, dtype=np.float32)
    for k in range(n_bars):
        a = off + k * bar_s
        b = off + (k + 1) * bar_s
        i0 = int(a / hop_s)
        if i0 >= nfr:
            break
        i1 = min(max(int(b / hop_s), i0 + 1), nfr)
        c = np.asarray(chroma[:, i0:i1], dtype=np.float64).mean(axis=1)
        m = np.asarray(mfcc[1:13, i0:i1], dtype=np.float64).mean(axis=1)
        v = np.concatenate([
            c / (np.linalg.norm(c) + 1e-9),
            m / (np.linalg.norm(m) + 1e-9),
        ])
        feats[k] = v.astype(np.float32)
        rms_bar = float(np.sqrt(np.mean(np.asarray(rms[i0:i1], dtype=np.float64) ** 2)))
        loud[k] = np.float32(20.0 * math.log10(rms_bar + 1e-6))
    return feats, loud


def bar_features(audio: np.ndarray, bpm: float, offset_s: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-bar chroma+MFCC features and RMS loudness (dB) aligned to `offset_s`."""
    y = np.asarray(audio, dtype=np.float32)
    if y.size == 0 or bpm <= 0 or not np.isfinite(bpm):
        return np.zeros((0, 24), dtype=np.float32), np.zeros(0, dtype=np.float32)
    import librosa

    bar_s = BEATS_PER_BAR * 60.0 / float(bpm)
    chroma = librosa.feature.chroma_stft(y=y, sr=SAMPLE_RATE, hop_length=FEATURE_HOP)
    mfcc = librosa.feature.mfcc(y=y, sr=SAMPLE_RATE, hop_length=FEATURE_HOP, n_mfcc=13)
    rms = librosa.feature.rms(y=y, hop_length=FEATURE_HOP)[0]
    hop_s = FEATURE_HOP / SAMPLE_RATE
    return features_from_frames(chroma, mfcc, rms, hop_s, bar_s, offset_s=offset_s)


def _empty_repeats(n: int, bar_s: float) -> BarRepeats:
    return BarRepeats(
        bar_s=float(bar_s),
        source=np.full(n, -1, dtype=np.int64),
        sim=np.full((n, n), -1.0, dtype=np.float32),
        phrase_len=np.zeros(n, dtype=np.int64),
        phrase_start=np.full(n, -1, dtype=np.int64),
    )


def _pairwise_similarity(
    feats: np.ndarray, loud: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float] | None:
    finite = np.isfinite(loud)
    if not finite.any():
        return None
    p95 = float(np.percentile(loud[finite], 95))
    ok = finite & (loud >= p95 - QUIET_DB) & (loud > MIN_AUDIBLE_DB)
    idx = np.nonzero(ok)[0]
    if len(idx) < 2:
        return None

    clean_feats = np.nan_to_num(np.asarray(feats, dtype=np.float64), nan=0.0)
    sim = (clean_feats @ clean_feats.T) / 2.0
    sim[~ok, :] = -1.0
    sim[:, ~ok] = -1.0
    np.fill_diagonal(sim, -1.0)

    sub = sim[np.ix_(idx, idx)]
    tri = sub[np.triu_indices(len(idx), 1)]
    thr = min(max(float(np.quantile(tri, REPEAT_QUANTILE)), MIN_BAR_SIM), MAX_REPEAT_THR)

    loud_f = np.where(ok, loud, 0.0)
    loud_ok = np.abs(loud_f[:, None] - loud_f[None, :]) <= MAX_REPEAT_LOUD_DIFF_DB
    loud_ok &= ok[:, None] & ok[None, :]
    return ok, sim, loud_ok, thr


def _best_phrase_source(
    j: int,
    length: int,
    n_assigned: int,
    budget: int,
    streak: int,
    ok: np.ndarray,
    source: np.ndarray,
    copy_count: np.ndarray,
    sim: np.ndarray,
    loud_ok: np.ndarray,
    thr: float,
) -> tuple[int, int] | None:
    if j + length > len(source):
        return None
    if not ok[j:j + length].all() or not (source[j:j + length] == -1).all():
        return None
    min_assign = length - 1 if length >= 4 else length
    if n_assigned + min_assign > budget or streak + min_assign > MAX_CONSECUTIVE_REPEAT_BARS:
        return None

    ks = np.arange(length)
    best_i = -1
    best_n_assign = 0
    best_score = -np.inf
    for i in range(0, j - length + 1, 2):
        if not ok[i:i + length].all():
            continue
        if not (source[i:i + length] == -1).all():
            continue
        if not (copy_count[i:i + length] < MAX_COPIES_PER_SOURCE).all():
            continue
        if not loud_ok[j + ks, i + ks].all():
            continue
        pair_sims = sim[j + ks, i + ks]
        if float(pair_sims.min()) < thr - PHRASE_BAR_SLACK or float(pair_sims.mean()) < thr:
            continue

        rep_num = int(copy_count[i:i + length].max()) + 1
        turnaround = (length == 8) or (rep_num % VARY_EVERY_NTH_REPEAT == 0 and length >= 4)
        n_assign = length - 1 if turnaround else length
        if n_assigned + n_assign > budget or streak + n_assign > MAX_CONSECUTIVE_REPEAT_BARS:
            continue

        score = float(pair_sims.mean()) + (0.01 if (j - i) % 4 == 0 else 0.0)
        if score > best_score:
            best_i, best_n_assign, best_score = i, n_assign, score
    return (best_i, best_n_assign) if best_i >= 0 else None


def find_repeats(feats: np.ndarray, loud: np.ndarray, bar_s: float) -> BarRepeats:
    """Find hierarchical 8/4/2-bar phrase repeats with canonical sources and anti-monotony caps."""
    feats = np.asarray(feats, dtype=np.float32)
    loud = np.asarray(loud, dtype=np.float32)
    n = len(loud)
    if n < 2 or feats.shape[0] != n:
        return _empty_repeats(n, bar_s)

    prepared = _pairwise_similarity(feats, loud)
    if prepared is None:
        return _empty_repeats(n, bar_s)
    ok, sim, loud_ok, thr = prepared

    source = np.full(n, -1, dtype=np.int64)
    phrase_len = np.zeros(n, dtype=np.int64)
    phrase_start = np.full(n, -1, dtype=np.int64)
    copy_count = np.zeros(n, dtype=np.int64)
    budget = int(math.floor(MAX_RHYTHM_REPEAT_SHARE * int(ok.sum())))

    j = 0
    n_assigned = 0
    streak = 0
    cooldown_left = 0
    while j + 2 <= n:
        if cooldown_left > 0:
            cooldown_left = max(0, cooldown_left - 2)
            streak = 0
            j += 2
            continue

        matched = None
        for length in PHRASE_LENGTHS:
            hit = _best_phrase_source(
                j, length, n_assigned, budget, streak,
                ok, source, copy_count, sim, loud_ok, thr,
            )
            if hit is not None:
                matched = (length, *hit)
                break

        if matched is None:
            streak = 0
            j += 2
            continue

        length, src_i, n_assign = matched
        copy_count[src_i:src_i + n_assign] += 1
        source[j:j + n_assign] = np.arange(src_i, src_i + n_assign, dtype=np.int64)
        phrase_len[j:j + n_assign] = length
        phrase_start[j:j + n_assign] = j
        n_assigned += n_assign
        if n_assign < length:
            streak = 0
        else:
            streak += n_assign
            if streak >= MAX_CONSECUTIVE_REPEAT_BARS:
                cooldown_left = COOLDOWN_BARS
                streak = 0
        j += length

    return BarRepeats(
        bar_s=float(bar_s),
        source=source,
        sim=sim.astype(np.float32),
        phrase_len=phrase_len,
        phrase_start=phrase_start,
    )


def bar_repeats(audio: np.ndarray, bpm: float, offset_s: float = 0.0) -> BarRepeats:
    """Detect hierarchical audio phrase repeats directly from mono audio and steady BPM."""
    bar_s = BEATS_PER_BAR * 60.0 / float(bpm) if bpm > 0 and np.isfinite(bpm) else 0.0
    feats, loud = bar_features(audio, bpm, offset_s=offset_s)
    return find_repeats(feats, loud, bar_s)
