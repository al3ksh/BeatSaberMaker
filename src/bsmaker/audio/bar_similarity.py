"""Which bars of a song repeat earlier music: chroma + MFCC per bar, compared pairwise.

The features match the reference-map analysis (.superpowers/refmaps/scripts/pat_analyze.py):
per bar the mean chroma (12) and the mean MFCC 1-12, each half unit-normalised, so the cosine
similarity of two bars is in [-1, 1]. A bar "repeats" an earlier one when their similarity is
above the song's REPEAT_QUANTILE of all bar pairs.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from bsmaker.audio.decode import SAMPLE_RATE

FEATURE_HOP = 512
BEATS_PER_BAR = 4
# The reference analysis calls a bar pair an audio repeat above the song's 95th percentile.
REPEAT_QUANTILE = 0.95
# Bars quieter than this many dB below the song's loudest bars (95th percentile) are not
# compared: near-silent chroma is noise and would make every quiet bar "similar".
QUIET_DB = 30.0


@dataclass(frozen=True)
class BarRepeats:
    """Bars of BEATS_PER_BAR beats from time 0 (the map's beat 0) and their earlier twin.

    `source[j]` is the index of the earlier bar bar `j` repeats, or -1.
    """
    bar_s: float
    source: np.ndarray
    sim: np.ndarray | None = None

    @property
    def n_bars(self) -> int:
        return len(self.source)

    def bar_of(self, t: float) -> int:
        return int(np.floor(t / self.bar_s + 1e-9))


def bar_features(audio: np.ndarray, bar_s: float) -> tuple[np.ndarray, np.ndarray]:
    """Per-bar feature rows `[n_bars, 24]` and the per-bar loudness in dB."""
    import librosa

    y = np.asarray(audio, dtype=np.float32)
    chroma = librosa.feature.chroma_stft(y=y, sr=SAMPLE_RATE, hop_length=FEATURE_HOP)
    mfcc = librosa.feature.mfcc(y=y, sr=SAMPLE_RATE, hop_length=FEATURE_HOP, n_mfcc=13)
    rms = librosa.feature.rms(y=y, hop_length=FEATURE_HOP)[0]
    return features_from_frames(chroma, mfcc, rms, FEATURE_HOP / SAMPLE_RATE, bar_s)


def features_from_frames(chroma: np.ndarray, mfcc: np.ndarray, rms: np.ndarray, hop_s: float,
                         bar_s: float) -> tuple[np.ndarray, np.ndarray]:
    nfr = rms.shape[0]
    n_bars = max(1, int(np.ceil(nfr * hop_s / bar_s)))
    feats = np.zeros((n_bars, chroma.shape[0] + mfcc.shape[0] - 1), dtype=np.float64)
    loud = np.full(n_bars, -np.inf)
    for k in range(n_bars):
        i0 = int(k * bar_s / hop_s)
        i1 = min(max(int((k + 1) * bar_s / hop_s), i0 + 1), nfr)
        if i0 >= nfr:
            break
        c = chroma[:, i0:i1].mean(1)
        m = mfcc[1:, i0:i1].mean(1)
        feats[k] = np.concatenate([c / (np.linalg.norm(c) + 1e-9), m / (np.linalg.norm(m) + 1e-9)])
        loud[k] = 20 * np.log10(np.sqrt(np.mean(rms[i0:i1] ** 2)) + 1e-6)
    return feats, loud


def find_repeats(feats: np.ndarray, loud: np.ndarray, bar_s: float,
                 quantile: float = REPEAT_QUANTILE) -> BarRepeats:
    """For every bar, the earlier bar it repeats (or -1).

    Candidates are earlier bars whose similarity is above the song's `quantile` of all pairs of
    audible bars. Among them the one whose neighbours also match best wins (similarity averaged
    along the diagonal, bars j-1..j+1 against i-1..i+1), so a repeated phrase copies bar by bar
    from one place instead of jumping between look-alike bars.
    """
    n = len(feats)
    source = np.full(n, -1, dtype=np.int64)
    finite = loud[np.isfinite(loud)]
    if n < 4 or len(finite) < 4:
        return BarRepeats(bar_s, source)
    ok = np.isfinite(loud) & (loud >= np.percentile(finite, 95) - QUIET_DB)
    sim = feats @ feats.T / 2.0
    sim[~ok, :] = -1.0
    sim[:, ~ok] = -1.0
    np.fill_diagonal(sim, -1.0)
    idx = np.nonzero(ok)[0]
    if len(idx) < 4:
        return BarRepeats(bar_s, source, sim)
    pairs = sim[np.ix_(idx, idx)][np.triu_indices(len(idx), 1)]
    thr = float(np.quantile(pairs, quantile))
    smooth = sim.copy()
    cnt = np.ones_like(sim)
    for d in (-1, 1):
        shifted = np.full_like(sim, np.nan)
        a0, a1 = max(0, -d), n - max(0, d)
        shifted[a0:a1, a0:a1] = sim[a0 + d:a1 + d, a0 + d:a1 + d]
        good = np.isfinite(shifted) & (shifted > -1.0)
        smooth[good] += shifted[good]
        cnt[good] += 1
    smooth /= cnt
    for j in idx:
        cand = np.nonzero(sim[j, :j] >= thr)[0]
        if len(cand):
            source[j] = int(cand[np.argmax(smooth[j, cand])])
    return BarRepeats(bar_s, source, sim)


def bar_repeats(audio: np.ndarray, bpm: float) -> BarRepeats:
    """Bar repeats of a song whose map uses `bpm` (bars of BEATS_PER_BAR map beats from 0 s)."""
    bar_s = BEATS_PER_BAR * 60.0 / bpm
    feats, loud = bar_features(audio, bar_s)
    return find_repeats(feats, loud, bar_s)
