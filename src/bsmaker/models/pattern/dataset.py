"""Runs of consecutive events of one difficulty for the pattern model."""
from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from bsmaker.audio.decode import SAMPLE_RATE
from bsmaker.audio.mel import HOP
from bsmaker.data.events_cache import Entry, EventCache
from bsmaker.data.styles import UNKNOWN
from bsmaker.formats.tokens import NONE_TOKEN
from bsmaker.models.common import log_nps_value, normalise_mel, style_vector
from bsmaker.models.pattern.model import BOS_TOKEN, N_MELS, N_NEXT, SLICE_FRAMES

_HALF = SLICE_FRAMES // 2


def _gaps(times_s: np.ndarray) -> np.ndarray:
    """[L, 5]: log1p gap to the previous event, then to each of the next four (0 if none)."""
    t = np.asarray(times_s, dtype=np.float64)
    n = len(t)
    out = np.zeros((n, 1 + N_NEXT), dtype=np.float32)
    if n > 1:
        out[1:, 0] = np.log1p(np.maximum(t[1:] - t[:-1], 0.0))
    for j in range(1, N_NEXT + 1):
        if n > j:
            out[:n - j, j] = np.log1p(np.maximum(t[j:] - t[:n - j], 0.0))
    return out


def _slices(times_s: np.ndarray, mel: np.ndarray, stats: dict) -> np.ndarray:
    """[L, 35, 80] normalised mel around each event; zeros beyond the song edges."""
    n_frames = mel.shape[0]
    frames = np.round(np.asarray(times_s, dtype=np.float64) * SAMPLE_RATE / HOP).astype(np.int64)
    out = np.zeros((len(frames), SLICE_FRAMES, N_MELS), dtype=np.float32)
    if len(frames) == 0 or n_frames == 0:
        return out
    idx = frames[:, None] + np.arange(-_HALF, _HALF + 1)[None, :]
    ok = (idx >= 0) & (idx < n_frames)
    if not ok.any():
        return out
    lo, hi = int(idx[ok].min()), int(idx[ok].max()) + 1
    chunk = normalise_mel(mel[lo:hi], stats)
    out[ok] = chunk[idx[ok] - lo]
    return out


def event_inputs(times_s: np.ndarray, mel: np.ndarray, stats: dict) -> tuple[np.ndarray, np.ndarray]:
    """(slices [L, 35, 80], gaps [L, 5]) for events at `times_s` (seconds, ascending)."""
    return _slices(times_s, mel, stats), _gaps(times_s)


class PatternSequences(Dataset):
    """One random run of up to `context` events of one difficulty per item.

    Events with both hands empty are skipped. Pass `events_dir` to pickle without the cache.
    """

    def __init__(self, cache: EventCache | None, entries: Sequence[Entry], features_dir: Path,
                 stats: dict, context: int, seed: int, style_dropout: float = 0.15,
                 fixed: bool = False, events_dir: Path | None = None) -> None:
        if cache is None and events_dir is None:
            raise ValueError("give a cache or an events_dir")
        self._cache = cache
        self.events_dir = Path(events_dir) if events_dir is not None else None
        self.entries = list(entries)
        self.features_dir = Path(features_dir)
        self.stats = {"mel_mean": stats["mel_mean"], "mel_std": stats["mel_std"]}
        self.context = context
        self.seed = seed
        self.style_dropout = style_dropout
        self.fixed = fixed
        self._rng: np.random.Generator | None = None

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state["_rng"] = None
        if self.events_dir is not None:
            state["_cache"] = None
        return state

    def _process_rng(self) -> np.random.Generator:
        if self._rng is None:
            info = torch.utils.data.get_worker_info()
            base = info.seed if info is not None else torch.initial_seed()
            self._rng = np.random.default_rng([self.seed, int(base)])
        return self._rng

    def __len__(self) -> int:
        return len(self.entries)

    def _events(self, i: int):
        if self._cache is None:
            self._cache = EventCache.load(self.events_dir)
        t, left, right = self._cache.events(i)
        keep = (left != NONE_TOKEN) | (right != NONE_TOKEN)
        return (np.asarray(t)[keep], np.asarray(left)[keep].astype(np.int64),
                np.asarray(right)[keep].astype(np.int64))

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        e = self.entries[idx]
        rng = np.random.default_rng([self.seed, idx]) if self.fixed else self._process_rng()
        times, left, right = self._events(e.i)
        total = len(times)
        c = self.context
        start = int(rng.integers(0, total - c + 1)) if total > c else 0
        n = min(c, total - start)
        sl = slice(start, start + n)
        gaps = np.zeros((c, 1 + N_NEXT), dtype=np.float32)
        gaps[:n] = _gaps(times)[sl]  # true neighbours, also outside the run
        slices = np.zeros((c, SLICE_FRAMES, N_MELS), dtype=np.float32)
        if n:
            mel = np.load(self.features_dir / f"{e.map_key}.npy", mmap_mode="r",
                          allow_pickle=False)
            mel = mel[:min(int(e.n_frames), mel.shape[0])]
            slices[:n] = _slices(times[sl], mel, self.stats)
        prev_l = np.full(c, BOS_TOKEN, dtype=np.int64)
        prev_r = np.full(c, BOS_TOKEN, dtype=np.int64)
        if n:
            prev_l[1:n] = left[start:start + n - 1]
            prev_r[1:n] = right[start:start + n - 1]
            if start > 0:
                prev_l[0], prev_r[0] = left[start - 1], right[start - 1]
        tl = np.zeros(c, dtype=np.int64)
        tr = np.zeros(c, dtype=np.int64)
        tl[:n], tr[:n] = left[sl], right[sl]
        valid = np.zeros(c, dtype=bool)
        valid[:n] = True
        styles = e.styles
        if not self.fixed and styles != (UNKNOWN,) and rng.random() < self.style_dropout:
            styles = ()
        return {"mel_slices": torch.from_numpy(slices), "gaps": torch.from_numpy(gaps),
                "prev_left": torch.from_numpy(prev_l), "prev_right": torch.from_numpy(prev_r),
                "left": torch.from_numpy(tl), "right": torch.from_numpy(tr),
                "valid": torch.from_numpy(valid),
                "log_nps": torch.tensor(log_nps_value(e.nps), dtype=torch.float32),
                "styles": torch.from_numpy(style_vector(styles))}
