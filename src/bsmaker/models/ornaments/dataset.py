"""Runs of consecutive notes of one difficulty for the ornament tagger."""
from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from bsmaker.data.events_cache import Entry
from bsmaker.data.layers_cache import LayerCache
from bsmaker.models.common import log_nps_value
from bsmaker.models.ornaments.model import N_GAPS
from bsmaker.models.pattern.dataset import _slices
from bsmaker.models.pattern.model import N_MELS, SLICE_FRAMES


def _prev_next(t: np.ndarray, idx: np.ndarray, prev: np.ndarray, nxt: np.ndarray) -> None:
    """Gaps in seconds (0 where none) to the previous / next note among positions `idx`."""
    if len(idx) > 1:
        d = np.maximum(np.diff(t[idx]), 0.0)
        prev[idx[1:]] = d
        nxt[idx[:-1]] = d


def note_gaps(times_s: np.ndarray, hands: np.ndarray) -> np.ndarray:
    """[L, 4] log1p seconds: previous/next note of any hand, previous/next of the same hand."""
    t = np.asarray(times_s, dtype=np.float64)
    h = np.asarray(hands)
    cols = np.zeros((N_GAPS, len(t)), dtype=np.float64)
    _prev_next(t, np.arange(len(t)), cols[0], cols[1])
    for hand in (0, 1):
        _prev_next(t, np.flatnonzero(h == hand), cols[2], cols[3])
    return np.log1p(cols).T.astype(np.float32)


def note_inputs(times_s: np.ndarray, hands: np.ndarray, mel: np.ndarray,
                stats: dict) -> tuple[np.ndarray, np.ndarray]:
    """(slices [L, 35, 80], gaps [L, 4]) for notes sorted by time."""
    return _slices(times_s, mel, stats), note_gaps(times_s, hands)


def ornament_entries(cache: LayerCache, entries: Sequence[Entry]) -> list[Entry]:
    """Entries whose difficulty is v3 and has at least one arc or chain."""
    return [e for e in entries
            if cache.meta(e.i)["has_v3"] and cache.meta(e.i)["n_arcs"] + cache.meta(e.i)["n_chains"] > 0]


class OrnamentNotes(Dataset):
    """One random run of up to `context` notes of one difficulty per item.

    Pass `layers_dir` so DataLoader workers pickle the dataset without the cache arrays.
    """

    def __init__(self, layer_cache: LayerCache, entries: Sequence[Entry], features_dir: Path,
                 stats: dict, context: int, seed: int, fixed: bool = False,
                 layers_dir: Path | None = None) -> None:
        self._cache: LayerCache | None = layer_cache
        self.layers_dir = Path(layers_dir) if layers_dir is not None else None
        self.entries = ornament_entries(layer_cache, entries)
        self.features_dir = Path(features_dir)
        self.stats = {"mel_mean": stats["mel_mean"], "mel_std": stats["mel_std"]}
        self.context = context
        self.seed = seed
        self.fixed = fixed
        self._rng: np.random.Generator | None = None

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state["_rng"] = None
        if self.layers_dir is not None:
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

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        e = self.entries[idx]
        rng = np.random.default_rng([self.seed, idx]) if self.fixed else self._process_rng()
        if self._cache is None:
            self._cache = LayerCache.load(self.layers_dir)
        t, hand, token, chain, arc_len, arc_cell, arc_dir = (
            np.asarray(a) for a in self._cache.ornaments(e.i))
        total = len(t)
        c = self.context
        start = int(rng.integers(0, total - c + 1)) if total > c else 0
        n = min(c, total - start)
        sl = slice(start, start + n)
        gaps = np.zeros((c, N_GAPS), dtype=np.float32)
        slices = np.zeros((c, SLICE_FRAMES, N_MELS), dtype=np.float32)
        if n:
            gaps[:n] = note_gaps(t, hand)[sl]  # true neighbours, also outside the run
            mel = np.load(self.features_dir / f"{e.map_key}.npy", mmap_mode="r", allow_pickle=False)
            mel = mel[:min(int(e.n_frames), mel.shape[0])]
            slices[:n] = _slices(t[sl], mel, self.stats)

        def pad(a: np.ndarray) -> torch.Tensor:
            out = np.zeros(c, dtype=np.int64)
            out[:n] = a[sl]
            return torch.from_numpy(out)

        chain_cls = np.where(chain > 0, np.clip(chain.astype(np.int64), 3, 7) - 2, 0)
        valid = np.zeros(c, dtype=bool)
        valid[:n] = True
        return {"mel_slices": torch.from_numpy(slices), "gaps": torch.from_numpy(gaps),
                "hand": pad(hand), "token": pad(token), "chain": pad(chain_cls),
                "arc_len": pad(arc_len), "arc_cell": pad(arc_cell), "arc_dir": pad(arc_dir),
                "valid": torch.from_numpy(valid),
                "log_nps": torch.tensor(log_nps_value(e.nps), dtype=torch.float32)}
