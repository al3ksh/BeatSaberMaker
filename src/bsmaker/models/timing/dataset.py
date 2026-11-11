"""Random fixed-length windows of (mel, event-frame targets) for the timing model."""
from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from bsmaker.audio.decode import SAMPLE_RATE
from bsmaker.audio.mel import HOP, N_MELS
from bsmaker.data.events_cache import Entry, EventCache
from bsmaker.data.styles import UNKNOWN
from bsmaker.models.common import log_nps_value, style_vector


def _frames(times_s: np.ndarray) -> np.ndarray:
    """Frame index of each time (same rounding as `time_to_frame`)."""
    return np.round(np.asarray(times_s, dtype=np.float64) * SAMPLE_RATE / HOP).astype(np.int64)


def _targets_from_frames(frames: np.ndarray, n_frames: int) -> np.ndarray:
    out = np.zeros(n_frames, dtype=np.float32)
    for off in (-1, 1):
        f = frames + off
        f = f[(f >= 0) & (f < n_frames)]
        out[f] = np.maximum(out[f], 0.25)
    f = frames[(frames >= 0) & (frames < n_frames)]
    out[f] = 1.0
    return out


def frame_targets(times_s: np.ndarray, n_frames: int) -> np.ndarray:
    """1.0 at each event frame, 0.25 at its two neighbours (max where they overlap)."""
    return _targets_from_frames(_frames(times_s), n_frames)


class TimingWindows(Dataset):
    """One random window of one difficulty per item (`fixed=True`: always the same window).

    Pass `events_dir` so the object pickles without the loaded cache: each DataLoader worker
    then loads the cache itself on first use.
    """

    def __init__(self, cache: EventCache | None, entries: Sequence[Entry], features_dir: Path,
                 stats: dict, window: int, seed: int, style_dropout: float = 0.15,
                 fixed: bool = False, events_dir: Path | None = None) -> None:
        if cache is None and events_dir is None:
            raise ValueError("give a cache or an events_dir")
        self._cache = cache
        self.events_dir = Path(events_dir) if events_dir is not None else None
        self.entries = list(entries)
        self.features_dir = Path(features_dir)
        self.mean = float(stats["mel_mean"])
        self.std = float(stats["mel_std"]) or 1.0
        self.window = window
        self.seed = seed
        self.style_dropout = style_dropout
        self.fixed = fixed
        self._rng: np.random.Generator | None = None

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state["_rng"] = None  # every process seeds its own generator
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

    def _event_times(self, i: int) -> np.ndarray:
        if self._cache is None:
            self._cache = EventCache.load(self.events_dir)
        return self._cache.events(i)[0]

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        e = self.entries[idx]
        if self.fixed:
            rng = np.random.default_rng([self.seed, idx])
        else:  # one stream per process, seeded from (seed, worker/torch seed)
            rng = self._process_rng()
        mel_all = np.load(self.features_dir / f"{e.map_key}.npy", mmap_mode="r",
                          allow_pickle=False)
        total = min(int(e.n_frames), mel_all.shape[0])
        start = int(rng.integers(0, total - self.window + 1)) if total > self.window else 0
        length = min(self.window, total - start)
        mel = np.zeros((self.window, N_MELS), dtype=np.float32)
        mel[:length] = (np.asarray(mel_all[start:start + length], dtype=np.float32)
                        - self.mean) / self.std
        frames = _frames(self._event_times(e.i)) - start
        target = np.zeros(self.window, dtype=np.float32)
        target[:length] = _targets_from_frames(frames, length)
        valid = np.zeros(self.window, dtype=bool)
        valid[:length] = True
        styles = e.styles
        if not self.fixed and styles != (UNKNOWN,) and rng.random() < self.style_dropout:
            styles = ()
        return {"mel": torch.from_numpy(mel), "target": torch.from_numpy(target),
                "valid": torch.from_numpy(valid),
                "log_nps": torch.tensor(log_nps_value(e.nps), dtype=torch.float32),
                "styles": torch.from_numpy(style_vector(styles))}
