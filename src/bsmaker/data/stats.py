"""Dataset statistics (train split only): density ranks, NJS, doubles, mel normalisation."""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from bsmaker.data.events_cache import EventCache
from bsmaker.formats.events import FRAMES_PER_SECOND
from bsmaker.formats.tokens import NONE_TOKEN

NPS_BINS = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 7, 9, 12]
RANKS = ("Easy", "Normal", "Hard", "Expert", "ExpertPlus")
_RANK_BY_KEY = {r.lower(): r for r in RANKS} | {"expert+": "ExpertPlus"}


def _bin(edges: list[float], nps: float) -> int:
    b = int(np.searchsorted(edges, nps, side="right")) - 1
    return min(max(b, 0), len(edges) - 2)


def _fill_empty(values: list[float | None]) -> list[float]:
    known = [i for i, v in enumerate(values) if v is not None]
    if not known:
        return [0.0] * len(values)
    out: list[float] = []
    for i in range(len(values)):
        v = values[min(known, key=lambda k: (abs(k - i), k))]
        assert v is not None
        out.append(v)
    return out


def _median(v: list[float]) -> float | None:
    return float(np.median(v)) if v else None


def compute_stats(cache: EventCache, features_dir: Path, sample_maps: int = 500,
                  seed: int = 0) -> dict:
    """Train-split statistics. A rank with no train data is omitted from `rank_median_nps`."""
    train = cache.select("train")
    n_bins = len(NPS_BINS) - 1
    by_rank: dict[str, list[float]] = {r: [] for r in RANKS}
    njs: list[list[float]] = [[] for _ in range(n_bins)]
    off: list[list[float]] = [[] for _ in range(n_bins)]
    doubles = [0] * n_bins
    totals = [0] * n_bins
    coverage: list[float] = []
    for e in train:
        rank = _RANK_BY_KEY.get(e.rank.lower())
        if rank:
            by_rank[rank].append(e.nps)
        b = _bin(NPS_BINS, e.nps)
        njs[b].append(e.njs)
        off[b].append(e.njs_offset)
        t, left, right = cache.events(e.i)
        doubles[b] += int(((left != NONE_TOKEN) & (right != NONE_TOKEN)).sum())
        totals[b] += len(t)
        if e.n_frames > 0 and len(t):
            coverage.append(float(t[-1] - t[0]) / (e.n_frames / FRAMES_PER_SECOND))

    keys = sorted({e.map_key for e in train})
    if len(keys) > sample_maps:
        rng = np.random.default_rng(seed)
        keys = sorted(rng.choice(keys, size=sample_maps, replace=False).tolist())
    s = s2 = 0.0
    n = 0
    for k in keys:
        arr = np.load(Path(features_dir) / f"{k}.npy", allow_pickle=False).astype(np.float32)
        s += float(arr.sum(dtype=np.float64))
        s2 += float(np.square(arr, dtype=np.float64).sum())
        n += arr.size
    mean = s / n if n else 0.0
    std = float(np.sqrt(max(s2 / n - mean * mean, 0.0))) if n else 1.0

    return {
        "rank_median_nps": {r: _median(v) for r, v in by_rank.items() if v},
        "nps_bins": NPS_BINS,
        "njs_by_bin": _fill_empty([_median(v) for v in njs]),
        "njs_offset_by_bin": _fill_empty([_median(v) for v in off]),
        "double_share_by_bin": _fill_empty([d / t if t else None for d, t in zip(doubles, totals)]),
        "coverage_median": float(np.median(coverage)) if coverage else 0.0,
        "mel_mean": mean,
        "mel_std": std,
    }


def write_stats(stats: dict, path: Path) -> None:
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def load_stats(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def lookup(stats: dict, table: str, nps: float) -> float:
    if table not in ("njs_by_bin", "njs_offset_by_bin", "double_share_by_bin"):
        raise ValueError(f"unknown table {table!r}")
    values = stats[table]
    return float(values[_bin(stats["nps_bins"], nps)])
