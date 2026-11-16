"""Layer statistics from train difficulties: bomb vocabulary, per-minute rates, shares."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np

from bsmaker.data.events_cache import EventCache
from bsmaker.data.layers_cache import LayerCache
from bsmaker.data.stats import _bin, _fill_empty, write_stats
from bsmaker.formats.events import FRAMES_PER_SECOND
from bsmaker.formats.layer_tokens import LIGHT_CHANNELS

BOMB_COVERAGE = 0.95
BOMB_VOCAB_CAP = 127
PERCENTILES = (25, 50, 75)
ELEMENTS = ("arcs", "chains", "walls", "bomb_groups") + tuple(
    f"light_ch{c}" for c in range(LIGHT_CHANNELS))
_COUNT_KEYS = {"arcs": "n_arcs", "chains": "n_chains", "walls": "n_walls", "bomb_groups": "n_bomb_groups"}


def compute_layer_stats(layer_cache: LayerCache, event_cache: EventCache, stats: dict) -> dict:
    """Train-split statistics; light channels only use lights-eligible difficulties."""
    edges = stats["nps_bins"]
    n_bins = len(edges) - 1
    train = [e for e in event_cache.select("train") if e.n_frames > 0]
    rates: dict[str, list[list[float]]] = {el: [[] for _ in range(n_bins)] for el in ELEMENTS}
    having: Counter = Counter()
    n_eligible = 0
    masks: Counter = Counter()
    for e in train:
        meta = layer_cache.meta(e.i)
        minutes = e.n_frames / FRAMES_PER_SECOND / 60.0
        b = _bin(edges, e.nps)
        counts = {el: meta[k] for el, k in _COUNT_KEYS.items()}
        if meta["lights_ok"]:
            n_eligible += 1
            ch = layer_cache.lights(e.i)[1]
            for c in range(LIGHT_CHANNELS):
                counts[f"light_ch{c}"] = int((ch == c).sum())
        for el, n in counts.items():
            if n > 0:
                rates[el][b].append(n / minutes)
                having[el] += 1
        masks.update(int(m) for m in layer_cache.bombs(e.i)[1])

    total = sum(masks.values())
    vocab: list[int] = []
    covered = 0
    for mask, cnt in sorted(masks.items(), key=lambda kv: (-kv[1], kv[0])):
        if covered >= BOMB_COVERAGE * total or len(vocab) >= BOMB_VOCAB_CAP:
            break
        vocab.append(mask)
        covered += cnt

    out_rates: dict[str, list[list[float]]] = {}
    for el, bins in rates.items():
        per = [[float(np.percentile(v, p)) for p in PERCENTILES] if v else None for v in bins]
        cols = [_fill_empty([row[j] if row else None for row in per]) for j in range(len(PERCENTILES))]
        out_rates[el] = [[cols[j][b] for j in range(len(PERCENTILES))] for b in range(n_bins)]
    share = {el: having[el] / (n_eligible if el.startswith("light_ch") else len(train))
             if (n_eligible if el.startswith("light_ch") else len(train)) else 0.0 for el in ELEMENTS}
    return {"nps_bins": list(edges), "bomb_vocab": vocab,
            "bomb_coverage": covered / total if total else 0.0,
            "rates": out_rates, "share": share}


def write_layer_stats(stats: dict, path: Path) -> None:
    write_stats(stats, path)


def load_layer_stats(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def level_rate(stats: dict, element: str, nps: float, level: int) -> float:
    """Per-minute rate for a level: 0 = off, 1/2/3 = 25th/50th/75th percentile of the nps bin."""
    if level == 0:
        return 0.0
    if level not in (1, 2, 3):
        raise ValueError(f"level must be 0-3, got {level}")
    return float(stats["rates"][element][_bin(stats["nps_bins"], nps)][level - 1])
