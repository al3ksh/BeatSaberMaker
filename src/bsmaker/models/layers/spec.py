"""Layer specifications shared by the generic "when" and "what" models."""
from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import partial

import numpy as np

from bsmaker.data.layers_cache import LayerCache
from bsmaker.formats.layer_tokens import LIGHT_CHANNELS, LIGHT_VOCAB, WALL_VOCAB

Extract = Callable[[LayerCache, int], tuple[np.ndarray, np.ndarray, np.ndarray]]


@dataclass(frozen=True)
class LayerSpec:
    """Channels, vocabulary and data access of one layer (picklable: module-level callables)."""
    name: str
    n_channels: int
    vocab_size: int
    channel_tokens: tuple[tuple[int, ...], ...]
    min_gap_frames: tuple[int, ...]
    rate_elements: tuple[str, ...]
    eligible: Callable[[dict], bool]
    extract: Extract


def _sorted(t: np.ndarray, ch: np.ndarray, tok: np.ndarray):
    order = np.lexsort((ch, t))  # time first, then channel (stable)
    return (np.asarray(t, dtype=np.float64)[order], np.asarray(ch, dtype=np.int64)[order],
            np.asarray(tok, dtype=np.int64)[order])


def _lights_ok(meta: dict) -> bool:
    return bool(meta["lights_ok"])


def _lights_extract(cache: LayerCache, i: int):
    t, ch, tok = cache.lights(i)
    return _sorted(t, ch, tok)


def _decor_ok(meta: dict) -> bool:
    return meta["n_walls"] > 0 or meta["n_bomb_groups"] > 0


def _decor_extract(masks: tuple[int, ...], cache: LayerCache, i: int):
    index = {int(m): WALL_VOCAB + k for k, m in enumerate(masks)}
    wt, side, wtok = cache.walls(i)
    bt, bmask = cache.bombs(i)
    keep = np.array([int(m) in index for m in bmask], dtype=bool)
    btok = np.array([index[int(m)] for m in bmask[keep]], dtype=np.int64)
    t = np.concatenate([np.asarray(wt, np.float64), np.asarray(bt, np.float64)[keep]])
    ch = np.concatenate([np.asarray(side, np.int64), np.full(len(btok), 2, np.int64)])
    tok = np.concatenate([np.asarray(wtok, np.int64), btok])
    return _sorted(t, ch, tok)


def _lights_channel_tokens() -> tuple[tuple[int, ...], ...]:
    out = []
    for c in range(LIGHT_CHANNELS):
        if c <= 4:
            out.append(tuple(range(7)))
        elif c <= 6:
            out.append((17,))
        else:
            out.append(tuple(range(7, 17)))
    return tuple(out)


LIGHTS = LayerSpec(
    "lights", LIGHT_CHANNELS, LIGHT_VOCAB, _lights_channel_tokens(), (2,) * LIGHT_CHANNELS,
    tuple(f"light_ch{c}" for c in range(LIGHT_CHANNELS)), _lights_ok, _lights_extract)


def decor_spec(layer_stats: dict) -> LayerSpec:
    """Walls (left, right) and bombs; the bomb vocabulary comes from `layer_stats["bomb_vocab"]`."""
    masks = tuple(int(m) for m in layer_stats["bomb_vocab"])
    walls = tuple(range(WALL_VOCAB))
    bombs = tuple(range(WALL_VOCAB, WALL_VOCAB + len(masks)))
    return LayerSpec("decor", 3, WALL_VOCAB + len(masks), (walls, walls, bombs), (3, 3, 3),
                     ("walls", "walls", "bomb_groups"), _decor_ok, partial(_decor_extract, masks))


def get_spec(name: str, layer_stats: dict | None = None) -> LayerSpec:
    if name == "lights":
        return LIGHTS
    if name == "decor":
        if layer_stats is None:
            raise ValueError("the decor layer needs layer_stats (bomb vocabulary)")
        return decor_spec(layer_stats)
    raise ValueError(f"unknown layer {name!r}; expected lights or decor")


def channel_mask(spec: LayerSpec) -> np.ndarray:
    """Bool [n_channels, vocab_size]: tokens allowed on each channel."""
    m = np.zeros((spec.n_channels, spec.vocab_size), dtype=bool)
    for c, toks in enumerate(spec.channel_tokens):
        m[c, list(toks)] = True
    return m


def eligible_entries(entries: Sequence, layer_cache: LayerCache, spec: LayerSpec,
                     split: str) -> list:
    return [e for e in entries if e.split == split and e.n_frames > 0
            and spec.eligible(layer_cache.meta(e.i))]


def evenly_spaced(items: Sequence, cap: int) -> list:
    if len(items) <= cap:
        return list(items)
    return [items[i] for i in np.linspace(0, len(items) - 1, cap).astype(int)]
