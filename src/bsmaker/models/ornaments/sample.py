"""Applies the ornament tagger to generated notes: picks chains and arc starts."""
from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from bsmaker.audio.decode import SAMPLE_RATE
from bsmaker.audio.mel import HOP
from bsmaker.data.layer_stats import level_rate
from bsmaker.formats.layer_tokens import ARC_LENGTH_BEATS
from bsmaker.formats.model import Arc, Chain, Note
from bsmaker.formats.tokens import encode_token
from bsmaker.models.common import load_checkpoint, log_nps_value
from bsmaker.models.ornaments.dataset import note_inputs
from bsmaker.models.ornaments.model import OrnamentConfig, OrnamentModel

log = logging.getLogger(__name__)

TEMPERATURE = 0.8
# replaced by playability.layers at the join
MIN_CHAIN_GAP_S = 0.25
CHAIN_TAIL_BEATS = 1.0 / 24.0
_DOT = 8
_DIR_VEC = {0: (0, 1), 1: (0, -1), 2: (-1, 0), 3: (1, 0), 4: (-1, 1), 5: (1, 1), 6: (-1, -1),
            7: (1, -1)}


@dataclass(frozen=True)
class LoadedOrnaments:
    model: OrnamentModel
    config: OrnamentConfig
    stats: dict


@dataclass(frozen=True)
class OrnamentReport:
    arcs_requested: int
    chains_requested: int
    arcs_placed: int
    chains_placed: int
    rejected_chain: int  # notes that cannot be a chain by the hard rules
    rejected_arc: int  # notes that cannot start an arc by the hard rules


def load_ornaments(path: Path, device: str) -> LoadedOrnaments:
    payload = load_checkpoint(path, map_location=device)
    if not payload.get("stats"):
        raise ValueError(f"{path}: checkpoint has no mel stats; it cannot be used for sampling")
    config = OrnamentConfig(**payload["model_config"])
    model = OrnamentModel(config)
    model.load_state_dict(payload["model"])
    model.to(device).eval()
    return LoadedOrnaments(model, config, payload["stats"])


# replaced by playability.layers at the join
def _next_same_hand(notes: Sequence[Note]) -> list[float | None]:
    nxt: list[float | None] = [None] * len(notes)
    seen: dict[int, float] = {}
    for i in range(len(notes) - 1, -1, -1):
        nxt[i] = seen.get(notes[i].color)
        seen[notes[i].color] = notes[i].time_s
    return nxt


# replaced by playability.layers at the join
def _chain_tail_cell(note: Note) -> tuple[int, int] | None:
    """Cell one step along the cut, clipped to the grid; None for dots or when clipping empties it."""
    if note.direction not in _DIR_VEC:
        return None
    dx, dy = _DIR_VEC[note.direction]
    tx, ty = min(max(note.x + dx, 0), 3), min(max(note.y + dy, 0), 2)
    return None if (tx, ty) == (note.x, note.y) else (tx, ty)


# replaced by playability.layers at the join
def _chain_ok(notes: Sequence[Note], i: int, next_t: float | None) -> bool:
    n = notes[i]
    if next_t is None or next_t - n.time_s < MIN_CHAIN_GAP_S:
        return False
    tail = _chain_tail_cell(n)
    if tail is None:
        return False
    cells = {(n.x, n.y), tail}
    for j, o in enumerate(notes):
        if j != i and n.time_s <= o.time_s <= next_t and (o.x, o.y) in cells:
            return False
    return True


# replaced by playability.layers at the join
def _arc_buckets(note: Note, next_t: float | None, bpm: float) -> list[int]:
    """Allowed arc length classes (1-6): the tail must not come after the same hand's next note."""
    return [k + 1 for k, beats in enumerate(ARC_LENGTH_BEATS)
            if next_t is None or note.time_s + beats * 60.0 / bpm <= next_t]


@torch.no_grad()
def _probabilities(loaded: LoadedOrnaments, notes: Sequence[Note], mel: np.ndarray, nps: float,
                   device: str) -> dict[str, np.ndarray]:
    """Per-note softmax of every head (chunks of `context` notes, true gaps across chunks)."""
    t = np.array([n.time_s for n in notes], dtype=np.float64)
    hand = np.array([n.color for n in notes], dtype=np.int64)
    token = np.array([encode_token(n.x, n.y, n.direction) for n in notes], dtype=np.int64)
    slices, gaps = note_inputs(t, hand, mel, loaded.stats)
    ctx = loaded.config.context
    outs: dict[str, list[np.ndarray]] = {k: [] for k in ("chain", "arc_len", "arc_cell", "arc_dir")}
    log_nps = torch.tensor([log_nps_value(nps)], dtype=torch.float32, device=device)
    for a in range(0, len(notes), ctx):
        sl = slice(a, a + ctx)
        n = len(t[sl])
        out = loaded.model(
            torch.from_numpy(slices[sl]).to(device)[None], torch.from_numpy(gaps[sl]).to(device)[None],
            torch.from_numpy(hand[sl]).to(device)[None], torch.from_numpy(token[sl]).to(device)[None],
            log_nps, torch.ones(1, n, dtype=torch.bool, device=device))
        for k in outs:
            outs[k].append(out[k][0].float().cpu().numpy())
    return {k: np.concatenate(v) for k, v in outs.items()}


def _softmax(x: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    z = x / temperature
    z = np.exp(z - z.max(-1, keepdims=True))
    return z / z.sum(-1, keepdims=True)


def sample_ornaments(loaded: LoadedOrnaments, notes: Sequence[Note], mel: np.ndarray, nps: float,
                     level: int, layer_stats: dict, bpm: float, seed: int,
                     device: str) -> tuple[list[Arc], list[Chain], OrnamentReport]:
    """Chains and arcs for `notes`; counts follow the level's human rate for the map's nps.

    Every returned object satisfies the hard rules; when too few notes qualify the result has
    fewer objects than requested (a warning is logged), never a rule violation.
    """
    notes = sorted(notes, key=lambda n: (n.time_s, n.color))
    if level == 0 or not notes or not bpm > 0:
        return [], [], OrnamentReport(0, 0, 0, 0, 0, 0)
    mel_s = len(mel) * HOP / SAMPLE_RATE
    minutes = max(notes[-1].time_s, mel_s) / 60.0
    want_arcs = int(round(level_rate(layer_stats, "arcs", nps, level) * minutes))
    want_chains = int(round(level_rate(layer_stats, "chains", nps, level) * minutes))

    next_t = _next_same_hand(notes)
    chain_ok = [_chain_ok(notes, i, next_t[i]) for i in range(len(notes))]
    arc_buckets = [_arc_buckets(notes[i], next_t[i], bpm) for i in range(len(notes))]
    rejected_chain = len(notes) - sum(chain_ok)
    rejected_arc = sum(1 for b in arc_buckets if not b)
    if want_arcs + want_chains == 0:
        return [], [], OrnamentReport(0, 0, 0, 0, rejected_chain, rejected_arc)

    probs = _probabilities(loaded, notes, mel, nps, device)
    p_chain = _softmax(probs["chain"])
    p_arc = _softmax(probs["arc_len"])
    cands = []  # (-p, time, kind, index)
    for i, n in enumerate(notes):
        if chain_ok[i]:
            cands.append((-(1.0 - float(p_chain[i, 0])), n.time_s, 0, i))
        if arc_buckets[i]:
            cands.append((-(1.0 - float(p_arc[i, 0])), n.time_s, 1, i))
    cands.sort()

    want = {0: want_chains, 1: want_arcs}
    used: set[int] = set()
    picked: dict[int, list[int]] = {0: [], 1: []}
    for _, _, kind, i in cands:
        if i in used or len(picked[kind]) >= want[kind]:
            continue
        used.add(i)
        picked[kind].append(i)

    rng = np.random.default_rng(seed)
    beat_s = 60.0 / bpm
    chains = []
    for i in sorted(picked[0]):
        n = notes[i]
        tx, ty = _chain_tail_cell(n)  # type: ignore[misc]  # chain_ok guarantees a tail cell
        segments = 3 + int(np.argmax(p_chain[i, 1:]))
        tail_s = n.time_s + CHAIN_TAIL_BEATS * beat_s
        chains.append(Chain(n.time_s, n.time_s / beat_s, n.color, n.x, n.y, n.direction, tail_s,
                            tail_s / beat_s, tx, ty, segments, 1.0))
    arcs = []
    for i in sorted(picked[1]):
        n = notes[i]
        allowed = arc_buckets[i]
        cls = 1 + int(np.argmax(p_arc[i, 1:]))
        cls = min(cls, max(allowed))  # allowed classes are a prefix 1..k: shorter always fits
        cell = int(rng.choice(12, p=_softmax(probs["arc_cell"][i], TEMPERATURE)))
        direction = int(rng.choice(9, p=_softmax(probs["arc_dir"][i], TEMPERATURE)))
        tail_s = n.time_s + ARC_LENGTH_BEATS[cls - 1] * 60.0 / bpm
        arcs.append(Arc(n.time_s, n.time_s / beat_s, n.color, n.x, n.y, n.direction, tail_s,
                        tail_s / beat_s, cell % 4, cell // 4, direction))
    report = OrnamentReport(want_arcs, want_chains, len(arcs), len(chains), rejected_chain,
                            rejected_arc)
    if len(arcs) < want_arcs or len(chains) < want_chains:
        log.warning("ornaments: placed %d/%d arcs and %d/%d chains (rules leave too few notes)",
                    len(arcs), want_arcs, len(chains), want_chains)
    return arcs, chains, report
