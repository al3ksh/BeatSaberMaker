"""Applies the ornament tagger to generated notes: picks chains and arc starts."""
from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import torch

from bsmaker.audio.decode import SAMPLE_RATE
from bsmaker.audio.mel import HOP
from bsmaker.data.layer_stats import level_rate
from bsmaker.errors import GenerateCancelled
from bsmaker.formats.layer_tokens import ARC_LENGTH_BEATS
from bsmaker.formats.model import Arc, Chain, Note
from bsmaker.formats.tokens import encode_token
from bsmaker.models.common import load_checkpoint, log_nps_value
from bsmaker.models.ornaments.dataset import note_inputs
from bsmaker.models.ornaments.model import OrnamentConfig, OrnamentModel
from bsmaker.playability.layers import NoteIndex, arc_ok, chain_ok

log = logging.getLogger(__name__)

TEMPERATURE = 0.8
CHAIN_TAIL_BEATS = 1.0 / 24.0
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


def _chain_tail_cell(note: Note) -> tuple[int, int] | None:
    """Cell one step along the cut, clipped to the grid; None for dots or when clipping empties it."""
    if note.direction not in _DIR_VEC:
        return None
    dx, dy = _DIR_VEC[note.direction]
    tx, ty = min(max(note.x + dx, 0), 3), min(max(note.y + dy, 0), 2)
    return None if (tx, ty) == (note.x, note.y) else (tx, ty)


def _build_chain(note: Note, tail: tuple[int, int], segments: int, beat_s: float) -> Chain:
    tail_s = note.time_s + CHAIN_TAIL_BEATS * beat_s
    return Chain(note.time_s, note.time_s / beat_s, note.color, note.x, note.y, note.direction,
                 tail_s, tail_s / beat_s, tail[0], tail[1], segments, 1.0)


def _build_arc(note: Note, length_beats: float, tx: int, ty: int, td: int, beat_s: float) -> Arc:
    tail_s = note.time_s + length_beats * beat_s
    return Arc(note.time_s, note.time_s / beat_s, note.color, note.x, note.y, note.direction,
               tail_s, tail_s / beat_s, tx, ty, td)


SNAP_MAX_BEATS = 1.0
SNAP_MAX_CELLS = 1


def snap_arc_tail(arc: Arc, notes: Sequence[Note]) -> Arc:
    """Snap the tail onto the same hand's next note when it would end just short of it.

    The next note is the hand's first colour note strictly after the head. If the tail is at most
    1 beat before it (never after) and within 1 cell (Chebyshev), the tail takes that note's beat,
    cell and direction. Otherwise the free-ending arc is returned unchanged.
    """
    nxt = next((n for n in sorted(notes, key=lambda n: n.time_s)
                if n.color == arc.color and n.time_s > arc.time_s + 1e-6), None)
    if nxt is None:
        return arc
    gap = nxt.beat - arc.tail_beat
    if not -1e-6 <= gap <= SNAP_MAX_BEATS + 1e-6:
        return arc
    if max(abs(nxt.x - arc.tx), abs(nxt.y - arc.ty)) > SNAP_MAX_CELLS:
        return arc
    return replace(arc, tail_time_s=nxt.time_s, tail_beat=nxt.beat, tx=nxt.x, ty=nxt.y, td=nxt.direction)


@torch.no_grad()
def _probabilities(loaded: LoadedOrnaments, notes: Sequence[Note], mel: np.ndarray, nps: float,
                   device: str,
                   is_cancelled: Callable[[], bool] | None = None) -> dict[str, np.ndarray]:
    """Per-note softmax of every head (chunks of `context` notes, true gaps across chunks)."""
    if is_cancelled is not None and is_cancelled():
        raise GenerateCancelled("cancelled")
    t = np.array([n.time_s for n in notes], dtype=np.float64)
    hand = np.array([n.color for n in notes], dtype=np.int64)
    token = np.array([encode_token(n.x, n.y, n.direction) for n in notes], dtype=np.int64)
    slices, gaps = note_inputs(t, hand, mel, loaded.stats)
    ctx = loaded.config.context
    outs: dict[str, list[np.ndarray]] = {k: [] for k in ("chain", "arc_len", "arc_cell", "arc_dir")}
    log_nps = torch.tensor([log_nps_value(nps)], dtype=torch.float32, device=device)
    for a in range(0, len(notes), ctx):
        if is_cancelled is not None and is_cancelled():
            raise GenerateCancelled("cancelled")
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
    z = np.asarray(x, dtype=np.float64) / temperature
    z = np.exp(z - z.max(-1, keepdims=True))
    return z / z.sum(-1, keepdims=True)


def sample_ornaments(loaded: LoadedOrnaments, notes: Sequence[Note], mel: np.ndarray, nps: float,
                     level: int, layer_stats: dict, bpm: float, seed: int,
                     device: str,
                     is_cancelled: Callable[[], bool] | None = None) -> tuple[list[Arc], list[Chain], OrnamentReport]:
    """Chains and arcs for `notes`; counts follow the level's human rate for the map's nps.

    Every returned object satisfies the hard rules; when too few notes qualify the result has
    fewer objects than requested (a warning is logged), never a rule violation.
    """
    if is_cancelled is not None and is_cancelled():
        raise GenerateCancelled("cancelled")
    notes = sorted(notes, key=lambda n: (n.time_s, n.color))
    if level == 0 or not notes or not bpm > 0:
        return [], [], OrnamentReport(0, 0, 0, 0, 0, 0)
    mel_s = len(mel) * HOP / SAMPLE_RATE
    minutes = max(notes[-1].time_s, mel_s) / 60.0
    want_arcs = int(round(level_rate(layer_stats, "arcs", nps, level) * minutes))
    want_chains = int(round(level_rate(layer_stats, "chains", nps, level) * minutes))

    idx = NoteIndex(notes)
    beat_s = 60.0 / bpm
    tail_cells = [_chain_tail_cell(n) for n in notes]
    can_chain = []
    for i, (n, tc) in enumerate(zip(notes, tail_cells)):
        if i % 32 == 0 and is_cancelled is not None and is_cancelled():
            raise GenerateCancelled("cancelled")
        # segments=3 is a placeholder for the rule check (`chain_ok` ignores segment count).
        can_chain.append(tc is not None and chain_ok(_build_chain(n, tc, 3, beat_s), idx))
    arc_buckets = []
    for i, n in enumerate(notes):
        if i % 32 == 0 and is_cancelled is not None and is_cancelled():
            raise GenerateCancelled("cancelled")
        arc_buckets.append([
            k + 1 for k, beats in enumerate(ARC_LENGTH_BEATS)
            if arc_ok(_build_arc(n, beats, n.x, n.y, n.direction, beat_s), idx)
        ])
    rejected_chain = len(notes) - sum(can_chain)
    rejected_arc = sum(1 for b in arc_buckets if not b)
    if want_arcs + want_chains == 0:
        return [], [], OrnamentReport(0, 0, 0, 0, rejected_chain, rejected_arc)

    probs = _probabilities(loaded, notes, mel, nps, device, is_cancelled=is_cancelled)
    p_chain = _softmax(probs["chain"])
    p_arc = _softmax(probs["arc_len"])
    cands = []  # (-p, time, kind, index)
    for i, n in enumerate(notes):
        if can_chain[i]:
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
    chains = []
    for idx_pos, i in enumerate(sorted(picked[0])):
        if idx_pos % 32 == 0 and is_cancelled is not None and is_cancelled():
            raise GenerateCancelled("cancelled")
        n = notes[i]
        tc = tail_cells[i]
        if tc is None:
            continue
        segments = 3 + int(np.argmax(p_chain[i, 1:]))
        chains.append(_build_chain(n, tc, segments, beat_s))
    arcs = []
    for idx_pos, i in enumerate(sorted(picked[1])):
        if idx_pos % 32 == 0 and is_cancelled is not None and is_cancelled():
            raise GenerateCancelled("cancelled")
        n = notes[i]
        allowed = arc_buckets[i]
        cls = 1 + int(np.argmax(p_arc[i, 1:]))
        cls = min(cls, max(allowed))  # allowed classes are a prefix 1..k: shorter always fits
        cell = int(rng.choice(12, p=_softmax(probs["arc_cell"][i], TEMPERATURE)))
        direction = int(rng.choice(9, p=_softmax(probs["arc_dir"][i], TEMPERATURE)))
        arc = _build_arc(n, ARC_LENGTH_BEATS[cls - 1], cell % 4, cell // 4, direction, beat_s)
        arcs.append(snap_arc_tail(arc, notes))
    if is_cancelled is not None and is_cancelled():
        raise GenerateCancelled("cancelled")
    report = OrnamentReport(want_arcs, want_chains, len(arcs), len(chains), rejected_chain,
                            rejected_arc)
    if len(arcs) < want_arcs or len(chains) < want_chains:
        log.warning("ornaments: placed %d/%d arcs and %d/%d chains (rules leave too few notes)",
                    len(arcs), want_arcs, len(chains), want_chains)
    return arcs, chains, report
