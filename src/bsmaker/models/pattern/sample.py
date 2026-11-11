"""Sample notes from the pattern model while the playability mask removes every rule-breaking token."""
from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from bsmaker.formats.events import FRAMES_PER_SECOND
from bsmaker.formats.model import Note
from bsmaker.formats.tokens import NONE_TOKEN, decode_token
from bsmaker.models.common import load_checkpoint, log_nps_value, style_vector
from bsmaker.models.pattern.dataset import _gaps, event_inputs
from bsmaker.models.pattern.model import BOS_TOKEN, N_NEXT, PatternConfig, PatternModel
from bsmaker.playability.checker import advance, allowed
from bsmaker.playability.config import PlayabilityConfig
from bsmaker.playability.state import initial_state


@dataclass
class LoadedPattern:
    model: PatternModel
    config: PatternConfig
    stats: dict


def load_pattern(path: Path, device: str) -> LoadedPattern:
    payload = load_checkpoint(path, map_location=device)
    if not payload.get("stats"):
        raise ValueError(f"{path}: checkpoint has no mel stats; it cannot be used for sampling")
    config = PatternConfig(**payload["model_config"])
    model = PatternModel(config)
    model.load_state_dict(payload["model"])
    model.to(device).eval()
    return LoadedPattern(model, config, payload["stats"])


@dataclass(frozen=True)
class SampleReport:
    n_events: int
    n_skipped: int
    n_single_hand_fallback: int
    masked_mass_mean: float
    n_frame_collisions: int = 0


def window_gaps(kept_times: Sequence[float], cur_time: float, future_times: Sequence[float],
                context: int) -> np.ndarray:
    """[L, 5] gap rows for the model input: the last `context - 1` kept events plus the current one.

    Gaps to the previous event are measured to the last KEPT event (0 when none); gaps to the next
    events run through the current event and then the planned future times.
    """
    k = min(len(kept_times), context - 1)
    extra = 1 if len(kept_times) > k else 0
    before = list(kept_times[len(kept_times) - k - extra:])
    combined = np.asarray(before + [cur_time] + list(future_times[:N_NEXT]), dtype=np.float64)
    return _gaps(combined)[extra:extra + k + 1]


def _top_p_probs(probs: torch.Tensor, top_p: float) -> torch.Tensor:
    """Keep the smallest set of most probable tokens whose mass reaches `top_p` (>= 1 token)."""
    sorted_p, order = torch.sort(probs, descending=True)
    before = torch.cumsum(sorted_p, dim=0) - sorted_p
    keep = before < top_p
    keep[0] = True
    out = torch.zeros_like(probs)
    out[order[keep]] = sorted_p[keep]
    return out / out.sum()


def _pick(logits: torch.Tensor, mask: np.ndarray, temperature: float, top_p: float,
          gen: torch.Generator, masses: list[float]) -> int:
    """Sample one allowed token from CPU float32 `logits`; record the mass the mask removed."""
    m = torch.from_numpy(mask)
    full = torch.softmax(logits, dim=0)
    masses.append(float(full[~m].sum()))
    scaled = torch.where(m, logits / temperature, torch.tensor(float("-inf")))
    probs = _top_p_probs(torch.softmax(scaled, dim=0), top_p)
    return int(torch.multinomial(probs, 1, generator=gen))


def sample_notes(loaded: LoadedPattern, event_times_s: np.ndarray, mel: np.ndarray, nps: float,
                 styles: Iterable[str], seed: int, cfg: PlayabilityConfig, device: str,
                 temperature: float = 0.9, top_p: float = 0.95) -> tuple[list[Note], SampleReport]:
    """Notes for the planned events; no sampled note breaks a hard rule of `cfg` (the mask profile).

    The model works on the mel frame grid; the mask must hold both on frame times (what `to_events`
    and `check` see) and on the exact planned times (what the game plays), so two playability
    states are kept and their masks are ANDed. Returned notes carry the exact planned times.
    Events landing on an already used frame (first planned time wins) are counted as frame
    collisions, not as skipped.
    """
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    planned_in = np.sort(np.asarray(event_times_s, dtype=np.float64))
    n_events = len(planned_in)
    all_frames = np.round(planned_in * FRAMES_PER_SECOND).astype(np.int64)
    frames, first = np.unique(all_frames, return_index=True)
    times = frames.astype(np.float64) / FRAMES_PER_SECOND
    exact = planned_in[first]
    n_collisions = n_events - len(times)
    n_skipped = 0
    n_fallback = 0
    masses: list[float] = []
    notes: list[Note] = []

    ctx = loaded.config.context
    dev = torch.device(device)
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed))
    slices_all, _ = event_inputs(times, mel, loaded.stats)
    slices_t = torch.from_numpy(slices_all)
    log_nps = torch.full((1,), log_nps_value(nps), device=dev)
    sty = torch.from_numpy(style_vector(tuple(styles))).to(dev).unsqueeze(0)
    model = loaded.model

    state = initial_state()
    state_x = initial_state()
    kept_idx: list[int] = []        # indices into `times` of kept events
    kept_tok: list[tuple[int, int]] = []
    with torch.inference_mode():
        for i, t in enumerate(times.tolist()):
            k = min(len(kept_idx), ctx - 1)
            win = kept_idx[len(kept_idx) - k:] + [i]
            gaps = window_gaps([times[j] for j in kept_idx], t, times[i + 1:i + 1 + N_NEXT],
                               ctx)
            pre = kept_tok[-k - 1] if len(kept_tok) > k else (BOS_TOKEN, BOS_TOKEN)
            prev = [pre] + kept_tok[len(kept_tok) - k:]
            prev_l = torch.tensor([[p[0] for p in prev]], device=dev)
            prev_r = torch.tensor([[p[1] for p in prev]], device=dev)
            h = model.encode(slices_t[win].unsqueeze(0).to(dev),
                             torch.from_numpy(gaps).unsqueeze(0).to(dev), prev_l, prev_r,
                             log_nps, sty)[:, -1]
            left_lg = model.left_logits(h)[0].float().cpu()
            te = float(exact[i])
            mask_l = allowed(state, t, 0, cfg) & allowed(state_x, te, 0, cfg)
            left = _pick(left_lg, mask_l, temperature, top_p, gen, masses)
            right_lg = model.right_logits(h, torch.tensor([left], device=dev))[0].float().cpu()
            mask_r = (allowed(state, t, 1, cfg, other_token=left)
                      & allowed(state_x, te, 1, cfg, other_token=left))
            right = _pick(right_lg, mask_r, temperature, top_p, gen, masses)
            if left == NONE_TOKEN and right == NONE_TOKEN:
                mask_r = mask_r.copy()
                mask_r[NONE_TOKEN] = False
                if not mask_r.any():
                    n_skipped += 1
                    continue
                right = _pick(right_lg, mask_r, temperature, top_p, gen, masses)
                n_fallback += 1
            for color, token in ((0, left), (1, right)):
                if token != NONE_TOKEN:
                    x, y, d = decode_token(token)
                    notes.append(Note(te, 0.0, x, y, color, d))
            state = advance(state, t, left, right)
            state_x = advance(state_x, te, left, right)
            kept_idx.append(i)
            kept_tok.append((left, right))
    report = SampleReport(n_events, n_skipped, n_fallback,
                          float(np.mean(masses)) if masses else 0.0, n_collisions)
    return notes, report
