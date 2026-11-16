"""Sample notes from the pattern model while the playability mask removes every rule-breaking token."""
from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from bsmaker.errors import GenerateCancelled

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


# Sampler defaults. Sharper settings (0.9 / 0.95) collapsed the output to up/down cuts: 93% of
# notes versus 51% in human maps, diagonals 6% versus 38% (val, 20 songs, human event times).
# With these values up/down is 64%, diagonals 25%, and soft-rule rates stay at the human level.
DEFAULT_TEMPERATURE = 1.05
DEFAULT_TOP_P = 0.99


@dataclass(frozen=True)
class DoubleBias:
    """Extra logit added to the right hand's NONE token when the left hand plays a note.

    It pushes against doubles (both hands on one event) and is larger where the model overdoes
    them: right after another double (`per_run` per consecutive double, at most `run_cap`) and
    at short gaps to the previous event (`short` below 0.25 s, `very_short` more below 0.15 s).
    The mask is applied afterwards, so a token the mask removed stays impossible.
    """
    base: float = 0.0
    short: float = 0.0
    very_short: float = 0.0
    per_run: float = 0.0
    run_cap: int = 4

    def __post_init__(self) -> None:
        for name in ("base", "short", "very_short", "per_run"):
            v = getattr(self, name)
            if not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0:
                raise ValueError(f"DoubleBias.{name} must be a finite number >= 0, got {v!r}")
        if not isinstance(self.run_cap, int) or isinstance(self.run_cap, bool) or self.run_cap < 0:
            raise ValueError(f"DoubleBias.run_cap must be an integer >= 0, got {self.run_cap!r}")

    def at(self, gap_s: float, run: int) -> float:
        """Bias for an event `gap_s` seconds after the previous kept event, after `run` doubles."""
        b = self.base + self.per_run * min(run, self.run_cap)
        if gap_s < 0.25:
            b += self.short
        if gap_s < 0.15:
            b += self.very_short
        return b


# Val, 60 difficulties, human event times, seed 0: doubles 24.9% -> 17.2% (human 16.6%), runs of
# 4+ doubles 521 -> 214 (human 277), left-hand share of notes 0.494 -> 0.501, tv_row_dir 0.061 -> 0.064.
DEFAULT_DOUBLE_BIAS = DoubleBias(base=0.0, short=0.5, very_short=1.5, per_run=0.25, run_cap=4)


@dataclass(frozen=True)
class DotBrake:
    """Negative logit on the dot tokens of a hand after that hand's consecutive dot notes.

    `per_dot` is subtracted per consecutive dot note of the hand beyond the first `free` ones, for
    at most `cap` of them. Events where the hand plays nothing neither extend nor reset the run.
    Applied before the mask, so masked tokens stay masked and dots stay possible when the mask
    leaves only dots.
    """
    per_dot: float = 0.0
    cap: int = 4
    free: int = 0

    def __post_init__(self) -> None:
        v = self.per_dot
        if (not isinstance(v, (int, float)) or isinstance(v, bool) or not math.isfinite(v)
                or v < 0):
            raise ValueError(f"DotBrake.per_dot must be a finite number >= 0, got {v!r}")
        for name in ("cap", "free"):
            n = getattr(self, name)
            if not isinstance(n, int) or isinstance(n, bool) or n < 0:
                raise ValueError(f"DotBrake.{name} must be an integer >= 0, got {n!r}")

    def penalty(self, run: int) -> float:
        return self.per_dot * min(max(run - self.free, 0), self.cap)


DOT_DIRECTION = 8
DOT_TOKENS = tuple(1 + x * 27 + y * 9 + DOT_DIRECTION for x in range(4) for y in range(3))
_DOT_SET = frozenset(DOT_TOKENS)
_DOT_IDX = list(DOT_TOKENS)
# GHOST seed 0 (nps 6): 39% dots, runs up to 109 -> 4.6%, longest run 6. Val, 60 difficulties: dots 2.9% (human
# 3.9%), dot runs of 4+ 90 (human 105), doubles .172 -> .169, tv_row_dir .064 -> .069.
DEFAULT_DOT_BRAKE = DotBrake(per_dot=2.0, cap=3, free=3)


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
                 temperature: float = DEFAULT_TEMPERATURE,
                 top_p: float = DEFAULT_TOP_P,
                 double_bias: DoubleBias | None = DEFAULT_DOUBLE_BIAS,
                 dot_brake: DotBrake | None = DEFAULT_DOT_BRAKE,
                 is_cancelled: Callable[[], bool] | None = None,
                 on_progress: Callable[[int, int], None] | None = None) -> tuple[list[Note], SampleReport]:
    """Notes for the planned events; no sampled note breaks a hard rule of `cfg` (the mask profile).

    The model works on the mel frame grid; the mask must hold both on frame times (what `to_events`
    and `check` see) and on the exact planned times (what the game plays), so two playability
    states are kept and their masks are ANDed. Returned notes carry the exact planned times.
    Events landing on an already used frame (first planned time wins) are counted as frame
    collisions, not as skipped. `double_bias=None` and `dot_brake=None` switch the double-note calibration and the dot-run brake
    off.
    """
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if is_cancelled is not None and is_cancelled():
        raise GenerateCancelled("cancelled")
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
    dot_run = [0, 0]                # consecutive dot notes per hand
    run = 0                         # consecutive doubles ending at the last kept event
    total_times = len(times)
    with torch.inference_mode():
        for i, t in enumerate(times.tolist()):
            if i % 32 == 0 and is_cancelled is not None and is_cancelled():
                raise GenerateCancelled("cancelled")
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
            pen_l = dot_brake.penalty(dot_run[0]) if dot_brake is not None else 0.0
            if pen_l:
                left_lg = left_lg.clone()
                left_lg[_DOT_IDX] -= pen_l
            left = _pick(left_lg, mask_l, temperature, top_p, gen, masses)
            right_lg = model.right_logits(h, torch.tensor([left], device=dev))[0].float().cpu()
            pen_r = dot_brake.penalty(dot_run[1]) if dot_brake is not None else 0.0
            if pen_r:
                right_lg = right_lg.clone()
                right_lg[_DOT_IDX] -= pen_r
            if double_bias is not None and left != NONE_TOKEN:
                gap = t - times[kept_idx[-1]] if kept_idx else math.inf
                bias = double_bias.at(gap, run)
                if bias:
                    right_lg = right_lg.clone()
                    right_lg[NONE_TOKEN] += bias
            mask_r = (allowed(state, t, 1, cfg, other_token=left)
                      & allowed(state_x, te, 1, cfg, other_token=left))
            right = _pick(right_lg, mask_r, temperature, top_p, gen, masses)
            if left == NONE_TOKEN and right == NONE_TOKEN:
                mask_r = mask_r.copy()
                mask_r[NONE_TOKEN] = False
                if not mask_r.any():
                    n_skipped += 1
                    if on_progress is not None and (i % 16 == 0 or i + 1 == total_times):
                        on_progress(i + 1, total_times)
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
            for hand, token in enumerate((left, right)):
                if token != NONE_TOKEN:
                    dot_run[hand] = dot_run[hand] + 1 if token in _DOT_SET else 0
            run = run + 1 if left != NONE_TOKEN and right != NONE_TOKEN else 0
            if on_progress is not None and (i % 16 == 0 or i + 1 == total_times):
                on_progress(i + 1, total_times)
    if is_cancelled is not None and is_cancelled():
        raise GenerateCancelled("cancelled")
    report = SampleReport(n_events, n_skipped, n_fallback,
                          float(np.mean(masses)) if masses else 0.0, n_collisions)
    return notes, report
