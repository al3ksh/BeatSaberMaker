"""Causal sequence model: what is each event of a layer (channel-restricted token)."""
from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.data import Dataset

from bsmaker.audio.mel import N_MELS
from bsmaker.data.events_cache import Entry, EventCache
from bsmaker.data.layers_cache import LayerCache
from bsmaker.formats.model import Note
from bsmaker.models.common import load_checkpoint, log_nps_value
from bsmaker.models.layers.spec import LayerSpec, get_spec
from bsmaker.models.layers.when import hand_times
from bsmaker.models.pattern.dataset import _gaps, _slices
from bsmaker.models.pattern.model import N_NEXT, SLICE_FRAMES, _SliceEncoder
from bsmaker.models.pattern.sample import _pick, window_gaps

IGNORE = -100
NOTE_WINDOW_S = 0.25


@dataclass(frozen=True)
class WhatConfig:
    n_channels: int
    vocab_size: int
    d_model: int = 384
    n_layers: int = 8
    n_heads: int = 6
    context: int = 256
    dropout: float = 0.1


def note_context(times_s: np.ndarray, left_s: np.ndarray, right_s: np.ndarray) -> np.ndarray:
    """[L, 2] log1p of the left / right note counts within +-0.25 s of each event."""
    t = np.asarray(times_s, dtype=np.float64)
    out = np.zeros((len(t), 2), dtype=np.float32)
    for h, notes in enumerate((left_s, right_s)):
        n = np.sort(np.asarray(notes, dtype=np.float64))
        if len(n):
            hi = np.searchsorted(n, t + NOTE_WINDOW_S, side="right")
            lo = np.searchsorted(n, t - NOTE_WINDOW_S, side="left")
            out[:, h] = np.log1p(hi - lo)
    return out


def hand_times_from_notes(notes: Sequence[Note]) -> tuple[np.ndarray, np.ndarray]:
    return (np.array([n.time_s for n in notes if n.color == 0], dtype=np.float64),
            np.array([n.time_s for n in notes if n.color == 1], dtype=np.float64))


class WhatModel(nn.Module):
    def __init__(self, cfg: WhatConfig, mask: np.ndarray | None = None) -> None:
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model
        self.audio = _SliceEncoder(d)
        self.gap = nn.Sequential(nn.Linear(1 + N_NEXT, d), nn.GELU(), nn.Linear(d, d))
        self.notes = nn.Sequential(nn.Linear(2, d), nn.GELU(), nn.Linear(d, d))
        self.channel = nn.Embedding(cfg.n_channels, d)
        self.prev = nn.Embedding(cfg.vocab_size + 1, d)  # last id = BOS
        self.nps = nn.Sequential(nn.Linear(1, d), nn.GELU(), nn.Linear(d, d))
        self.pos = nn.Embedding(cfg.context + 1, d)
        layer = nn.TransformerEncoderLayer(d, cfg.n_heads, 4 * d, cfg.dropout, activation="gelu",
                                           batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, cfg.n_layers, norm=nn.LayerNorm(d),
                                             enable_nested_tensor=False)
        self.head = nn.Linear(d, cfg.vocab_size)
        if mask is None:
            mask = np.ones((cfg.n_channels, cfg.vocab_size), dtype=bool)
        self.register_buffer("channel_mask", torch.as_tensor(np.asarray(mask, dtype=bool)))

    @property
    def bos(self) -> int:
        return self.cfg.vocab_size

    def forward(self, mel_slices: Tensor, gaps: Tensor, channels: Tensor, prev: Tensor,
                notes: Tensor, log_nps: Tensor) -> Tensor:
        """Logits [B, N, vocab]; tokens outside each event's channel are -inf."""
        b, n = prev.shape
        if n > self.cfg.context:
            raise ValueError(f"sequence length {n} exceeds context {self.cfg.context}")
        a = self.audio(mel_slices.reshape(b * n, 1, SLICE_FRAMES, N_MELS)).reshape(b, n, -1)
        x = (a + self.gap(gaps) + self.notes(notes.to(a.dtype)) + self.channel(channels)
             + self.prev(prev))
        cond = self.nps(log_nps.unsqueeze(-1).to(x.dtype)).unsqueeze(1)
        x = torch.cat([cond, x], dim=1) + self.pos(torch.arange(n + 1, device=x.device))
        causal = torch.triu(torch.ones(n + 1, n + 1, dtype=torch.bool, device=x.device), 1)
        h = self.encoder(x, mask=causal)[:, 1:]
        logits = self.head(h).float()
        return logits.masked_fill(~self.channel_mask[channels], float("-inf"))


class WhatSequences(Dataset):
    """One random run of up to `context` events of one difficulty per item."""

    def __init__(self, layer_cache: LayerCache | None, event_cache: EventCache | None,
                 entries: Sequence[Entry], features_dir: Path, stats: dict, spec: LayerSpec,
                 context: int, seed: int, fixed: bool = False, layers_dir: Path | None = None,
                 events_dir: Path | None = None) -> None:
        if (layer_cache is None and layers_dir is None) or (
                event_cache is None and events_dir is None):
            raise ValueError("give the caches or their directories")
        self._layers, self._events = layer_cache, event_cache
        self.layers_dir = Path(layers_dir) if layers_dir is not None else None
        self.events_dir = Path(events_dir) if events_dir is not None else None
        self.entries = list(entries)
        self.features_dir = Path(features_dir)
        self.stats = {"mel_mean": stats["mel_mean"], "mel_std": stats["mel_std"]}
        self.spec = spec
        self.context = context
        self.seed = seed
        self.fixed = fixed
        self._rng: np.random.Generator | None = None

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state["_rng"] = None
        if self.layers_dir is not None:
            state["_layers"] = None
        if self.events_dir is not None:
            state["_events"] = None
        return state

    def _process_rng(self) -> np.random.Generator:
        if self._rng is None:
            info = torch.utils.data.get_worker_info()
            base = info.seed if info is not None else torch.initial_seed()
            self._rng = np.random.default_rng([self.seed, int(base)])
        return self._rng

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        if self._layers is None:
            self._layers = LayerCache.load(self.layers_dir)
        if self._events is None:
            self._events = EventCache.load(self.events_dir)
        e = self.entries[idx]
        rng = np.random.default_rng([self.seed, idx]) if self.fixed else self._process_rng()
        times, chan, tok = self.spec.extract(self._layers, e.i)
        total, c = len(times), self.context
        start = int(rng.integers(0, total - c + 1)) if total > c else 0
        n = min(c, total - start)
        sl = slice(start, start + n)
        bos = self.spec.vocab_size
        gaps = np.zeros((c, 1 + N_NEXT), dtype=np.float32)
        slices = np.zeros((c, SLICE_FRAMES, N_MELS), dtype=np.float32)
        ctx = np.zeros((c, 2), dtype=np.float32)
        channels = np.zeros(c, dtype=np.int64)
        prev = np.full(c, bos, dtype=np.int64)
        target = np.full(c, IGNORE, dtype=np.int64)
        valid = np.zeros(c, dtype=bool)
        if n:
            gaps[:n] = _gaps(times)[sl]
            mel = np.load(self.features_dir / f"{e.map_key}.npy", mmap_mode="r",
                          allow_pickle=False)
            mel = mel[:min(int(e.n_frames), mel.shape[0])]
            slices[:n] = _slices(times[sl], mel, self.stats)
            left, right = hand_times(self._events.events(e.i))
            ctx[:n] = note_context(times[sl], left, right)
            channels[:n] = chan[sl]
            target[:n] = tok[sl]
            prev[1:n] = tok[start:start + n - 1]
            if start > 0:
                prev[0] = tok[start - 1]
            valid[:n] = True
        return {"mel_slices": torch.from_numpy(slices), "gaps": torch.from_numpy(gaps),
                "channels": torch.from_numpy(channels), "prev": torch.from_numpy(prev),
                "notes": torch.from_numpy(ctx), "target": torch.from_numpy(target),
                "valid": torch.from_numpy(valid),
                "log_nps": torch.tensor(log_nps_value(e.nps), dtype=torch.float32)}


def _ce_sum(model: nn.Module, batch: dict) -> tuple[Tensor, Tensor]:
    logits = model(batch["mel_slices"], batch["gaps"], batch["channels"], batch["prev"],
                   batch["notes"], batch["log_nps"])
    ce = F.cross_entropy(logits.transpose(1, 2), batch["target"], ignore_index=IGNORE,
                         reduction="sum")
    return ce, batch["valid"].sum().float()


def what_step(model: nn.Module, batch: dict) -> Tensor:
    ce, n = _ce_sum(model, batch)
    return ce / n.clamp(min=1.0)


@torch.no_grad()
def what_eval(model: nn.Module, loader) -> dict[str, float]:
    device = next(model.parameters()).device
    total = count = 0.0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        ce, n = _ce_sum(model, batch)
        total, count = total + float(ce), count + float(n)
    return {"ce": total / max(count, 1.0)}


def what_better(new: dict, best: dict | None) -> bool:
    return best is None or new["ce"] < best["ce"]


@dataclass
class LoadedWhat:
    model: WhatModel
    config: WhatConfig
    stats: dict
    layer: str
    bomb_masks: list[int] | None

    @property
    def spec(self) -> LayerSpec:
        return get_spec(self.layer, {"bomb_vocab": self.bomb_masks or []})


def load_what(path: Path, device: str) -> LoadedWhat:
    payload = load_checkpoint(path, map_location=device)
    if not payload.get("stats"):
        raise ValueError(f"{path}: checkpoint has no mel stats; it cannot be used for sampling")
    config = WhatConfig(**payload["model_config"])
    model = WhatModel(config)
    model.load_state_dict(payload["model"])
    model.to(device).eval()
    stats = payload["stats"]
    layer = (payload.get("train_options") or {}).get("layer", "")
    return LoadedWhat(model, config, stats, layer, stats.get("bomb_masks"))


def sample_layer_values(loaded: LoadedWhat, times: Sequence[float], channels: Sequence[int],
                        mel: np.ndarray, notes: Sequence[Note], nps: float, seed: int,
                        device: str,
                        allowed: Callable[[int, float, int], np.ndarray] | None = None,
                        temperature: float = 1.0, top_p: float = 0.99
                        ) -> tuple[np.ndarray, int]:
    """Token per event (int64, in the order given; -1 for a skipped event) and the number of
    events skipped because `allowed(event_index, time_s, channel)` left no token on the channel.

    `event_index` is the index into the given arrays. Events are processed in (time, channel)
    order; skipped events are left out of the model context.
    """
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    cfg = loaded.config
    t_in = np.asarray(times, dtype=np.float64)
    ch_in = np.asarray(channels, dtype=np.int64)
    if t_in.shape != ch_in.shape or t_in.ndim != 1:
        raise ValueError("times and channels must be 1-D arrays of equal length")
    if len(ch_in) and (ch_in.min() < 0 or ch_in.max() >= cfg.n_channels):
        raise ValueError("channel out of range")
    out = np.full(len(t_in), -1, dtype=np.int64)
    if len(t_in) == 0:
        return out, 0
    order = np.lexsort((ch_in, t_in))
    t, ch = t_in[order], ch_in[order]
    dev = torch.device(device)
    model = loaded.model
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed))
    slices = torch.from_numpy(_slices(t, mel, loaded.stats))
    left, right = hand_times_from_notes(notes)
    ctx = torch.from_numpy(note_context(t, left, right))
    chan_t = torch.from_numpy(ch)
    log_nps = torch.full((1,), log_nps_value(nps), device=dev)
    cmask = model.channel_mask.cpu().numpy()
    bos, window = model.bos, cfg.context
    kept_idx: list[int] = []
    kept_t: list[float] = []
    kept_tok: list[int] = []
    skipped = 0
    masses: list[float] = []
    with torch.inference_mode():
        for i in range(len(t)):
            mask = cmask[ch[i]].copy()
            if allowed is not None:
                extra = np.asarray(allowed(int(order[i]), float(t[i]), int(ch[i])), dtype=bool)
                if extra.shape != mask.shape:
                    raise ValueError(f"allowed() must return a mask of length {len(mask)}")
                mask &= extra
            if not mask.any():
                skipped += 1
                continue
            k = min(len(kept_idx), window - 1)
            win = kept_idx[len(kept_idx) - k:] + [i]
            gaps = window_gaps(kept_t, float(t[i]), t[i + 1:i + 1 + N_NEXT], window)
            pre = kept_tok[-k - 1] if len(kept_tok) > k else bos
            prev = torch.tensor([[pre] + kept_tok[len(kept_tok) - k:]], device=dev)
            logits = model(slices[win].unsqueeze(0).to(dev),
                           torch.from_numpy(gaps).unsqueeze(0).to(dev),
                           chan_t[win].unsqueeze(0).to(dev), prev,
                           ctx[win].unsqueeze(0).to(dev), log_nps)[0, -1].cpu()
            tok = _pick(logits, mask, temperature, top_p, gen, masses)
            out[order[i]] = tok
            kept_idx.append(i)
            kept_t.append(float(t[i]))
            kept_tok.append(tok)
    return out, skipped


