"""Multi-channel frame tagger: when do the events of a layer happen."""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import Dataset

from bsmaker.audio.decode import SAMPLE_RATE
from bsmaker.audio.mel import FRAMES_PER_SECOND, HOP, N_MELS
from bsmaker.data.events_cache import Entry, EventCache
from bsmaker.data.layers_cache import LayerCache
from bsmaker.formats.model import Note
from bsmaker.formats.tokens import NONE_TOKEN
from bsmaker.models.common import (
    STYLE_DIM, load_checkpoint, log_nps_value, normalise_mel)
from bsmaker.models.layers.spec import LayerSpec, get_spec
from bsmaker.models.timing.dataset import _frames, _targets_from_frames, frame_targets
from bsmaker.models.timing.decode import (
    BATCH, MIN_FIRST_EVENT_S, pick_peaks, threshold_for_count)
from bsmaker.models.timing.metrics import onset_f1
from bsmaker.models.timing.model import TimingConfig, TimingModel, _sinusoid, timing_loss

F1_TOL_S = 0.025
N_NOTE_CHANNELS = 2


@dataclass(frozen=True)
class WhenConfig:
    n_mels: int = 80
    n_channels: int = 9
    d_model: int = 256
    n_layers: int = 6
    n_heads: int = 4
    window: int = 1024
    dropout: float = 0.1


def note_track(left_times: np.ndarray, right_times: np.ndarray, n_frames: int) -> np.ndarray:
    """[n_frames, 2] float32: left and right hand note onsets (1.0 / 0.25 on the neighbours)."""
    out = np.zeros((n_frames, N_NOTE_CHANNELS), dtype=np.float32)
    out[:, 0] = frame_targets(np.asarray(left_times, dtype=np.float64), n_frames)
    out[:, 1] = frame_targets(np.asarray(right_times, dtype=np.float64), n_frames)
    return out


def note_track_from_notes(notes: Sequence[Note], n_frames: int) -> np.ndarray:
    left = [n.time_s for n in notes if n.color == 0]
    right = [n.time_s for n in notes if n.color == 1]
    return note_track(np.asarray(left, dtype=np.float64), np.asarray(right, dtype=np.float64),
                      n_frames)


def hand_times(events: tuple[np.ndarray, np.ndarray, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """(left, right) note times of an event-cache entry."""
    t, left, right = events
    t = np.asarray(t, dtype=np.float64)
    return t[np.asarray(left) != NONE_TOKEN], t[np.asarray(right) != NONE_TOKEN]


class WhenModel(nn.Module):
    """The timing model's stem and encoder; mel plus two note channels in, C logits per frame."""

    def __init__(self, cfg: WhenConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.base = TimingModel(TimingConfig(
            n_mels=cfg.n_mels + N_NOTE_CHANNELS, d_model=cfg.d_model, n_layers=cfg.n_layers,
            n_heads=cfg.n_heads, window=cfg.window, dropout=cfg.dropout))
        self.base.head = nn.Linear(cfg.d_model, cfg.n_channels)

    def forward(self, mel: Tensor, notes: Tensor, log_nps: Tensor) -> Tensor:
        """mel [B, T, n_mels], notes [B, T, 2] -> logits [B, T, C]."""
        m = self.base
        b, t, _ = mel.shape
        x = m.stem(torch.cat([mel, notes.to(mel.dtype)], dim=-1).unsqueeze(1))
        x = m.proj(x.permute(0, 2, 1, 3).reshape(b, t, -1))
        x = x + _sinusoid(t, self.cfg.d_model, x.device).to(x.dtype)
        styles = torch.zeros(b, STYLE_DIM, device=x.device, dtype=x.dtype)
        x = x + m.cond(log_nps, styles).unsqueeze(1)
        return m.head(m.norm(m.encoder(x)))


class WhenWindows(Dataset):
    """One random window of one difficulty per item (`fixed=True`: always the same window).

    Pass `layers_dir` / `events_dir` so the object pickles without the loaded caches.
    """

    def __init__(self, layer_cache: LayerCache | None, event_cache: EventCache | None,
                 entries: Sequence[Entry], features_dir: Path, stats: dict, spec: LayerSpec,
                 window: int, seed: int, fixed: bool = False, layers_dir: Path | None = None,
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
        self.window = window
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
        mel_all = np.load(self.features_dir / f"{e.map_key}.npy", mmap_mode="r",
                          allow_pickle=False)
        total = min(int(e.n_frames), mel_all.shape[0])
        start = int(rng.integers(0, total - self.window + 1)) if total > self.window else 0
        length = max(0, min(self.window, total - start))
        mel = np.zeros((self.window, N_MELS), dtype=np.float32)
        mel[:length] = normalise_mel(mel_all[start:start + length], self.stats)
        left, right = hand_times(self._events.events(e.i))
        notes = np.zeros((self.window, N_NOTE_CHANNELS), dtype=np.float32)
        for h, times in enumerate((left, right)):
            notes[:length, h] = _targets_from_frames(_frames(times) - start, length)
        t, ch, _ = self.spec.extract(self._layers, e.i)
        frames = _frames(t) - start
        target = np.zeros((self.window, self.spec.n_channels), dtype=np.float32)
        for c in range(self.spec.n_channels):
            target[:length, c] = _targets_from_frames(frames[ch == c], length)
        valid = np.zeros(self.window, dtype=bool)
        valid[:length] = True
        return {"mel": torch.from_numpy(mel), "notes": torch.from_numpy(notes),
                "target": torch.from_numpy(target), "valid": torch.from_numpy(valid),
                "log_nps": torch.tensor(log_nps_value(e.nps), dtype=torch.float32)}


def _loss(logits: Tensor, batch: dict) -> Tensor:
    valid = batch["valid"].unsqueeze(-1).expand_as(batch["target"])
    return timing_loss(logits, batch["target"], pos_weight=5.0, valid=valid)


def when_step(model: nn.Module, batch: dict) -> Tensor:
    return _loss(model(batch["mel"], batch["notes"], batch["log_nps"]), batch)


def _hits(pred_s: np.ndarray, true_s: np.ndarray) -> int:
    """One-to-one matches within F1_TOL_S (recovered from the shared onset metric's recall)."""
    if len(pred_s) == 0 or len(true_s) == 0:
        return 0
    _, recall, _ = onset_f1(pred_s, true_s, F1_TOL_S)
    return int(round(recall * len(true_s)))


@torch.no_grad()
def when_eval(model: nn.Module, loader, min_gap_frames: Sequence[int] | None = None
              ) -> dict[str, float]:
    """Window loss and mean per-channel F1 (+-25 ms), decoding each window and channel with
    the human count; channels without a human event in any window are left out of the mean."""
    device = next(model.parameters()).device
    n_ch = model.cfg.n_channels  # type: ignore[attr-defined]
    gaps = list(min_gap_frames) if min_gap_frames is not None else [2] * n_ch
    total = n = 0.0
    hits = np.zeros(n_ch)
    n_pred = np.zeros(n_ch)
    n_true = np.zeros(n_ch)
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        logits = model(batch["mel"], batch["notes"], batch["log_nps"])
        count = float(batch["valid"].sum()) * n_ch
        total += float(_loss(logits, batch)) * count
        n += count
        probs = torch.sigmoid(logits.float()).cpu().numpy()
        target = batch["target"].cpu().numpy()
        valid = batch["valid"].cpu().numpy()
        for b in range(probs.shape[0]):
            length = int(valid[b].sum())
            for c in range(n_ch):
                true = np.nonzero(target[b, :length, c] >= 0.99)[0]
                if len(true) == 0:
                    continue
                p = probs[b, :length, c]
                pred = pick_peaks(p, threshold_for_count(p, len(true), gaps[c]), gaps[c])
                hits[c] += _hits(pred / FRAMES_PER_SECOND, true / FRAMES_PER_SECOND)
                n_pred[c] += len(pred)
                n_true[c] += len(true)
    f1s = []
    for c in range(n_ch):
        if n_true[c] == 0:
            continue
        p = hits[c] / n_pred[c] if n_pred[c] else 0.0
        r = hits[c] / n_true[c]
        f1s.append(2 * p * r / (p + r) if hits[c] else 0.0)
    return {"loss": total / max(n, 1.0), "f1": float(np.mean(f1s)) if f1s else 0.0}


def when_better(new: dict, best: dict | None) -> bool:
    return best is None or "f1" not in best or new["f1"] > best["f1"]


@dataclass
class LoadedWhen:
    model: WhenModel
    config: WhenConfig
    stats: dict
    layer: str
    bomb_masks: list[int] | None

    @property
    def spec(self) -> LayerSpec:
        return get_spec(self.layer, {"bomb_vocab": self.bomb_masks or []})


def load_when(path: Path, device: str) -> LoadedWhen:
    payload = load_checkpoint(path, map_location=device)
    if not payload.get("stats"):
        raise ValueError(f"{path}: checkpoint has no mel stats; it cannot be used for sampling")
    config = WhenConfig(**payload["model_config"])
    model = WhenModel(config)
    model.load_state_dict(payload["model"])
    model.to(device).eval()
    stats = payload["stats"]
    layer = (payload.get("train_options") or {}).get("layer", "")
    return LoadedWhen(model, config, stats, layer, stats.get("bomb_masks"))


def layer_frame_probs(model: WhenModel, mel: np.ndarray, track: np.ndarray, nps: float,
                      stats: dict, device: str) -> np.ndarray:
    """[frames, C] event probabilities for a whole song (overlapping windows, triangular blend)."""
    window = model.cfg.window
    hop = max(1, window // 2)
    n = int(mel.shape[0])
    n_ch = model.cfg.n_channels
    if n == 0:
        return np.zeros((0, n_ch), dtype=np.float32)
    x = normalise_mel(mel, stats)
    tr = np.asarray(track, dtype=np.float32)
    if n < window:
        x = np.concatenate([x, np.zeros((window - n, x.shape[1]), np.float32)])
        tr = np.concatenate([tr, np.zeros((window - n, tr.shape[1]), np.float32)])
    starts = list(range(0, max(n - window, 0) + 1, hop))
    if starts[-1] + window < n:
        starts.append(n - window)
    dev = torch.device(device)
    idx = np.arange(window)
    tri = (1 + np.minimum(idx, window - 1 - idx)).astype(np.float64)
    acc = np.zeros((max(n, window), n_ch))
    wsum = np.zeros(max(n, window))
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16,
                                                enabled=dev.type == "cuda"):
        for b in range(0, len(starts), BATCH):
            chunk = starts[b:b + BATCH]
            mb = torch.from_numpy(np.stack([x[s:s + window] for s in chunk])).to(dev)
            tb = torch.from_numpy(np.stack([tr[s:s + window] for s in chunk])).to(dev)
            ln = torch.full((len(chunk),), log_nps_value(nps), device=dev)
            probs = torch.sigmoid(model(mb, tb, ln).float()).cpu().numpy().astype(np.float64)
            for s, p in zip(chunk, probs):
                acc[s:s + window] += p * tri[:, None]
                wsum[s:s + window] += tri
    return (acc[:n] / wsum[:n, None]).astype(np.float32)


def predict_layer_events(loaded: LoadedWhen, mel: np.ndarray, notes: Sequence[Note], nps: float,
                         counts: Sequence[int], device: str) -> list[np.ndarray]:
    """Per channel the event times (s, float64, sorted): about `counts[c]` peaks of that
    channel's curve, at least the spec's min gap apart and not before MIN_FIRST_EVENT_S."""
    cfg = loaded.config
    if len(counts) != cfg.n_channels:
        raise ValueError(f"expected {cfg.n_channels} counts, got {len(counts)}")
    gaps = loaded.spec.min_gap_frames if loaded.layer else (2,) * cfg.n_channels
    n = int(mel.shape[0])
    empty = [np.zeros(0, dtype=np.float64) for _ in counts]
    if n == 0 or not any(int(c) > 0 for c in counts):
        return empty
    track = note_track_from_notes(notes, n)
    probs = layer_frame_probs(loaded.model, mel, track, nps, loaded.stats, device)
    lead = int(round(MIN_FIRST_EVENT_S * FRAMES_PER_SECOND))
    out = []
    for c, count in enumerate(counts):
        if int(count) <= 0:
            out.append(empty[c])
            continue
        p = probs[:, c].copy()
        p[:lead] = 0.0
        thr = threshold_for_count(p, int(count), gaps[c])
        t = pick_peaks(p, thr, gaps[c]).astype(np.float64) * HOP / SAMPLE_RATE
        out.append(t[t >= MIN_FIRST_EVENT_S])
    return out
