"""Turn timing-model frame probabilities into event times at a requested density."""
from __future__ import annotations

import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from bsmaker.audio.decode import SAMPLE_RATE
from bsmaker.audio.mel import FRAMES_PER_SECOND, HOP
from bsmaker.data.stats import lookup
from bsmaker.models.common import (
    load_checkpoint, log_nps_value, normalise_mel, style_vector)
from bsmaker.models.timing.metrics import onset_f1
from bsmaker.models.timing.model import TimingConfig, TimingModel

MIN_FIRST_EVENT_S = 1.5
BATCH = 16


def frame_probs(model: TimingModel, mel: np.ndarray, nps: float, styles: Iterable[str],
                stats: dict, device: str, hop: int | None = None) -> np.ndarray:
    """Per-frame event probability for a whole song (float32 `[frames]`).

    Overlapping full windows (hop = window // 2) are averaged with a triangular weight; a song
    shorter than one window is zero-padded after normalisation and the padding is discarded.
    """
    window = model.cfg.window
    hop = hop or max(1, window // 2)
    n = int(mel.shape[0])
    if n == 0:
        return np.zeros(0, dtype=np.float32)
    x = normalise_mel(mel, stats)
    if n < window:
        x = np.concatenate([x, np.zeros((window - n, x.shape[1]), np.float32)])
    starts = list(range(0, max(n - window, 0) + 1, hop))
    if starts[-1] + window < n:
        starts.append(n - window)
    dev = torch.device(device)
    log_nps = torch.full((1,), log_nps_value(nps), device=dev)
    sty = torch.from_numpy(style_vector(styles)).to(dev).unsqueeze(0)
    idx = np.arange(window)
    tri = (1 + np.minimum(idx, window - 1 - idx)).astype(np.float64)
    acc = np.zeros(max(n, window), dtype=np.float64)
    wsum = np.zeros_like(acc)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16,
                                                enabled=dev.type == "cuda"):
        for b in range(0, len(starts), BATCH):
            chunk = starts[b:b + BATCH]
            batch = torch.from_numpy(np.stack([x[s:s + window] for s in chunk])).to(dev)
            logits = model(batch, log_nps.expand(len(chunk)), sty.expand(len(chunk), -1))
            probs = torch.sigmoid(logits.float()).cpu().numpy().astype(np.float64)
            for s, p in zip(chunk, probs):
                acc[s:s + window] += p * tri
                wsum[s:s + window] += tri
    return (acc[:n] / wsum[:n]).astype(np.float32)


def _candidates(probs: np.ndarray) -> np.ndarray:
    """Indices of local maxima (plateaus: the first frame), highest first."""
    p = np.asarray(probs, dtype=np.float32)
    left = np.concatenate([[-np.inf], p[:-1]]).astype(np.float32)
    right = np.concatenate([p[1:], [-np.inf]]).astype(np.float32)
    cand = np.nonzero((p >= left) & (p > right))[0]
    return cand[np.argsort(-p[cand], kind="stable")]


def _greedy(n: int, cand: np.ndarray, min_gap_frames: int) -> np.ndarray:
    """Accepted peaks in descending height order (a peak within the gap of a higher one is out)."""
    blocked = np.zeros(n, dtype=bool)
    kept = []
    g = max(min_gap_frames, 1)
    for i in cand.tolist():
        if blocked[i]:
            continue
        kept.append(i)
        blocked[max(0, i - g + 1):i + g] = True
    return np.asarray(kept, dtype=np.int64)


def pick_peaks(probs: np.ndarray, threshold: float, min_gap_frames: int = 3) -> np.ndarray:
    """Frame indices of local maxima above `threshold`, at least `min_gap_frames` apart."""
    probs = np.asarray(probs, dtype=np.float32)
    cand = _candidates(probs)
    cand = cand[probs[cand] > threshold]
    return np.sort(_greedy(len(probs), cand, min_gap_frames))


def threshold_for_count(probs: np.ndarray, n_events: int, min_gap_frames: int = 3) -> float:
    """Threshold in [0, 1] whose peak count is closest to `n_events` (30-step binary search).

    Greedy suppression runs in descending height order, so the peaks for any threshold are a
    prefix of one run; it is done once and each step only counts that prefix.
    """
    probs = np.asarray(probs, dtype=np.float32)
    kept = _greedy(len(probs), _candidates(probs), min_gap_frames)
    neg = np.sort(-probs[kept])  # ascending negated heights

    def count(t: float) -> int:  # peaks with height > t
        return int(np.searchsorted(neg, -t, side="left"))

    lo, hi = 0.0, 1.0
    best_t, best_d = 0.0, abs(count(0.0) - n_events)
    for _ in range(30):
        mid = (lo + hi) / 2
        c = count(mid)
        d = abs(c - n_events)
        if d < best_d or (d == best_d and mid < best_t):
            best_t, best_d = mid, d
        if c > n_events:
            lo = mid
        else:
            hi = mid
    return best_t


@dataclass
class LoadedTiming:
    model: TimingModel
    config: TimingConfig
    stats: dict


def load_timing(path: Path, device: str) -> LoadedTiming:
    payload = load_checkpoint(path, map_location=device)
    config = TimingConfig(**payload["model_config"])
    model = TimingModel(config)
    model.load_state_dict(payload["model"])
    model.to(device).eval()
    return LoadedTiming(model, config, payload.get("stats", {}))


def predict_events(loaded: LoadedTiming, mel: np.ndarray, nps: float, styles: Iterable[str],
                   device: str, count_scale: float = 1.0) -> np.ndarray:
    """Event times in seconds (float64, sorted) at the density `nps`."""
    stats = loaded.stats
    probs = frame_probs(loaded.model, mel, nps, tuple(styles), stats, device)
    duration = len(mel) / FRAMES_PER_SECOND
    target = (nps * duration * float(stats["coverage_median"])
              / (1.0 + lookup(stats, "double_share_by_bin", nps)) * count_scale)
    thr = threshold_for_count(probs, int(round(target)))
    times = pick_peaks(probs, thr).astype(np.float64) * HOP / SAMPLE_RATE
    return times[times >= MIN_FIRST_EVENT_S]


def timing_better(new: dict, best: dict | None) -> bool:
    """Higher val F1 wins; a record without `f1` (old, loss-only checkpoints) is always worse."""
    if best is None or "f1" not in best:
        return True
    return "f1" in new and new["f1"] > best["f1"]


def make_timing_eval(cache, entries: Sequence, features_dir: Path, stats: dict, base_eval,
                     n_songs: int = 64):
    """eval_fn: the window loss from `base_eval` plus whole-song onset P/R/F1 on fixed entries.

    Each decode uses the human event count, so F1 measures placement only.
    """
    entries = list(entries)
    if len(entries) > n_songs:
        entries = [entries[i] for i in np.linspace(0, len(entries) - 1, n_songs).astype(int)]

    def eval_fn(model, loader) -> dict[str, float]:
        metrics = dict(base_eval(model, loader))
        device = str(next(model.parameters()).device)
        t0 = time.monotonic()
        scores = []
        for e in entries:
            mel = np.load(Path(features_dir) / f"{e.map_key}.npy", allow_pickle=False)
            mel = mel[:min(int(e.n_frames), mel.shape[0])]
            true = np.asarray(cache.events(e.i)[0], dtype=np.float64)
            probs = frame_probs(model, mel, e.nps, e.styles, stats, device)
            thr = threshold_for_count(probs, len(true))
            pred = pick_peaks(probs, thr).astype(np.float64) * HOP / SAMPLE_RATE
            scores.append(onset_f1(pred, true))
        arr = np.asarray(scores, dtype=np.float64).reshape(-1, 3)
        metrics.update(precision=float(arr[:, 0].mean()), recall=float(arr[:, 1].mean()),
                       f1=float(arr[:, 2].mean()))
        metrics["f1_seconds"] = time.monotonic() - t0
        return metrics

    return eval_fn
