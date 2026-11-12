"""Timing model: marks the spectrogram frames where notes fall."""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from bsmaker.models.common import Conditioning


@dataclass(frozen=True)
class TimingConfig:
    n_mels: int = 80
    d_model: int = 256
    n_layers: int = 6
    n_heads: int = 4
    window: int = 1024
    dropout: float = 0.1


def _sinusoid(length: int, dim: int, device: torch.device) -> Tensor:
    pos = torch.arange(length, device=device, dtype=torch.float32).unsqueeze(1)
    freq = torch.exp(torch.arange(0, dim, 2, device=device, dtype=torch.float32)
                     * (-math.log(10000.0) / dim))
    pe = torch.zeros(length, dim, device=device)
    pe[:, 0::2] = torch.sin(pos * freq)
    pe[:, 1::2] = torch.cos(pos * freq)[:, : dim // 2]
    return pe


class TimingModel(nn.Module):
    def __init__(self, cfg: TimingConfig) -> None:
        super().__init__()
        self.cfg = cfg
        # Convolutions stride over the frequency axis only, so the time axis is kept.
        self.stem = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.GELU(),
            nn.Conv2d(16, 32, 3, stride=(1, 2), padding=1), nn.GELU(),
            nn.Conv2d(32, 32, 3, stride=(1, 2), padding=1), nn.GELU())
        f = cfg.n_mels
        for _ in range(2):
            f = (f + 1) // 2
        self.proj = nn.Linear(32 * f, cfg.d_model)
        self.cond = Conditioning(cfg.d_model)
        layer = nn.TransformerEncoderLayer(
            cfg.d_model, cfg.n_heads, dim_feedforward=cfg.d_model * 4, dropout=cfg.dropout,
            activation="gelu", batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, cfg.n_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, 1)

    def forward(self, mel: Tensor, log_nps: Tensor, styles: Tensor) -> Tensor:
        """mel [B, T, n_mels] -> logits [B, T]."""
        b, t, _ = mel.shape
        x = self.stem(mel.unsqueeze(1))                       # [B, C, T, F']
        x = self.proj(x.permute(0, 2, 1, 3).reshape(b, t, -1))
        x = x + _sinusoid(t, self.cfg.d_model, x.device).to(x.dtype)
        x = x + self.cond(log_nps, styles).unsqueeze(1)
        x = self.norm(self.encoder(x))
        return self.head(x).squeeze(-1)


def timing_loss(logits: Tensor, targets: Tensor, pos_weight: float = 5.0,
                valid: Tensor | None = None) -> Tensor:
    """Binary cross-entropy against fuzzy targets; frames at the event itself weigh `pos_weight`.

    `valid` ([B, T] bool) excludes padded frames from the mean.
    """
    logits = logits.float()
    w = 1.0 + (pos_weight - 1.0) * (targets >= 0.99).float()
    per = F.binary_cross_entropy_with_logits(logits, targets, reduction="none") * w
    if valid is None:
        return per.mean()
    v = valid.float()
    return (per * v).sum() / v.sum().clamp_min(1.0)


def timing_step(model: nn.Module, batch: dict) -> Tensor:
    logits = model(batch["mel"], batch["log_nps"], batch["styles"])
    return timing_loss(logits, batch["target"], valid=batch["valid"])


@torch.no_grad()
def timing_eval(model: nn.Module, loader) -> dict[str, float]:
    device = next(model.parameters()).device
    total = 0.0
    n = 0.0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        logits = model(batch["mel"], batch["log_nps"], batch["styles"])
        count = float(batch["valid"].sum())
        total += float(timing_loss(logits, batch["target"], valid=batch["valid"])) * count
        n += count
    return {"loss": total / max(n, 1.0)}
