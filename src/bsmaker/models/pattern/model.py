"""Causal transformer that picks the left and right note token at every event."""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from bsmaker.formats.tokens import VOCAB_SIZE
from bsmaker.models.common import Conditioning

SLICE_FRAMES = 35  # +-0.2 s of mel around each event
N_NEXT = 4
BOS_TOKEN = VOCAB_SIZE  # input-only id: "no previous event"
N_MELS = 80


@dataclass(frozen=True)
class PatternConfig:
    d_model: int = 384
    n_layers: int = 8
    n_heads: int = 6
    context: int = 256
    dropout: float = 0.1


class _SliceEncoder(nn.Module):
    """Small conv net: [N, 1, 35, 80] -> [N, d]."""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1), nn.GELU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.GELU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.GELU(),
            nn.AdaptiveAvgPool2d((2, 4)))
        self.proj = nn.Linear(64 * 2 * 4, d_model)

    def forward(self, x: Tensor) -> Tensor:
        return self.proj(self.convs(x).flatten(1))


class PatternModel(nn.Module):
    def __init__(self, cfg: PatternConfig) -> None:
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model
        self.audio = _SliceEncoder(d)
        self.gap = nn.Sequential(nn.Linear(1 + N_NEXT, d), nn.GELU(), nn.Linear(d, d))
        self.prev_left = nn.Embedding(VOCAB_SIZE + 1, d)
        self.prev_right = nn.Embedding(VOCAB_SIZE + 1, d)
        self.cond = Conditioning(d)
        self.pos = nn.Embedding(cfg.context + 1, d)
        layer = nn.TransformerEncoderLayer(d, cfg.n_heads, 4 * d, cfg.dropout, activation="gelu",
                                           batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, cfg.n_layers, norm=nn.LayerNorm(d),
                                             enable_nested_tensor=False)
        self.left_head = nn.Linear(d, VOCAB_SIZE)
        self.left_emb = nn.Embedding(VOCAB_SIZE, d)
        self.right_head = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, VOCAB_SIZE))

    def encode(self, mel_slices: Tensor, gaps: Tensor, prev_left: Tensor, prev_right: Tensor,
               log_nps: Tensor, styles: Tensor) -> Tensor:
        b, n = prev_left.shape
        if n > self.cfg.context:
            raise ValueError(f"sequence length {n} exceeds context {self.cfg.context}")
        a = self.audio(mel_slices.reshape(b * n, 1, SLICE_FRAMES, N_MELS)).reshape(b, n, -1)
        x = a + self.gap(gaps) + self.prev_left(prev_left) + self.prev_right(prev_right)
        x = torch.cat([self.cond(log_nps, styles).unsqueeze(1).to(x.dtype), x], dim=1)
        x = x + self.pos(torch.arange(n + 1, device=x.device))
        mask = torch.triu(torch.ones(n + 1, n + 1, dtype=torch.bool, device=x.device), 1)
        return self.encoder(x, mask=mask)[:, 1:]

    def left_logits(self, h: Tensor) -> Tensor:
        return self.left_head(h)

    def right_logits(self, h: Tensor, left: Tensor) -> Tensor:
        return self.right_head(h + self.left_emb(left))


def pattern_ce(model: PatternModel, batch: dict) -> tuple[Tensor, Tensor, Tensor]:
    """(sum of left CE, sum of right CE, count) over valid positions; right uses the true left."""
    h = model.encode(batch["mel_slices"], batch["gaps"], batch["prev_left"],
                     batch["prev_right"], batch["log_nps"], batch["styles"])
    valid = batch["valid"].float()
    ce_l = F.cross_entropy(model.left_logits(h).float().transpose(1, 2), batch["left"],
                           reduction="none")
    ce_r = F.cross_entropy(model.right_logits(h, batch["left"]).float().transpose(1, 2),
                           batch["right"], reduction="none")
    return (ce_l * valid).sum(), (ce_r * valid).sum(), valid.sum()


def pattern_loss(model: PatternModel, batch: dict) -> tuple[Tensor, dict[str, float]]:
    sl, sr, n = pattern_ce(model, batch)
    n = n.clamp(min=1.0)
    ce_l, ce_r = sl / n, sr / n
    return ce_l + ce_r, {"ce_left": ce_l.item(), "ce_right": ce_r.item()}


def pattern_step(model: nn.Module, batch: dict) -> Tensor:
    sl, sr, n = pattern_ce(model, batch)  # type: ignore[arg-type]
    return (sl + sr) / n.clamp(min=1.0)


@torch.no_grad()
def pattern_eval(model: nn.Module, loader) -> dict[str, float]:
    device = next(model.parameters()).device
    tl = tr = cnt = 0.0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        sl, sr, n = pattern_ce(model, batch)  # type: ignore[arg-type]
        tl, tr, cnt = tl + float(sl), tr + float(sr), cnt + float(n)
    cnt = max(cnt, 1.0)
    return {"ce": (tl + tr) / cnt, "ce_left": tl / cnt, "ce_right": tr / cnt}


def pattern_better(new: dict, best: dict | None) -> bool:
    return best is None or new["ce"] < best["ce"]
