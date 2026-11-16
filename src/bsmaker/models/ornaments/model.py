"""Bidirectional transformer over a run of notes that tags chains and arc starts."""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from bsmaker.formats.tokens import VOCAB_SIZE
from bsmaker.models.pattern.model import N_MELS, SLICE_FRAMES, _SliceEncoder

N_GAPS = 4  # log1p seconds to the previous / next note of any hand, previous / next of the same hand
N_CHAIN = 6  # none, 3..7 segments
N_ARC_LEN = 7  # none, 6 length buckets
N_ARC_CELL = 12
N_ARC_DIR = 9
NONE_WEIGHT = 0.2


@dataclass(frozen=True)
class OrnamentConfig:
    d_model: int = 256
    n_layers: int = 6
    n_heads: int = 4
    context: int = 256
    dropout: float = 0.1


class OrnamentModel(nn.Module):
    def __init__(self, cfg: OrnamentConfig) -> None:
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model
        self.audio = _SliceEncoder(d)
        self.gap = nn.Sequential(nn.Linear(N_GAPS, d), nn.GELU(), nn.Linear(d, d))
        self.hand = nn.Embedding(2, d)
        self.token = nn.Embedding(VOCAB_SIZE, d)
        self.nps = nn.Sequential(nn.Linear(1, d), nn.GELU(), nn.Linear(d, d))
        self.pos = nn.Embedding(cfg.context, d)
        layer = nn.TransformerEncoderLayer(d, cfg.n_heads, 4 * d, cfg.dropout, activation="gelu",
                                           batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, cfg.n_layers, norm=nn.LayerNorm(d),
                                             enable_nested_tensor=False)
        self.chain_head = nn.Linear(d, N_CHAIN)
        self.arc_len_head = nn.Linear(d, N_ARC_LEN)
        self.arc_cell_head = nn.Linear(d, N_ARC_CELL)
        self.arc_dir_head = nn.Linear(d, N_ARC_DIR)

    def forward(self, mel_slices: Tensor, gaps: Tensor, hand: Tensor, token: Tensor,
                log_nps: Tensor, valid: Tensor) -> dict[str, Tensor]:
        b, n = hand.shape
        if n > self.cfg.context:
            raise ValueError(f"sequence length {n} exceeds context {self.cfg.context}")
        a = self.audio(mel_slices.reshape(b * n, 1, SLICE_FRAMES, N_MELS)).reshape(b, n, -1)
        x = (a + self.gap(gaps) + self.hand(hand) + self.token(token)
             + self.nps(log_nps.unsqueeze(-1)).unsqueeze(1))
        x = x + self.pos(torch.arange(n, device=x.device))
        h = self.encoder(x, src_key_padding_mask=~valid.bool())
        return {"chain": self.chain_head(h), "arc_len": self.arc_len_head(h),
                "arc_cell": self.arc_cell_head(h), "arc_dir": self.arc_dir_head(h)}


def _ce(logits: Tensor, target: Tensor, n_classes: int, none_weight: bool) -> Tensor:
    ce = F.cross_entropy(logits.float().transpose(1, 2), target, reduction="none")
    if not none_weight:
        return ce
    w = torch.ones(n_classes, device=ce.device)
    w[0] = NONE_WEIGHT
    return ce * w[target]


def _sums(model: nn.Module, batch: dict) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
    out = model(batch["mel_slices"], batch["gaps"], batch["hand"], batch["token"],
                batch["log_nps"], batch["valid"])
    valid = batch["valid"].float()
    has_arc = valid * (batch["arc_len"] > 0).float()
    geo = (_ce(out["arc_cell"], batch["arc_cell"], N_ARC_CELL, False)
           + _ce(out["arc_dir"], batch["arc_dir"], N_ARC_DIR, False)) * has_arc
    s = {"cls": ((_ce(out["chain"], batch["chain"], N_CHAIN, True)
                  + _ce(out["arc_len"], batch["arc_len"], N_ARC_LEN, True)) * valid).sum(),
         "geo": geo.sum(), "n": valid.sum(), "n_arc": has_arc.sum()}
    return s, out


def _loss(s: dict[str, Tensor]) -> Tensor:
    return s["cls"] / s["n"].clamp(min=1.0) + s["geo"] / s["n_arc"].clamp(min=1.0)


def ornament_step(model: nn.Module, batch: dict) -> Tensor:
    return _loss(_sums(model, batch)[0])


@torch.no_grad()
def ornament_eval(model: nn.Module, loader) -> dict[str, float]:
    device = next(model.parameters()).device
    tot = {"cls": 0.0, "geo": 0.0, "n": 0.0, "n_arc": 0.0}
    counts = {"chain": [0, 0, 0], "arc_len": [0, 0, 0]}  # tp, fp, fn
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        s, out = _sums(model, batch)
        for k in tot:
            tot[k] += float(s[k])
        valid = batch["valid"].bool()
        for key, c in counts.items():
            pred = (1.0 - out[key].float().softmax(-1)[..., 0]) >= 0.5
            true = batch[key] > 0
            c[0] += int((pred & true & valid).sum())
            c[1] += int((pred & ~true & valid).sum())
            c[2] += int((~pred & true & valid).sum())

    def f1(c: list[int]) -> float:
        return 2 * c[0] / (2 * c[0] + c[1] + c[2]) if c[0] else 0.0

    loss = tot["cls"] / max(tot["n"], 1.0) + tot["geo"] / max(tot["n_arc"], 1.0)
    return {"loss": loss, "chain_f1": f1(counts["chain"]), "arc_f1": f1(counts["arc_len"])}


def ornament_better(new: dict, best: dict | None) -> bool:
    return best is None or new["loss"] < best["loss"]
