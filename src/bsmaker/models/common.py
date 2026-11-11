"""Pieces shared by the timing and note models: conditioning inputs and checkpoint I/O."""
from __future__ import annotations

import os
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn

from bsmaker.data.styles import STYLES, UNKNOWN

STYLE_DIM = len(STYLES) + 1
_STYLE_INDEX = {name: i for i, name in enumerate((*STYLES, UNKNOWN))}


def style_vector(styles: Iterable[str]) -> np.ndarray:
    """Multi-hot float32 vector over STYLES + unknown; no known style means `unknown`."""
    v = np.zeros(STYLE_DIM, dtype=np.float32)
    for s in styles:
        if s in _STYLE_INDEX and s != UNKNOWN:
            v[_STYLE_INDEX[s]] = 1.0
    if not v.any():
        v[_STYLE_INDEX[UNKNOWN]] = 1.0
    return v


def log_nps_value(nps: float) -> float:
    """Density conditioning value fed to `Conditioning`."""
    return float(np.log1p(max(nps, 0.0)))


class Conditioning(nn.Module):
    """Embeds (log notes-per-second, style multi-hot) into one vector per sample."""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.nps = nn.Sequential(nn.Linear(1, d_model), nn.GELU(), nn.Linear(d_model, d_model))
        self.styles = nn.Linear(STYLE_DIM, d_model)

    def forward(self, log_nps: Tensor, styles: Tensor) -> Tensor:
        return self.nps(log_nps.unsqueeze(-1)) + self.styles(styles)


def save_checkpoint(path: Path, payload: dict) -> None:
    """Write via a temporary file and rename, so a crash never leaves a half-written file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def load_checkpoint(path: Path, map_location: str | torch.device = "cpu") -> dict:
    return torch.load(Path(path), map_location=map_location, weights_only=True)
