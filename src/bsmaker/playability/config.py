"""Playability thresholds."""
from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PlayabilityConfig:
    strict_angle_deg: float = 45.0
    lenient_angle_deg: float = 90.0
    slow_gap_s: float = 0.35
    reset_gap_s: float = 0.80
    min_hand_gap_s: float = 0.09
    min_hand_gap_per_cell_s: float = 0.02
    vision_gap_s: float = 0.12
    forbid_crossover: bool = True
    # Blocked tokens for (left hand, right hand).
    rare_tokens: tuple[frozenset[int], frozenset[int]] = (frozenset(), frozenset())


def load_playability_config(path: Path | None) -> PlayabilityConfig:
    """Load from TOML (flat keys; rare_tokens_left / rare_tokens_right lists). None gives defaults."""
    if path is None:
        return PlayabilityConfig()
    with open(path, "rb") as fh:
        data = tomllib.load(fh)
    left = frozenset(data.pop("rare_tokens_left", ()))
    right = frozenset(data.pop("rare_tokens_right", ()))
    if left or right:
        data["rare_tokens"] = (left, right)
    return PlayabilityConfig(**data)
