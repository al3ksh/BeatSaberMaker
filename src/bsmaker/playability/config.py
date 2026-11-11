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
    # Same-hand notes closer than this and with compatible directions are slider tails (0 = off).
    slider_gap_s: float = 0.0
    # Blocked tokens for (left hand, right hand).
    rare_tokens: tuple[frozenset[int], frozenset[int]] = (frozenset(), frozenset())


PROFILES = ("mask", "data")
DATA_DEFAULTS = PlayabilityConfig(
    reset_gap_s=0.25, forbid_crossover=False, strict_angle_deg=90.0, lenient_angle_deg=90.0,
    slider_gap_s=0.12, min_hand_gap_s=0.06, min_hand_gap_per_cell_s=0.0,
)


def load_playability_config(path: Path | None, profile: str = "mask") -> PlayabilityConfig:
    """Load one profile table ([mask] or [data]) from TOML; None gives that profile's defaults."""
    if profile not in PROFILES:
        raise ValueError(f"unknown playability profile {profile!r}; expected one of {PROFILES}")
    if path is None:
        return PlayabilityConfig() if profile == "mask" else DATA_DEFAULTS
    with open(path, "rb") as fh:
        doc = tomllib.load(fh)
    table = doc.get(profile)
    if not isinstance(table, dict):
        raise ValueError(f"{path}: missing table [{profile}] (regenerate with "
                         "`bsmaker data calibrate-playability`)")
    data = dict(table)
    left = frozenset(data.pop("rare_tokens_left", ()))
    right = frozenset(data.pop("rare_tokens_right", ()))
    if left or right:
        data["rare_tokens"] = (left, right)
    return PlayabilityConfig(**data)
