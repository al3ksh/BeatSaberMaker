"""Immutable representation of a Beat Saber map."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Note:
    time_s: float
    beat: float
    x: int
    y: int
    color: int
    direction: int


@dataclass(frozen=True)
class DifficultyFlags:
    has_bpm_changes: bool
    has_offgrid_notes: bool
    has_custom_note_data: bool
    has_time_offset: bool


@dataclass(frozen=True)
class Difficulty:
    characteristic: str
    rank: str
    schema: str  # "v2", "v3" or "v4"
    njs: float
    njs_offset: float
    notes: tuple[Note, ...]
    n_bombs: int
    n_obstacles: int
    n_arcs: int
    n_chains: int
    flags: DifficultyFlags


@dataclass(frozen=True)
class BeatMap:
    folder: Path
    key: str
    title: str
    artist: str
    mapper: str
    bpm: float
    audio_path: Path | None
    requirements: tuple[str, ...]
    suggestions: tuple[str, ...]
    difficulties: tuple[Difficulty, ...]
    problems: tuple[str, ...]
