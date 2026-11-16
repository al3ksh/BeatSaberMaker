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
class Obstacle:
    time_s: float
    beat: float
    duration_s: float
    duration_beats: float
    x: int
    y: int
    w: int
    h: int


@dataclass(frozen=True)
class Bomb:
    time_s: float
    beat: float
    x: int
    y: int


@dataclass(frozen=True)
class Arc:
    time_s: float
    beat: float
    color: int
    x: int
    y: int
    d: int
    tail_time_s: float
    tail_beat: float
    tx: int
    ty: int
    td: int


@dataclass(frozen=True)
class Chain:
    time_s: float
    beat: float
    color: int
    x: int
    y: int
    d: int
    tail_time_s: float
    tail_beat: float
    tx: int
    ty: int
    segments: int
    squish: float


@dataclass(frozen=True)
class LightEvent:
    time_s: float
    beat: float
    type: int
    value: int
    float_value: float


@dataclass(frozen=True)
class Layers:
    bombs: tuple[Bomb, ...]
    obstacles: tuple[Obstacle, ...]
    arcs: tuple[Arc, ...]
    chains: tuple[Chain, ...]
    lights: tuple[LightEvent, ...]
    group_lighting: bool
    dropped: int


@dataclass(frozen=True)
class DifficultyFlags:
    has_bpm_changes: bool
    has_offgrid_notes: bool
    has_custom_note_data: bool
    has_time_offset: bool
    tempo_kind: str = "none"
    bpm_min: float = 0.0
    bpm_max: float = 0.0


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
    layers: Layers | None = None


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
    environment: str = ""
