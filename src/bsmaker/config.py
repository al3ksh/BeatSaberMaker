"""Project configuration loaded from bsmaker.toml."""
from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Config:
    custom_levels: Path
    data_dir: Path
    base_dir: Path = Path(".")  # folder holding bsmaker.toml

    @property
    def playability_path(self) -> Path | None:
        p = self.base_dir / "playability.toml"
        return p if p.is_file() else None

    @property
    def manifest_path(self) -> Path:
        return self.data_dir / "manifest.sqlite"

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def features_dir(self) -> Path:
        return self.data_dir / "features"

    @property
    def events_dir(self) -> Path:
        return self.data_dir / "events"

    @property
    def layers_dir(self) -> Path:
        return self.data_dir / "layers"

    @property
    def checkpoints_dir(self) -> Path:
        return self.data_dir / "checkpoints"

    @property
    def reports_dir(self) -> Path:
        return self.data_dir / "reports"


def load_config(path: Path | None = None) -> Config:
    """Load config from `path` (default: ./bsmaker.toml)."""
    path = Path("bsmaker.toml") if path is None else Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("rb") as fh:
        data = tomllib.load(fh)
    return Config(
        custom_levels=Path(data["custom_levels"]),
        data_dir=Path(data["data_dir"]),
        base_dir=path.parent,
    )
