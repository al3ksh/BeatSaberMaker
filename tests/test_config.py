from pathlib import Path

import pytest

from bsmaker.config import load_config

TOML = r"""custom_levels = 'C:\Levels'
data_dir = 'F:\Data'
"""


def test_load_config_reads_paths(tmp_path: Path) -> None:
    f = tmp_path / "bsmaker.toml"
    f.write_text(TOML, encoding="utf-8")
    cfg = load_config(f)
    assert isinstance(cfg.custom_levels, Path)
    assert isinstance(cfg.data_dir, Path)
    assert cfg.manifest_path.name == "manifest.sqlite"
    assert cfg.raw_dir.parent == cfg.data_dir


def test_load_config_missing_file_raises(tmp_path: Path) -> None:
    missing = tmp_path / "nope.toml"
    with pytest.raises(FileNotFoundError, match="nope.toml"):
        load_config(missing)
