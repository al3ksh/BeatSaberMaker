from pathlib import Path

import pytest

from bsmaker.config import Config, load_config

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


def test_config_from_paths_overrides_resources(tmp_path: Path) -> None:
    levels = tmp_path / "CustomLevels"
    ckpts = tmp_path / "pack" / "checkpoints"
    play = tmp_path / "pack" / "playability.toml"
    lstats = tmp_path / "pack" / "layer_stats.json"
    play.parent.mkdir(parents=True)
    play.write_text("[mask]\n", encoding="utf-8")
    lstats.write_text("{}", encoding="utf-8")

    cfg = Config.from_paths(
        levels, checkpoints_dir=ckpts, playability_path=play, layer_stats_path=lstats,
    )
    assert cfg.custom_levels == levels
    assert cfg.checkpoints_dir == ckpts
    assert cfg.playability_path == play
    assert cfg.layer_stats_path == lstats

    missing_cfg = Config.from_paths(
        levels, checkpoints_dir=ckpts,
        playability_path=tmp_path / "nope.toml",
        layer_stats_path=tmp_path / "nope.json",
    )
    assert missing_cfg.playability_path is None
    assert missing_cfg.layer_stats_path is None
    assert missing_cfg._layer_stats_path == tmp_path / "nope.json"
