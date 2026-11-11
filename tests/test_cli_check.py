from pathlib import Path
import pytest

from typer.testing import CliRunner

from bsmaker.cli import app

FIXTURES = Path(__file__).parent / "fixtures" / "maps"


def test_cli_check_prints_rules(tmp_path: Path, monkeypatch) -> None:
    _project(tmp_path, monkeypatch)
    result = CliRunner().invoke(app, ["check", str(FIXTURES / "v2_basic")])
    assert result.exit_code == 0
    assert "hard rate" in result.output


def test_cli_check_bad_folder_exit_2(tmp_path: Path) -> None:
    missing = CliRunner().invoke(app, ["check", str(tmp_path / "nope")])
    assert missing.exit_code == 2
    (tmp_path / "Info.dat").write_text("not json {{{", encoding="utf-8")
    garbage = CliRunner().invoke(app, ["check", str(tmp_path)])
    assert garbage.exit_code == 2
    assert "Traceback" not in garbage.output and len(garbage.output.strip().splitlines()) == 1
    empty = tmp_path / "empty"
    empty.mkdir()
    assert CliRunner().invoke(app, ["check", str(empty)]).exit_code == 2


def _project(tmp_path: Path, monkeypatch, mask_extra: str = "", data_extra: str = "") -> Path:
    levels = tmp_path / "levels"
    levels.mkdir()
    (tmp_path / "bsmaker.toml").write_text(
        f"custom_levels = '{levels}'\ndata_dir = '{tmp_path / 'data'}'\n", encoding="utf-8")
    (tmp_path / "playability.toml").write_text(
        f"[mask]\nforbid_crossover = true\n{mask_extra}\n[data]\nforbid_crossover = false\n{data_extra}\n",
        encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    return levels


def test_cli_check_profile_option(tmp_path: Path, monkeypatch) -> None:
    _project(tmp_path, monkeypatch)
    default = CliRunner().invoke(app, ["check", str(FIXTURES / "v2_basic")])
    assert default.exit_code == 0 and "profile: mask" in default.output
    data = CliRunner().invoke(app, ["check", str(FIXTURES / "v2_basic"), "--profile", "data"])
    assert data.exit_code == 0 and "profile: data" in data.output
    bad = CliRunner().invoke(app, ["check", str(FIXTURES / "v2_basic"), "--profile", "nope"])
    assert bad.exit_code == 2


def test_index_local_uses_data_profile(tmp_path: Path, monkeypatch) -> None:
    import json
    import shutil

    levels = _project(tmp_path, monkeypatch)
    folder = levels / "1f00 (Cross)"
    shutil.copytree(FIXTURES / "v2_basic", folder)
    notes = []
    for i in range(100):  # left hand on the right edge, right hand on the left: crossover only
        d = 1 if i % 2 == 0 else 0
        notes.append({"_time": 4 + i * 2, "_lineIndex": 3, "_lineLayer": 1, "_type": 0,
                      "_cutDirection": d})
        notes.append({"_time": 4 + i * 2, "_lineIndex": 0, "_lineLayer": 1, "_type": 1,
                      "_cutDirection": d})
    (folder / "ExpertStandard.dat").write_text(
        json.dumps({"_version": "2.2.0", "_notes": notes, "_obstacles": [], "_events": []}),
        encoding="utf-8")
    result = CliRunner().invoke(app, ["data", "index-local"])
    assert result.exit_code == 0, result.output
    import sqlite3
    row = sqlite3.connect(tmp_path / "data" / "manifest.sqlite").execute(
        "SELECT hard_rate, reject_reasons FROM difficulties").fetchone()
    assert row[0] == 0.0 and "hard_rate" not in row[1]
    # The same map under the mask profile has crossover violations.
    mask = CliRunner().invoke(app, ["check", str(folder)])
    assert "crossover=" in mask.output and "crossover=0" not in mask.output


def test_index_raw_missing_dir_exits_zero(tmp_path: Path, monkeypatch) -> None:
    _project(tmp_path, monkeypatch)
    result = CliRunner().invoke(app, ["data", "index-raw"])
    assert result.exit_code == 0 and "does not exist" in result.output


def test_check_warns_when_no_playability_toml(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)  # no bsmaker.toml, no playability.toml
    result = CliRunner().invoke(app, ["check", str(FIXTURES / "v2_basic")])
    assert result.exit_code == 0, result.output
    warnings = [ln for ln in result.output.splitlines() if "built-in defaults" in ln]
    assert len(warnings) == 1 and "playability.toml" in warnings[0]


def test_check_does_not_warn_with_playability_toml(tmp_path: Path, monkeypatch) -> None:
    _project(tmp_path, monkeypatch)
    result = CliRunner().invoke(app, ["check", str(FIXTURES / "v2_basic")])
    assert "built-in defaults" not in result.output


@pytest.mark.parametrize("args", [["data", "index-local"], ["data", "index-raw"],
                                  ["data", "features"], ["data", "scrape"]])
def test_data_commands_exit_2_without_ffmpeg(tmp_path: Path, monkeypatch, args) -> None:
    import shutil
    _project(tmp_path, monkeypatch)
    monkeypatch.setattr("bsmaker.cli.BeatSaverClient", lambda *a, **k: (_ for _ in ()).throw(
        AssertionError("must not reach the network")))
    real = shutil.which
    monkeypatch.setattr(shutil, "which",
                        lambda name, *a, **k: None if name in ("ffmpeg", "ffprobe") else real(name, *a, **k))
    result = CliRunner().invoke(app, args)
    assert result.exit_code == 2
    assert "ffmpeg" in result.output
    assert not (tmp_path / "data" / "manifest.sqlite").exists()


def test_summarise_counts_only_the_indexed_source(tmp_path: Path) -> None:
    from bsmaker.cli import _summarise
    from bsmaker.config import Config
    from bsmaker.data.index_local import IndexSummary
    from bsmaker.data.manifest import open_manifest, upsert_difficulty, upsert_map
    cfg = Config(custom_levels=tmp_path, data_dir=tmp_path / "d", base_dir=tmp_path)
    conn = open_manifest(cfg.manifest_path)
    for key, source, rank in (("1a", "local", "Easy"), ("2b", "beatsaver", "Expert")):
        upsert_map(conn, {"key": key, "source": source})
        upsert_difficulty(conn, {"map_key": key, "characteristic": "Standard", "rank": rank,
                                 "accepted": 1})
    _summarise(conn, IndexSummary(), "r.txt", cfg, "beatsaver")
    text = (cfg.reports_dir / "r.txt").read_text("utf-8")
    assert "Expert: 1" in text and "Easy" not in text
