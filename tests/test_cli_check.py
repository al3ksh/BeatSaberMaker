from pathlib import Path

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
