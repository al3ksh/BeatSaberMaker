from pathlib import Path

from typer.testing import CliRunner

from bsmaker.cli import app

FIXTURES = Path(__file__).parent / "fixtures" / "maps"


def test_cli_check_prints_rules() -> None:
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
