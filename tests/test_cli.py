from typer.testing import CliRunner

from bsmaker.cli import app


def test_cli_help() -> None:
    result = CliRunner().invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "data" in result.output
