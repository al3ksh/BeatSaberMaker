import re

from typer.testing import CliRunner

from bsmaker.cli import app


def test_cli_help() -> None:
    result = CliRunner().invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "data" in result.output


def test_train_help_lists_ornament_and_layer_commands() -> None:
    result = CliRunner().invoke(app, ["train", "--help"])
    assert result.exit_code == 0, result.output
    for cmd in ("timing", "pattern", "ornaments", "layer-when", "layer-what"):
        assert cmd in result.output


def test_generate_help_lists_layer_and_level_options() -> None:
    result = CliRunner().invoke(app, ["generate", "--help"])
    assert result.exit_code == 0, result.output
    # Rich colours the help on CI (GITHUB_ACTIONS), which splits option names with ANSI codes.
    plain = re.sub(r"\x1b\[[0-9;]*m", "", result.output)
    for opt in ("--levels", "--ozdoby", "--swiatla", "--tlo", "--swiatla-reguly"):
        assert opt in plain
