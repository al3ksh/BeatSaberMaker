import sys
from dataclasses import asdict
from pathlib import Path

from typer.testing import CliRunner

from bsmaker.cli import app
from bsmaker.models.common import save_checkpoint

sys.path.insert(0, str(Path(__file__).parent))
from test_generate import STATS, make_models, write_audio  # noqa: E402


def _project(tmp_path: Path, monkeypatch, checkpoints: bool = True) -> Path:
    levels = tmp_path / "levels"
    levels.mkdir()
    (tmp_path / "bsmaker.toml").write_text(
        f"custom_levels = '{levels}'\ndata_dir = '{tmp_path / 'data'}'\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    if checkpoints:
        t, p = make_models()
        for name, loaded in (("timing", t), ("pattern", p)):
            save_checkpoint(tmp_path / "data" / "checkpoints" / name / "best.pt", {
                "model": loaded.model.state_dict(), "model_config": asdict(loaded.config),
                "stats": STATS, "step": 1, "best": {}})
    return levels


def test_cli_nps_out_of_range_exit_2(tmp_path, monkeypatch):
    _project(tmp_path, monkeypatch, checkpoints=False)
    audio = write_audio(tmp_path / "a.wav")
    for bad in ("0.4", "12.5", "-1"):
        r = CliRunner().invoke(app, ["generate", str(audio), "--nps", bad])
        assert r.exit_code == 2, bad


def test_cli_unknown_style_exit_2(tmp_path, monkeypatch):
    _project(tmp_path, monkeypatch, checkpoints=False)
    audio = write_audio(tmp_path / "a.wav")
    r = CliRunner().invoke(app, ["generate", str(audio), "--nps", "4", "--style", "nope"])
    assert r.exit_code == 2


def test_cli_missing_checkpoint_exit_1(tmp_path, monkeypatch):
    levels = _project(tmp_path, monkeypatch, checkpoints=False)
    audio = write_audio(tmp_path / "a.wav")
    r = CliRunner().invoke(app, ["generate", str(audio), "--nps", "4", "--device", "cpu"])
    assert r.exit_code == 1 and "checkpoint" in r.output
    assert "Traceback" not in r.output and list(levels.iterdir()) == []


def test_cli_generate_to_out_dir(tmp_path, monkeypatch):
    levels = _project(tmp_path, monkeypatch)
    audio = write_audio(tmp_path / "a.wav")
    out = tmp_path / "out"
    args = ["generate", str(audio), "--nps", "3", "--style", "tech", "--out", str(out),
            "--device", "cpu"]
    r = CliRunner().invoke(app, args)
    assert r.exit_code == 0, r.output
    assert (out / "BSM - a [nps 3, tech]" / "Info.dat").is_file()
    assert "notes" in r.output and list(levels.iterdir()) == []
    again = CliRunner().invoke(app, args)
    assert again.exit_code == 1 and "already exists" in again.output
    assert CliRunner().invoke(app, args + ["--overwrite"]).exit_code == 0
    short = write_audio(tmp_path / "s.wav", seconds=5)
    r = CliRunner().invoke(app, ["generate", str(short), "--nps", "3", "--out", str(out),
                                 "--device", "cpu"])
    assert r.exit_code == 1 and "shorter" in r.output
