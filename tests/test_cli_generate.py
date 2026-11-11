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


def test_cli_missing_audio_and_bad_device(tmp_path, monkeypatch):
    _project(tmp_path, monkeypatch)
    r = CliRunner().invoke(app, ["generate", str(tmp_path / "nope.wav"), "--nps", "4",
                                 "--device", "cpu"])
    assert r.exit_code == 1 and "not found" in r.output
    audio = write_audio(tmp_path / "a.wav")
    r = CliRunner().invoke(app, ["generate", str(audio), "--nps", "4", "--device", "tpu"])
    assert r.exit_code == 2


def test_cli_checkpoint_without_stats_and_unwritable_out(tmp_path, monkeypatch):
    _project(tmp_path, monkeypatch)
    audio = write_audio(tmp_path / "a.wav")
    blocker = tmp_path / "file"
    blocker.write_text("x")
    r = CliRunner().invoke(app, ["generate", str(audio), "--nps", "3", "--out",
                                 str(blocker / "sub"), "--device", "cpu"])
    assert r.exit_code == 1 and "Traceback" not in r.output
    # pattern checkpoint stats are valid (needed to load); strip them from the timing one
    t, _ = make_models()
    save_checkpoint(tmp_path / "data" / "checkpoints" / "timing" / "best.pt", {
        "model": t.model.state_dict(), "model_config": asdict(t.config),
        "stats": {"coverage_median": 0.8, "nps_bins": [0.0, 100.0],
                  "double_share_by_bin": [0.2], "mel_mean": 0.0, "mel_std": 1.0},
        "step": 1, "best": {}})
    r = CliRunner().invoke(app, ["generate", str(audio), "--nps", "3", "--out",
                                 str(tmp_path / "o2"), "--device", "cpu"])
    assert r.exit_code == 1 and "dataset stats" in r.output and "njs_by_bin" in r.output


def test_cli_internal_keyerror_propagates(tmp_path, monkeypatch):
    from bsmaker import generate as G

    _project(tmp_path, monkeypatch)
    audio = write_audio(tmp_path / "a.wav")

    def boom(*a, **k):
        raise KeyError("bug")

    monkeypatch.setattr(G, "sample_notes", boom)
    r = CliRunner().invoke(app, ["generate", str(audio), "--nps", "3", "--out",
                                 str(tmp_path / "o3"), "--device", "cpu"])
    assert r.exit_code == 1 and isinstance(r.exception, KeyError)
    assert "dataset stats" not in r.output
