import json
import sys
from dataclasses import asdict
from pathlib import Path

from typer.testing import CliRunner

from bsmaker.cli import app
from bsmaker.data.layer_stats import write_layer_stats
from bsmaker.formats.reader import read_map
from bsmaker.models.common import save_checkpoint
from bsmaker.playability.layers import check_layers

sys.path.insert(0, str(Path(__file__).parent))
from test_generate import (  # noqa: E402
    BOMB_VOCAB, LAYER_STATS, STATS, make_layer_models, make_models, write_audio)


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


def test_cli_generate_warns_without_playability_toml(tmp_path, monkeypatch):
    _project(tmp_path, monkeypatch)
    audio = write_audio(tmp_path / "a.wav")
    r = CliRunner().invoke(app, ["generate", str(audio), "--nps", "3", "--style", "tech",
                                 "--out", str(tmp_path / "out"), "--device", "cpu"])
    assert r.exit_code == 0, r.output
    assert "playability.toml not found" in r.output


def test_resplit_runs_and_warns_only_on_change(tmp_path, monkeypatch):
    _project(tmp_path, monkeypatch, checkpoints=False)
    r = CliRunner().invoke(app, ["data", "resplit"])
    assert r.exit_code == 0, r.output
    assert "WARNING" not in r.output


def _write_layer_assets(tmp_path: Path, *, skip: set[str] = (), write_stats: bool = True) -> None:
    lm = make_layer_models()
    layers_dir = tmp_path / "data" / "checkpoints" / "layers"
    if "ornaments" not in skip:
        save_checkpoint(layers_dir / "ornaments" / "best.pt", {
            "model": lm.ornaments.model.state_dict(),
            "model_config": asdict(lm.ornaments.config),
            "stats": STATS,
        })
    specs = (
        ("decor-when", lm.decor_when, "decor", dict(STATS, bomb_masks=BOMB_VOCAB)),
        ("decor-what", lm.decor_what, "decor", dict(STATS, bomb_masks=BOMB_VOCAB)),
        ("lights-when", lm.lights_when, "lights", STATS),
        ("lights-what", lm.lights_what, "lights", STATS),
    )
    for name, loaded, layer, st in specs:
        if name not in skip:
            save_checkpoint(layers_dir / name / "best.pt", {
                "model": loaded.model.state_dict(),
                "model_config": asdict(loaded.config),
                "stats": st,
                "train_options": {"layer": layer},
            })
    if write_stats:
        write_layer_stats(LAYER_STATS, tmp_path / "layer_stats.json")


def test_cli_generate_option_validation_errors(tmp_path, monkeypatch):
    _project(tmp_path, monkeypatch, checkpoints=False)
    audio = write_audio(tmp_path / "a.wav")
    bad_invocations = [
        ["generate", str(audio)],
        ["generate", str(audio), "--nps", "3", "--levels", "hard"],
        ["generate", str(audio), "--levels", "hard,Hard"],
        ["generate", str(audio), "--levels", "easy,insane"],
        ["generate", str(audio), "--levels", "hard,"],
        ["generate", str(audio), "--nps", "3", "--ozdoby", "4"],
        ["generate", str(audio), "--nps", "3", "--ozdoby", "-1"],
        ["generate", str(audio), "--nps", "3", "--swiatla", "4"],
        ["generate", str(audio), "--nps", "3", "--swiatla", "-1"],
        ["generate", str(audio), "--nps", "3", "--tlo", "Space"],
        ["generate", str(audio), "--nps", "3", "--tlo", "DefaultEnvironment"],
    ]
    for args in bad_invocations:
        r = CliRunner().invoke(app, args)
        assert r.exit_code == 2, (args, r.output)


def test_cli_generate_multi_level_with_layers_and_summary(tmp_path, monkeypatch):
    _project(tmp_path, monkeypatch)
    _write_layer_assets(tmp_path)
    audio = write_audio(tmp_path / "a.wav", seconds=45.0)
    out = tmp_path / "out"
    r = CliRunner().invoke(app, [
        "generate", str(audio), "--levels", "hard,expertplus", "--tlo", "nice",
        "--ozdoby", "2", "--swiatla", "2", "--out", str(out), "--device", "cpu",
    ])
    assert r.exit_code == 0, r.output
    folder = out / "BSM - a [Hard+ExpertPlus, unknown]"
    assert (folder / "HardStandard.dat").is_file()
    assert (folder / "ExpertPlusStandard.dat").is_file()
    info = json.loads((folder / "Info.dat").read_text(encoding="utf-8"))
    assert info["_environmentName"] == "NiceEnvironment"
    m = read_map(folder, layers=True)
    assert len(m.difficulties) == 2
    for d in m.difficulties:
        assert d.layers is not None
        assert check_layers(d.notes, d.layers) == []
        assert d.layers.arcs and d.layers.chains and d.layers.obstacles and d.layers.bombs and d.layers.lights
    summary = r.output.strip().splitlines()[-1]
    for token in ("Hard:", "ExpertPlus:", "notes", "nps", "arcs", "chains",
                  "walls", "bombs", "light events", "NiceEnvironment"):
        assert token in summary, summary


def test_cli_generate_layers_off_and_rule_lights_suppress_missing_warnings(tmp_path, monkeypatch):
    _project(tmp_path, monkeypatch)
    (tmp_path / "playability.toml").write_text("[mask]\n[data]\n", encoding="utf-8")
    audio = write_audio(tmp_path / "a.wav")
    out1 = tmp_path / "out1"
    r1 = CliRunner().invoke(app, [
        "generate", str(audio), "--nps", "3", "--ozdoby", "0", "--swiatla", "0",
        "--out", str(out1), "--device", "cpu",
    ])
    assert r1.exit_code == 0, r1.output
    layer_warns1 = [ln for ln in r1.output.splitlines()
                    if ln.startswith("warning:") and ("checkpoint" in ln or "layer_stats" in ln)]
    assert layer_warns1 == []
    d1 = read_map(out1 / "BSM - a [nps 3, unknown]", layers=True).difficulties[0]
    assert d1.layers is not None
    assert (d1.layers.arcs, d1.layers.chains, d1.layers.obstacles, d1.layers.bombs, d1.layers.lights) == ((), (), (), (), ())

    out2 = tmp_path / "out2"
    r2 = CliRunner().invoke(app, [
        "generate", str(audio), "--nps", "3", "--ozdoby", "0", "--swiatla", "2",
        "--swiatla-reguly", "--out", str(out2), "--device", "cpu",
    ])
    assert r2.exit_code == 0, r2.output
    layer_warns2 = [ln for ln in r2.output.splitlines()
                    if ln.startswith("warning:") and ("checkpoint" in ln or "layer_stats" in ln)]
    assert layer_warns2 == []
    d2 = read_map(out2 / "BSM - a [nps 3, unknown]", layers=True).difficulties[0]
    assert len(d2.layers.lights) > 0


def test_cli_missing_checkpoints_and_stats_warn_once_each_and_write_map(tmp_path, monkeypatch):
    _project(tmp_path, monkeypatch)
    (tmp_path / "playability.toml").write_text("[mask]\n[data]\n", encoding="utf-8")
    _write_layer_assets(tmp_path, skip={"ornaments", "lights-what"})
    audio = write_audio(tmp_path / "a.wav")
    out = tmp_path / "out"
    r = CliRunner().invoke(app, [
        "generate", str(audio), "--nps", "3", "--out", str(out), "--device", "cpu",
    ])
    assert r.exit_code == 0, r.output
    warns = [ln for ln in r.output.splitlines()
             if ln.startswith("warning:") and ("checkpoint" in ln or "layer_stats" in ln)]
    assert len(warns) == 2
    assert any("ornaments" in w for w in warns)
    assert any("lights-what" in w for w in warns)
    d = read_map(out / "BSM - a [nps 3, unknown]", layers=True).difficulties[0]
    assert d.layers.arcs == () and d.layers.chains == ()
    assert len(d.layers.lights) > 0  # fell back to rule lights

    # Missing layer_stats.json prints one warning and still writes map
    _write_layer_assets(tmp_path, write_stats=False)
    (tmp_path / "layer_stats.json").unlink()
    out2 = tmp_path / "out2"
    r2 = CliRunner().invoke(app, [
        "generate", str(audio), "--nps", "3", "--out", str(out2), "--device", "cpu",
    ])
    assert r2.exit_code == 0, r2.output
    warns2 = [ln for ln in r2.output.splitlines()
              if ln.startswith("warning:") and ("checkpoint" in ln or "layer_stats" in ln)]
    assert len(warns2) == 1 and "layer_stats.json" in warns2[0]
    assert (out2 / "BSM - a [nps 3, unknown]" / "Info.dat").is_file()


def test_cli_waz_validation(tmp_path, monkeypatch):
    _project(tmp_path, monkeypatch, checkpoints=False)
    audio = write_audio(tmp_path / "a.wav")
    r = CliRunner().invoke(app, ["generate", str(audio), "--nps", "3", "--waz", "5"])
    assert r.exit_code == 2
    assert "--waz must be 0, 1, 2, or 3" in r.output

