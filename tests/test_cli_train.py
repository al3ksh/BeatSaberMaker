import json

import numpy as np
import pytest
from typer.testing import CliRunner

from bsmaker.cli import app
from bsmaker.formats.tokens import NONE_TOKEN, encode_token
from bsmaker.models.common import load_checkpoint

N_FRAMES = 1300
STATS = {"mel_mean": 0.0, "mel_std": 1.0, "nps_bins": [0.0, 3.0, 100.0]}


@pytest.fixture
def env(tmp_path, monkeypatch):
    """bsmaker.toml + a tiny event cache (2 train, 2 val difficulties) in tmp_path."""
    data = tmp_path / "data"
    (data / "events").mkdir(parents=True)
    (data / "features").mkdir()
    rng = np.random.default_rng(0)
    index, times, left, right, offsets = [], [], [], [], [0]
    for i, split in enumerate(["train", "train", "val", "val"]):
        key = f"k{i}"
        np.save(data / "features" / f"{key}.npy",
                rng.normal(size=(N_FRAMES, 80)).astype(np.float16))
        t = np.arange(1.0, 14.0, 0.25, dtype=np.float32)
        n = len(t)
        times.append(t)
        left.append(np.where(np.arange(n) % 2 == 0, encode_token(1, 1, 1), NONE_TOKEN))
        right.append(np.where(np.arange(n) % 2 == 1, encode_token(2, 1, 0), NONE_TOKEN))
        offsets.append(offsets[-1] + n)
        index.append({"map_key": key, "characteristic": "Standard", "rank": "Expert",
                      "split": split, "nps": 2.0, "styles": ["tech"], "njs": 16.0,
                      "njs_offset": 0.0, "n_frames": N_FRAMES, "score": 0.9})
    np.savez(data / "events" / "events.npz", time_s=np.concatenate(times),
             left=np.concatenate(left).astype(np.uint8),
             right=np.concatenate(right).astype(np.uint8), offsets=np.array(offsets))
    (data / "events" / "index.json").write_text(json.dumps(index), encoding="utf-8")
    (tmp_path / "dataset_stats.json").write_text(json.dumps(STATS), encoding="utf-8")
    (tmp_path / "bsmaker.toml").write_text(
        f"custom_levels = '{tmp_path / 'levels'}'\ndata_dir = '{data}'\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    return tmp_path


def _train(kind, out, *extra):
    return CliRunner().invoke(app, ["train", kind, "--device", "cpu", "--max-steps", "2",
                                    "--batch-size", "2", "--num-workers", "0",
                                    "--eval-every", "2", "--save-every", "2",
                                    "--out", str(out), *extra])


@pytest.mark.parametrize("bad", ["-0.1", "0.51", "1"])
def test_corrupt_p_out_of_range_is_a_usage_error(env, bad):
    r = _train("pattern", env / "ck", "--corrupt-p", bad)
    assert r.exit_code == 2 and "--corrupt-p" in r.output
    assert not (env / "ck").exists()


def test_pattern_trains_with_corruption_and_records_it(env):
    r = _train("pattern", env / "ck", "--corrupt-p", "0.3")
    assert r.exit_code == 0, r.output
    assert load_checkpoint(env / "ck" / "last.pt")["train_options"]["corrupt_p"] == 0.3
    again = _train("pattern", env / "ck", "--corrupt-p", "0.2", "--max-steps", "4")
    assert again.exit_code != 0 and "corrupt_p" in str(again.exception)


@pytest.mark.parametrize("sizes,figures", [
    ("PATTERN_SIZES", {"s": 15e6, "m": 38e6, "l": 85e6}),
    ("TIMING_SIZES", {"s": 5e6, "m": 14e6})])
def test_size_presets_build_models_of_the_planned_size(sizes, figures):
    from dataclasses import asdict

    from bsmaker import cli
    from bsmaker.models.pattern.model import PatternConfig, PatternModel
    from bsmaker.models.timing.model import TimingConfig, TimingModel
    table = getattr(cli, sizes)
    assert set(table) == set(figures)
    cfg_cls, model_cls = ((PatternConfig, PatternModel) if sizes == "PATTERN_SIZES"
                          else (TimingConfig, TimingModel))
    for size, want in figures.items():
        cfg = cfg_cls(**table[size])
        assert all(asdict(cfg)[k] == v for k, v in table[size].items())
        n = sum(p.numel() for p in model_cls(cfg).parameters())
        assert abs(n - want) <= 0.15 * want, (size, n)


@pytest.mark.parametrize("kind", ["pattern", "timing"])
def test_unknown_size_and_bad_lr_are_usage_errors(env, kind):
    for extra in (["--size", "xl"], ["--lr", "0"], ["--lr", "-1"]):
        r = _train(kind, env / "ck", *extra)
        assert r.exit_code == 2, (kind, extra, r.output)
    assert _train("pattern", env / "ck", "--size", "q").exit_code == 2
    assert not (env / "ck").exists()
    if kind == "timing":  # `l` exists for the pattern model only
        assert _train("timing", env / "ck", "--size", "l").exit_code == 2


def test_invalid_weighting_is_a_usage_error(env):
    for kind in ("pattern", "timing"):
        r = _train(kind, env / "ck", "--weighting", "bogus")
        assert r.exit_code == 2 and "--weighting" in r.output


@pytest.mark.parametrize("kind", ["pattern", "timing"])
def test_weighting_none_trains_and_start_log_names_size_params_options(env, kind):
    r = _train(kind, env / "ck", "--weighting", "none", "--lr", "0.001")
    assert r.exit_code == 0, r.output
    n = 15175770 if kind == "pattern" else 4986081
    assert f"size s: {n:,} parameters" in r.output
    assert "weighting=none" in r.output and "lr=0.001" in r.output
    opts = load_checkpoint(env / "ck" / "last.pt")["train_options"]
    assert opts["size"] == "s" and opts["lr"] == 0.001 and opts["weighting"] == "none"
    assert ("corrupt_p" in opts) == (kind == "pattern")


@pytest.mark.parametrize("kind", ["pattern", "timing"])
def test_lr_reaches_train_settings(env, monkeypatch, kind):
    seen = {}

    def fake(model, train_ds, val_ds, step_fn, eval_fn, out_dir, settings, *a, **kw):
        seen["settings"], seen["options"] = settings, kw["train_options"]
        return {"step": 0, "best": None}

    monkeypatch.setattr("bsmaker.models.train_loop.run_training", fake)
    r = _train(kind, env / "ck", "--lr", "0.002", "--weighting", "none")
    assert r.exit_code == 0, r.output
    assert seen["settings"].lr == 0.002 and seen["options"]["lr"] == 0.002


def test_resume_with_other_size_stops_naming_both_configs(env):
    assert _train("timing", env / "ck").exit_code == 0
    r = _train("timing", env / "ck", "--size", "m", "--max-steps", "4")
    assert r.exit_code != 0
    msg = str(r.exception)
    assert f"checkpoint in {env / 'ck'} was made with " in msg
    assert "'d_model': 256" in msg and "'d_model': 384" in msg and msg.endswith("use another --out")


@pytest.mark.parametrize("kind", ["pattern", "timing"])
def test_resume_with_other_schedule_or_batch_is_refused(env, kind):
    first = _train(kind, env / "ck", "--weighting", "none")
    assert first.exit_code == 0, first.output
    opts = load_checkpoint(env / "ck" / "last.pt")["train_options"]
    assert "max_steps" in opts and "batch_size" in opts
    again = _train(kind, env / "ck", "--weighting", "none", "--max-steps", str(opts["max_steps"] + 2))
    assert again.exit_code != 0 and "max_steps" in str(again.exception)
