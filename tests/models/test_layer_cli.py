import json

import numpy as np
import pytest
import typer
from typer.testing import CliRunner

from bsmaker.data.layers_cache import _ARRAYS, _GROUPS, _empty_record
from bsmaker.formats.tokens import NONE_TOKEN, encode_token
from bsmaker.models.common import load_checkpoint
from bsmaker.models.layers.cli import register

N_FRAMES = 1300
STATS = {"mel_mean": 0.0, "mel_std": 1.0, "nps_bins": [0.0, 3.0, 100.0]}


@pytest.fixture
def env(tmp_path, monkeypatch):
    data = tmp_path / "data"
    (data / "events").mkdir(parents=True)
    (data / "layers").mkdir()
    (data / "features").mkdir()
    rng = np.random.default_rng(0)
    index, times, left, right, eo = [], [], [], [], [0]
    recs, rows, lo = [], {k: [] for k in _ARRAYS}, {g: [0] for g in _GROUPS}
    for i, split in enumerate(["train", "train", "val", "val"]):
        np.save(data / "features" / f"k{i}.npy",
                rng.normal(size=(N_FRAMES, 80)).astype(np.float16))
        t = np.arange(1.0, 14.0, 0.25, dtype=np.float32)
        n = len(t)
        times.append(t)
        left.append(np.full(n, encode_token(1, 1, 1)))
        right.append(np.full(n, NONE_TOKEN))
        eo.append(eo[-1] + n)
        index.append({"map_key": f"k{i}", "characteristic": "Standard", "rank": "Expert",
                      "split": split, "nps": 2.0, "styles": ["tech"], "njs": 16.0,
                      "njs_offset": 0.0, "n_frames": N_FRAMES, "score": 0.9})
        m = 20
        lt = np.sort(rng.uniform(2, 13, m)).astype(np.float32)
        rows["light_t"] += lt.tolist()
        rows["light_ch"] += (np.arange(m) % 9).tolist()
        rows["light_tok"] += [0 if c <= 4 else (17 if c <= 6 else 7)
                              for c in (np.arange(m) % 9)]
        rows["wall_t"] += [3.0, 6.0]
        rows["wall_side"] += [0, 1]
        rows["wall_tok"] += [1, 2]
        rows["bomb_t"] += [4.0, 8.0]
        rows["bomb_mask"] += [5, 5]
        for g, keys in _GROUPS.items():
            lo[g].append(len(rows[keys[0]]))
        rec = _empty_record(i)
        rec.update(lights_ok=True, n_lights=m, n_walls=2, n_bomb_groups=2, readable=True)
        recs.append(rec)
    np.savez(data / "events" / "events.npz", time_s=np.concatenate(times),
             left=np.concatenate(left).astype(np.uint8),
             right=np.concatenate(right).astype(np.uint8), offsets=np.array(eo))
    (data / "events" / "index.json").write_text(json.dumps(index), encoding="utf-8")
    arrays = {k: np.asarray(v, dtype=_ARRAYS[k]) for k, v in rows.items()}
    arrays.update({f"{g}_offsets": np.array(o) for g, o in lo.items()})
    np.savez(data / "layers" / "layers.npz", **arrays)
    (data / "layers" / "index.json").write_text(json.dumps(recs), encoding="utf-8")
    (tmp_path / "dataset_stats.json").write_text(json.dumps(STATS), encoding="utf-8")
    (tmp_path / "layer_stats.json").write_text(json.dumps({"bomb_vocab": [5, 6]}),
                                               encoding="utf-8")
    (tmp_path / "bsmaker.toml").write_text(
        f"custom_levels = '{tmp_path / 'levels'}'\ndata_dir = '{data}'\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    return tmp_path


def _run(cmd, layer, out, *extra):
    app = typer.Typer()
    register(app)

    @app.command("noop")
    def noop() -> None:  # makes typer keep the sub-command names
        pass

    return CliRunner().invoke(app, [cmd, "--layer", layer, "--device", "cpu", "--max-steps", "2",
                                    "--batch-size", "2", "--num-workers", "0", "--eval-every",
                                    "2", "--save-every", "2", "--out", str(out), *extra])


@pytest.mark.parametrize("cmd", ["layer-when", "layer-what"])
@pytest.mark.parametrize("layer", ["lights", "decor"])
def test_train_commands_write_checkpoints(env, cmd, layer):
    r = _run(cmd, layer, env / "ck")
    assert r.exit_code == 0, r.output
    ck = load_checkpoint(env / "ck" / "best.pt")
    assert ck["train_options"]["layer"] == layer
    key = "n_channels" if cmd == "layer-when" else "vocab_size"
    assert key in ck["model_config"]
    if layer == "decor":
        assert ck["stats"]["bomb_masks"] == [5, 6]
        if cmd == "layer-what":
            assert ck["model_config"]["vocab_size"] == 26


def test_bad_layer_is_a_usage_error(env):
    r = _run("layer-when", "nope", env / "ck")
    assert r.exit_code == 2 and "--layer" in r.output
