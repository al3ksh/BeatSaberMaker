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
