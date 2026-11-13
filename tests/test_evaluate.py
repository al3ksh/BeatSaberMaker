import sys
from pathlib import Path

import numpy as np

from bsmaker import evaluate as E
from bsmaker.data.events_cache import Entry, EventCache
from bsmaker.formats.events import Event
from bsmaker.formats.tokens import NONE_TOKEN, encode_token

sys.path.insert(0, str(Path(__file__).parent))
from test_generate import MASK, STATS, make_models  # noqa: E402

N_FRAMES = 3000


def tiny_cache(tmp_path: Path) -> tuple[EventCache, Path]:
    feats = tmp_path / "features"
    feats.mkdir()
    rng = np.random.default_rng(0)
    for key in ("a", "b"):
        np.save(feats / f"{key}.npy", rng.normal(size=(N_FRAMES, 80)).astype(np.float16))
    times = np.arange(2.0, 32.0, 0.5, dtype=np.float32)
    left = np.where(np.arange(len(times)) % 2 == 0, encode_token(1, 1, 1), NONE_TOKEN)
    right = np.where(np.arange(len(times)) % 2 == 1, encode_token(2, 1, 0), NONE_TOKEN)
    entries = [Entry(i, k, "Standard", r, "test", 2.0, ("tech",), 16.0, 0.0, N_FRAMES)
               for i, (k, r) in enumerate([("a", "Hard"), ("a", "Expert"), ("b", "Hard")])]
    n = len(times)
    cache = EventCache(entries, np.tile(times, 3), np.tile(left, 3).astype(np.uint8),
                       np.tile(right, 3).astype(np.uint8), np.arange(4) * n)
    return cache, feats


def test_distribution_shares_sum_to_one():
    evs = [Event(1.0, encode_token(0, 0, 1), encode_token(3, 2, 8)),
           Event(2.0, encode_token(1, 1, 2), NONE_TOKEN)]
    d = E.distribution([evs])
    for axis in ("column", "row", "direction", "hand"):
        assert abs(sum(d[axis]) - 1.0) < 1e-9
    assert d["n_notes"] == 3 and abs(d["double_share"] - 0.5) < 1e-9
    assert E.distribution([[]])["double_share"] == 0.0


def test_tv_distance_zero_for_identical():
    p = [0.2, 0.3, 0.5]
    assert E.tv_distance(p, p) == 0.0
    assert abs(E.tv_distance([1, 0], [0, 1]) - 1.0) < 1e-12


def test_eval_timing_runs_on_tiny_cache(tmp_path):
    cache, feats = tiny_cache(tmp_path)
    timing, pattern = make_models()
    r = E.eval_timing(timing, cache, feats, "test", 2, "cpu")
    assert r["n"] == 2 and 0.0 <= r["f1"] <= 1.0
    assert sum(v["n"] for v in r["f1_by_nps_bin"].values()) == 2
    p = E.eval_pattern(pattern, cache, feats, "test", 3, "cpu")
    assert p["n"] == 3 and p["ce"] > 0 and abs(p["ce"] - p["ce_left"] - p["ce_right"]) < 1e-5


def test_eval_generated_reports_zero_hard_violations_and_density_share(tmp_path):
    cache, feats = tiny_cache(tmp_path)
    timing, pattern = make_models()
    g = E.eval_generated(timing, pattern, cache, feats, MASK, "test", 5, 0, "cpu")
    assert g["n"] == 2  # one difficulty per song
    assert g["hard_violations_generated"] == 0
    assert 0.0 <= g["density_hit_share"] <= 1.0
    assert set(g["tv"]) == {"column", "row", "direction", "hand"}
    assert 0.0 <= g["skipped_rate"] <= 1.0


def test_render_report_has_all_sections(tmp_path):
    cache, feats = tiny_cache(tmp_path)
    timing, pattern = make_models()
    res = {"timing": E.eval_timing(timing, cache, feats, "test", 2, "cpu"),
           "pattern": E.eval_pattern(pattern, cache, feats, "test", 2, "cpu"),
           "generated": E.eval_generated(timing, pattern, cache, feats, MASK, "test", 2, 0, "cpu"),
           "setup": {"timing_checkpoint": "t.pt", "timing_step": 1, "pattern_checkpoint": "p.pt",
                     "pattern_step": 2, "split": "test", "n": 2, "n_generated": 2, "seed": 0}}
    text = E.render_eval_report(res)
    heads = [text.index(h) for h in ("## Summary", "## Timing", "## Pattern",
                                     "## Generated vs human", "## Setup")]
    assert heads == sorted(heads)
    assert "single density round" in text and "| F1 (25 ms) |" in text


def test_cli_eval_bad_split_and_missing_checkpoint(tmp_path, monkeypatch):
    from typer.testing import CliRunner

    from bsmaker.cli import app
    (tmp_path / "bsmaker.toml").write_text(
        f"custom_levels = '{tmp_path / 'levels'}'\ndata_dir = '{tmp_path / 'data'}'\n",
        encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    assert CliRunner().invoke(app, ["eval", "--split", "train"]).exit_code == 2
    r = CliRunner().invoke(app, ["eval"])
    assert r.exit_code == 1 and "checkpoint not found" in r.output


def test_cli_eval_success(tmp_path, monkeypatch):
    import json
    from dataclasses import asdict

    from typer.testing import CliRunner

    from bsmaker.cli import app
    from bsmaker.models.common import save_checkpoint
    data = tmp_path / "data"
    cache, feats = tiny_cache(tmp_path)
    (data / "events").mkdir(parents=True)
    import shutil
    shutil.copytree(feats, data / "features")
    np.savez(data / "events" / "events.npz", time_s=cache._t, left=cache._l, right=cache._r,
             offsets=cache._o)
    (data / "events" / "index.json").write_text(json.dumps([
        {"map_key": e.map_key, "characteristic": e.characteristic, "rank": e.rank,
         "split": "val", "nps": e.nps, "styles": list(e.styles), "njs": e.njs,
         "njs_offset": e.njs_offset, "n_frames": e.n_frames} for e in cache.entries]))
    t, p = make_models()
    for name, m in (("timing", t), ("pattern", p)):
        save_checkpoint(data / "checkpoints" / name / "best.pt", {
            "model": m.model.state_dict(), "model_config": asdict(m.config), "stats": STATS,
            "step": 7, "best": {}})
    (tmp_path / "bsmaker.toml").write_text(
        f"custom_levels = '{tmp_path / 'levels'}'\ndata_dir = '{data}'\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    r = CliRunner().invoke(app, ["eval", "--device", "cpu", "--n", "2", "--n-generated", "2"])
    assert r.exit_code == 0, r.output
    text = (data / "reports" / "eval-val.md").read_text(encoding="utf-8")
    for h in ("## Summary", "## Timing", "## Pattern", "## Generated vs human", "## Setup"):
        assert h in text
    assert "## Summary" in r.output
    assert not list((data / "reports").glob("*.tmp"))


def test_eval_entries_restrict_all_evaluations(tmp_path):
    cache, feats = tiny_cache(tmp_path)
    timing, pattern = make_models()
    keys = {("a", "Standard", "Expert")}
    pool = E.select_entries(cache, "test", keys)
    assert [e.rank for e in pool] == ["Expert"]
    assert E.eval_timing(timing, cache, feats, "test", 10, "cpu", pool)["n"] == 1
    assert E.eval_pattern(pattern, cache, feats, "test", 10, "cpu", pool)["n"] == 1
    assert E.eval_generated(timing, pattern, cache, feats, MASK, "test", 10, 0, "cpu", pool)["n"] == 1
    sp = E.eval_pattern_sampled(pattern, cache, feats, MASK, "test", 10, 0, "cpu", pool)
    assert sp["n"] == 1 and sp["hard_violations"] == 0


def test_f1_10_not_above_f1_25(tmp_path):
    cache, feats = tiny_cache(tmp_path)
    timing, _ = make_models()
    r = E.eval_timing(timing, cache, feats, "test", 3, "cpu")
    assert r["f1_10"] <= r["f1"] + 1e-12
    assert all(v["f1_10"] <= v["f1"] + 1e-12 for v in r["f1_by_nps_bin"].values())


def test_row_dir_tv_zero_identical_one_disjoint():
    a = [[Event(1.0, encode_token(1, 0, 0), NONE_TOKEN)]]
    b = [[Event(1.0, encode_token(1, 2, 1), NONE_TOKEN)]]
    ta, na = E.row_dir_shares(a)
    tb, _ = E.row_dir_shares(b)
    assert na == 1
    assert E.tv_distance(np.ravel(ta), np.ravel(ta)) == 0.0
    assert abs(E.tv_distance(np.ravel(ta), np.ravel(tb)) - 1.0) < 1e-12


def test_sampled_shape_and_json_safe(tmp_path):
    import json
    cache, feats = tiny_cache(tmp_path)
    _, pattern = make_models()
    sp = E.eval_pattern_sampled(pattern, cache, feats, MASK, "test", 5, 0, "cpu")
    assert sp["n"] == 2 and len(sp["generated"]) == 3 and len(sp["human"][0]) == 9
    assert sp["n_notes_human"] > 0 and 0.0 <= sp["tv_row_dir"] <= 1.0
    assert json.loads(json.dumps(E.json_safe({"x": float("nan"), "y": np.float32(1.5)}))) == \
        {"x": None, "y": 1.5}


def _cli_setup(tmp_path, monkeypatch, extra_train_options=False):
    import json
    import shutil
    from dataclasses import asdict

    from bsmaker.models.common import save_checkpoint
    data = tmp_path / "data"
    cache, feats = tiny_cache(tmp_path)
    (data / "events").mkdir(parents=True)
    shutil.copytree(feats, data / "features")
    np.savez(data / "events" / "events.npz", time_s=cache._t, left=cache._l, right=cache._r,
             offsets=cache._o)
    (data / "events" / "index.json").write_text(json.dumps([
        {"map_key": e.map_key, "characteristic": e.characteristic, "rank": e.rank,
         "split": "val", "nps": e.nps, "styles": list(e.styles), "njs": e.njs,
         "njs_offset": e.njs_offset, "n_frames": e.n_frames} for e in cache.entries]))
    t, p = make_models()
    for name, m in (("timing", t), ("pattern", p)):
        payload = {"model": m.model.state_dict(), "model_config": asdict(m.config),
                   "stats": STATS, "step": 7, "best": {}}
        if extra_train_options:
            payload["train_options"] = {"lr": 0.001}
        save_checkpoint(tmp_path / f"{name}-custom.pt", payload)
    (tmp_path / "bsmaker.toml").write_text(
        f"custom_levels = '{tmp_path / 'levels'}'\ndata_dir = '{data}'\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    return data


def test_cli_eval_custom_ckpts_entries_tag_and_json(tmp_path, monkeypatch):
    import json

    from typer.testing import CliRunner

    from bsmaker.cli import app
    data = _cli_setup(tmp_path, monkeypatch, extra_train_options=True)
    sub = tmp_path / "sub.json"
    sub.write_text(json.dumps([{"map_key": "a", "characteristic": "Standard", "rank": "Hard"},
                               {"map_key": "b", "characteristic": "Standard", "rank": "Hard"}]))
    args = ["eval", "--split", "val", "--device", "cpu", "--n", "5", "--n-generated", "2",
            "--n-sampled", "2", "--timing-ckpt", str(tmp_path / "timing-custom.pt"),
            "--pattern-ckpt", str(tmp_path / "pattern-custom.pt"),
            "--entries-from", str(sub), "--tag", "run-1.a"]
    r = CliRunner().invoke(app, args)
    assert r.exit_code == 0, r.output
    md = (data / "reports" / "eval-val-run-1.a.md").read_text(encoding="utf-8")
    assert "timing-custom.pt" in md and "pattern-custom.pt" in md
    assert "kept 2 of 3" in md and "train_options" in md and "model_config" in md
    assert "## Pattern on human event times" in md and "F1 at 10 ms" in r.output
    js = json.loads((data / "reports" / "eval-val-run-1.a.json").read_text(encoding="utf-8"))
    assert js["timing"]["n"] == 2 and "f1_10" in js["timing"]
    assert "tv_row_dir" in js["pattern_sampled"]
    assert js["setup"]["entries_kept"] == 2


def test_cli_eval_default_names_and_skip_sampled(tmp_path, monkeypatch):
    from typer.testing import CliRunner

    from bsmaker.cli import app
    data = _cli_setup(tmp_path, monkeypatch)
    import shutil
    for name in ("timing", "pattern"):
        (data / "checkpoints" / name).mkdir(parents=True)
        shutil.copy(tmp_path / f"{name}-custom.pt", data / "checkpoints" / name / "best.pt")
    r = CliRunner().invoke(app, ["eval", "--device", "cpu", "--n", "2", "--n-generated", "2",
                                 "--n-sampled", "0"])
    assert r.exit_code == 0, r.output
    assert (data / "reports" / "eval-val.md").is_file() and (data / "reports" / "eval-val.json").is_file()
    assert "## Pattern on human event times" not in (data / "reports" / "eval-val.md").read_text("utf-8")


def test_cli_eval_errors(tmp_path, monkeypatch):
    from typer.testing import CliRunner

    from bsmaker.cli import app
    _cli_setup(tmp_path, monkeypatch)
    ck = ["--timing-ckpt", str(tmp_path / "timing-custom.pt"),
          "--pattern-ckpt", str(tmp_path / "pattern-custom.pt")]
    assert CliRunner().invoke(app, ["eval", "--tag", "bad/tag"] + ck).exit_code == 2
    r = CliRunner().invoke(app, ["eval", "--timing-ckpt", str(tmp_path / "nope.pt")])
    assert r.exit_code == 1 and "checkpoint not found" in r.output
    none = tmp_path / "none.json"
    none.write_text("[]")
    r = CliRunner().invoke(app, ["eval", "--device", "cpu", "--entries-from", str(none)] + ck)
    assert r.exit_code == 1 and "no val difficulty" in r.output
