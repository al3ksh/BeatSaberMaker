import json

import numpy as np
import torch
from torch.utils.data import DataLoader
from typer.testing import CliRunner

import typer

from bsmaker.data.events_cache import Entry
from bsmaker.models.ornaments import cli as orn_cli
from bsmaker.models.ornaments.dataset import OrnamentNotes, note_gaps
from bsmaker.models.ornaments.model import (
    OrnamentConfig, OrnamentModel, ornament_eval, ornament_step)

STATS = {"mel_mean": 0.0, "mel_std": 1.0}
TINY = OrnamentConfig(d_model=32, n_layers=2, n_heads=2, context=16, dropout=0.0)


class FakeLayers:
    """Minimal LayerCache stand-in: chains on every note whose gap to the next note exceeds 1 s."""

    def __init__(self, n_entries=4, n_notes=24, arcs=True, seed=0):
        rng = np.random.default_rng(seed)
        self.rows = []
        for _ in range(n_entries):
            gaps = rng.choice([0.4, 1.5], size=n_notes)
            t = np.cumsum(gaps).astype(np.float32)
            nxt = np.append(np.diff(t), 0.0)
            chain = np.where(nxt > 1.0, 3, 0).astype(np.uint8)
            arc_len = np.zeros(n_notes, np.uint8)
            if arcs:
                arc_len[::7] = 2
            hand = (np.arange(n_notes) % 2).astype(np.uint8)
            token = rng.integers(1, 109, n_notes).astype(np.uint8)
            cell = np.where(arc_len > 0, 5, 0).astype(np.uint8)
            d = np.where(arc_len > 0, 3, 0).astype(np.uint8)
            self.rows.append((t, hand, token, chain, arc_len, cell, d))

    def ornaments(self, i):
        return self.rows[i]

    def meta(self, i):
        r = self.rows[i]
        return {"has_v3": True, "n_arcs": int((r[4] > 0).sum()), "n_chains": int((r[3] > 0).sum())}


def _entries(n, split="train"):
    return [Entry(i, f"m{i}", "Standard", "Expert", split, 3.0, ("tech",), 18.0, 0.0, 5000)
            for i in range(n)]


def _features(tmp_path, n):
    for i in range(n):
        np.save(tmp_path / f"m{i}.npy", np.zeros((5000, 80), np.float16))


def _ds(tmp_path, layers, entries, context=16, fixed=False):
    return OrnamentNotes(layers, entries, tmp_path, STATS, context, seed=0, fixed=fixed)


def test_batch_shapes_and_model_output(tmp_path):
    layers = FakeLayers()
    _features(tmp_path, 4)
    ds = _ds(tmp_path, layers, _entries(4))
    batch = next(iter(DataLoader(ds, batch_size=3)))
    assert batch["mel_slices"].shape == (3, 16, 35, 80)
    assert batch["gaps"].shape == (3, 16, 4)
    model = OrnamentModel(TINY)
    out = model(batch["mel_slices"], batch["gaps"], batch["hand"], batch["token"],
                batch["log_nps"], batch["valid"])
    assert out["chain"].shape == (3, 16, 6)
    assert out["arc_len"].shape == (3, 16, 7)
    assert out["arc_cell"].shape == (3, 16, 12)
    assert out["arc_dir"].shape == (3, 16, 9)


def test_dataset_filters_entries_without_ornaments(tmp_path):
    layers = FakeLayers(arcs=False)
    layers.rows[1] = tuple(np.zeros_like(a) for a in layers.rows[1])
    ds = _ds(tmp_path, layers, _entries(4))
    assert [e.i for e in ds.entries] == [0, 2, 3]


def test_gaps_same_hand_and_any_hand():
    g = note_gaps(np.array([0.0, 1.0, 3.0]), np.array([0, 1, 0]))
    assert np.allclose(g[1], np.log1p([1.0, 2.0, 0.0, 0.0]))  # no same-hand neighbours
    assert np.allclose(g[0], np.log1p([0.0, 1.0, 0.0, 3.0]))
    assert np.allclose(g[2], np.log1p([2.0, 0.0, 3.0, 0.0]))


def test_padded_positions_do_not_change_loss(tmp_path):
    layers = FakeLayers(n_notes=10)
    _features(tmp_path, 4)
    ds = _ds(tmp_path, layers, _entries(4), context=16, fixed=True)
    batch = next(iter(DataLoader(ds, batch_size=2)))
    assert not batch["valid"][:, 10:].any()
    model = OrnamentModel(TINY).eval()
    base = ornament_step(model, batch)
    other = {k: v.clone() for k, v in batch.items()}
    other["chain"][:, 10:] = 3  # garbage targets in padding
    other["arc_len"][:, 10:] = 4
    other["mel_slices"][:, 10:] = 5.0
    assert torch.allclose(ornament_step(model, other), base, atol=1e-5)


def test_arc_geometry_ignored_without_arc(tmp_path):
    layers = FakeLayers()
    _features(tmp_path, 4)
    ds = _ds(tmp_path, layers, _entries(4), fixed=True)
    batch = next(iter(DataLoader(ds, batch_size=2)))
    model = OrnamentModel(TINY).eval()
    base = ornament_step(model, batch)
    other = dict(batch)
    no_arc = batch["arc_len"] == 0
    other["arc_cell"] = torch.where(no_arc, (batch["arc_cell"] + 3) % 12, batch["arc_cell"])
    other["arc_dir"] = torch.where(no_arc, (batch["arc_dir"] + 2) % 9, batch["arc_dir"])
    assert torch.allclose(ornament_step(model, other), base, atol=1e-6)
    changed = dict(batch)
    changed["arc_cell"] = torch.where(no_arc, batch["arc_cell"], (batch["arc_cell"] + 3) % 12)
    assert not torch.allclose(ornament_step(model, changed), base, atol=1e-6)


def test_tagger_learns_gap_rule(tmp_path):
    torch.manual_seed(0)
    layers = FakeLayers(n_entries=8, n_notes=16, arcs=False)
    _features(tmp_path, 8)
    ds = _ds(tmp_path, layers, _entries(8), context=16, fixed=True)
    loader = DataLoader(ds, batch_size=8)
    model = OrnamentModel(TINY)
    before = ornament_eval(model, loader)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    batch = next(iter(loader))
    first = None
    for _ in range(150):
        model.train()
        loss = ornament_step(model, batch)
        first = first if first is not None else float(loss.detach())
        opt.zero_grad()
        loss.backward()
        opt.step()
    after = ornament_eval(model, loader)
    assert float(loss) < first * 0.5
    assert after["loss"] < before["loss"]
    assert after["chain_f1"] > 0.9


def _write_cli_setup(tmp_path, monkeypatch):
    data = tmp_path / "data"
    for d in ("events", "layers", "features"):
        (data / d).mkdir(parents=True)
    for name in ("events/index.json", "events/events.npz", "layers/index.json", "layers/layers.npz"):
        (data / name).write_bytes(b"x")
    (tmp_path / "dataset_stats.json").write_text(json.dumps(STATS))
    (tmp_path / "bsmaker.toml").write_text(
        f'custom_levels = "{(tmp_path / "cl").as_posix()}"\ndata_dir = "{data.as_posix()}"\n')
    layers = FakeLayers(n_entries=6)
    _features(data / "features", 6)
    entries = _entries(4, "train") + [e for e in _entries(6)[4:]]
    for e in entries[4:]:
        object.__setattr__(e, "split", "val")

    class FakeEvents:
        def select(self, split):
            return [e for e in entries if e.split == split]

    monkeypatch.setattr("bsmaker.data.events_cache.EventCache.load", lambda p: FakeEvents())
    monkeypatch.setattr("bsmaker.data.layers_cache.LayerCache.load", lambda p: layers)
    monkeypatch.chdir(tmp_path)
    return data


def test_cli_smoke_run(tmp_path, monkeypatch):
    data = _write_cli_setup(tmp_path, monkeypatch)
    train_app = typer.Typer()
    orn_cli.register(train_app)
    train_app.command("noop")(lambda: None)  # keep a group so the subcommand name is required
    res = CliRunner().invoke(train_app, [
        "ornaments", "--max-steps", "3", "--batch-size", "2", "--num-workers", "0",
        "--device", "cpu", "--eval-every", "3", "--save-every", "3", "--weighting", "none"])
    assert res.exit_code == 0, res.output
    assert (data / "checkpoints" / "layers" / "ornaments" / "last.pt").is_file()
    assert "done at step 3" in res.output
