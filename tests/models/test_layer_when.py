from dataclasses import asdict, replace

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from bsmaker.audio.decode import SAMPLE_RATE
from bsmaker.audio.mel import HOP
from bsmaker.data.events_cache import Entry
from bsmaker.formats.model import Note
from bsmaker.formats.tokens import NONE_TOKEN
from bsmaker.models.common import save_checkpoint
from bsmaker.models.layers.spec import LIGHTS, LayerSpec, decor_spec, get_spec
from bsmaker.models.layers.when import (
    WhenConfig, WhenModel, WhenWindows, load_when, note_track, note_track_from_notes,
    predict_layer_events, when_better, when_eval, when_step)

STATS = {"mel_mean": 0.0, "mel_std": 1.0}
N_FRAMES = 640


def _t(frame):
    return frame * HOP / SAMPLE_RATE


class FakeLayers:
    def __init__(self, per):
        self.per = per

    def meta(self, i):
        return {"lights_ok": True, "n_walls": 1, "n_bomb_groups": 0}


class FakeEvents:
    def __init__(self, per):
        self.per = per

    def events(self, i):
        return self.per[i]


def _frames(rng, n_frames):
    out, f = [], int(rng.integers(20, 40))
    while f < n_frames - 10:
        out.append(f)
        f += int(rng.integers(10, 30))
    return np.array(out)


def _toy(tmp_path, n_songs, seed, split="train", base=0):
    """Channel 0 fires on left-hand notes, channel 1 on bright mel frames."""
    rng = np.random.default_rng(seed)
    layers, events, entries = {}, {}, []
    for k in range(n_songs):
        i = base + k
        left, bright = _frames(rng, N_FRAMES), _frames(rng, N_FRAMES)
        mel = np.zeros((N_FRAMES, 80), np.float32)
        mel[bright] = 3.0
        np.save(tmp_path / f"s{i}.npy", mel.astype(np.float16))
        t = np.concatenate([_t(left), _t(bright)])
        ch = np.concatenate([np.zeros(len(left), int), np.ones(len(bright), int)])
        order = np.lexsort((ch, t))
        layers[i] = (t[order], ch[order], np.ones(len(t), int))
        events[i] = (_t(left).astype(np.float32), np.full(len(left), 5, np.uint8),
                     np.full(len(left), NONE_TOKEN, np.uint8))
        entries.append(Entry(i, f"s{i}", "Standard", "Expert", split, 3.0, ("tech",), 18.0,
                             0.0, N_FRAMES))
    return layers, events, entries


TOY = LayerSpec("toy", 2, 2, ((0, 1), (0, 1)), (2, 2), ("a", "b"), lambda m: True,
                lambda c, i: c.per[i])
TINY = dict(d_model=32, n_layers=2, n_heads=2, window=128, dropout=0.0)


def test_note_track_marks_onsets_and_neighbours():
    tr = note_track(np.array([_t(100)]), np.array([]), 300)
    assert tr.shape == (300, 2) and tr.dtype == np.float32
    assert tr[100, 0] == 1.0 and tr[99, 0] == 0.25 and tr[101, 0] == 0.25
    assert tr[:, 1].sum() == 0 and tr[:, 0].sum() == 1.5
    notes = [Note(_t(50), 0.0, 1, 1, 0, 1), Note(_t(80), 0.0, 2, 1, 1, 1),
             Note(_t(80), 0.0, 3, 1, 0, 1)]
    tr = note_track_from_notes(notes, 100)
    assert tr[50, 0] == 1 and tr[80, 0] == 1 and tr[80, 1] == 1 and tr[50, 1] == 0
    assert note_track_from_notes([], 10).sum() == 0


class FakeLayerData:
    def lights(self, i):
        return (np.array([1.0, 2.0], np.float32), np.array([0, 3], np.uint8),
                np.array([1, 2], np.uint8))

    def walls(self, i):
        return (np.array([1.5, 3.0], np.float32), np.array([0, 1], np.uint8),
                np.array([4, 5], np.uint8))

    def bombs(self, i):
        return np.array([1.5, 2.0, 2.5], np.float32), np.array([7, 99, 9], np.uint16)

    def meta(self, i):
        return {"lights_ok": True, "n_walls": 2, "n_bomb_groups": 3}


def test_specs():
    assert LIGHTS.n_channels == 9 and LIGHTS.vocab_size == 18
    assert LIGHTS.channel_tokens[0] == tuple(range(7)) and LIGHTS.channel_tokens[5] == (17,)
    assert LIGHTS.channel_tokens[8] == tuple(range(7, 17)) and LIGHTS.min_gap_frames[0] == 2
    spec = decor_spec({"bomb_vocab": [7, 9]})
    assert spec.n_channels == 3 and spec.vocab_size == 26 and spec.min_gap_frames == (3, 3, 3)
    assert spec.channel_tokens[2] == (24, 25) and spec.channel_tokens[1] == tuple(range(24))
    t, ch, tok = spec.extract(FakeLayerData(), 0)  # mask 99 is not in the vocabulary: dropped
    assert t.tolist() == [1.5, 1.5, 2.5, 3.0]
    assert ch.tolist() == [0, 2, 2, 1] and tok.tolist() == [4, 24, 25, 5]
    assert spec.eligible({"n_walls": 0, "n_bomb_groups": 0}) is False
    assert spec.eligible({"n_walls": 0, "n_bomb_groups": 2}) is True
    assert get_spec("decor", {"bomb_vocab": [7]}).vocab_size == 25
    assert get_spec("lights").eligible({"lights_ok": False}) is False
    t, ch, tok = LIGHTS.extract(FakeLayerData(), 0)
    assert ch.tolist() == [0, 3] and t.dtype == np.float64


@pytest.mark.parametrize("spec", [LIGHTS, decor_spec({"bomb_vocab": [1, 2, 3]})])
def test_shapes_for_both_specs(tmp_path, spec):
    cfg = WhenConfig(n_channels=spec.n_channels, **TINY)
    model = WhenModel(cfg)
    out = model(torch.zeros(2, 128, 80), torch.zeros(2, 128, 2), torch.zeros(2))
    assert out.shape == (2, 128, spec.n_channels)
    layers, events, entries = _toy(tmp_path, 1, 0)
    fake = replace(spec, extract=lambda c, i: (c.per[i][0], c.per[i][1] % spec.n_channels,
                                               c.per[i][2]))
    ds = WhenWindows(FakeLayers(layers), FakeEvents(events), entries, tmp_path, STATS, fake,
                     128, seed=0)
    item = ds[0]
    assert item["mel"].shape == (128, 80) and item["notes"].shape == (128, 2)
    assert item["target"].shape == (128, spec.n_channels) and item["valid"].all()
    loss = when_step(model, {k: v.unsqueeze(0) for k, v in item.items()})
    assert loss.ndim == 0 and torch.isfinite(loss)


def test_learns_note_and_audio_channels(tmp_path):
    torch.set_num_threads(1)  # tiny CPU models run far faster single-threaded
    torch.manual_seed(0)
    layers, events, entries = _toy(tmp_path, 12, 1)
    vl, ve, vent = _toy(tmp_path, 4, 2, split="val", base=10)
    ds = WhenWindows(FakeLayers(layers), FakeEvents(events), entries, tmp_path, STATS, TOY,
                     128, seed=0)
    val = WhenWindows(FakeLayers(vl), FakeEvents(ve), vent, tmp_path, STATS, TOY, 128, seed=1,
                      fixed=True)
    model = WhenModel(WhenConfig(n_channels=2, **TINY))
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    loader = iter(DataLoader(ds, batch_size=4, shuffle=True, drop_last=True))
    for step in range(300):
        try:
            batch = next(loader)
        except StopIteration:
            loader = iter(DataLoader(ds, batch_size=4, shuffle=True, drop_last=True))
            batch = next(loader)
        loss = when_step(model, batch)
        opt.zero_grad()
        loss.backward()
        opt.step()
    model.eval()
    m = when_eval(model, DataLoader(val, batch_size=4), (2, 2))
    assert m["f1"] > 0.8, m
    assert when_better({"f1": 0.9}, {"f1": 0.8}) and not when_better({"f1": 0.7}, {"f1": 0.8})
    assert when_better({"f1": 0.1}, None)


def _checkpoint(tmp_path, layer, stats_extra=None):
    spec = get_spec(layer, {"bomb_vocab": [1, 2, 3]})
    cfg = WhenConfig(n_channels=spec.n_channels, **TINY)
    torch.manual_seed(0)
    model = WhenModel(cfg)
    stats = dict(STATS, **(stats_extra or {}))
    path = tmp_path / f"{layer}.pt"
    save_checkpoint(path, {"model": model.state_dict(), "model_config": asdict(cfg),
                           "stats": stats, "train_options": {"layer": layer}})
    return path


def test_load_exposes_bomb_masks_and_spec(tmp_path):
    loaded = load_when(_checkpoint(tmp_path, "decor", {"bomb_masks": [1, 2, 3]}), "cpu")
    assert loaded.bomb_masks == [1, 2, 3] and loaded.spec.vocab_size == 27
    assert loaded.config.n_channels == 3
    lights = load_when(_checkpoint(tmp_path, "lights"), "cpu")
    assert lights.bomb_masks is None and lights.spec.n_channels == 9


def test_load_when_requires_stats(tmp_path):
    cfg = WhenConfig(n_channels=LIGHTS.n_channels, **TINY)
    model = WhenModel(cfg)
    base = {"model": model.state_dict(), "model_config": asdict(cfg),
            "train_options": {"layer": "lights"}}
    path = tmp_path / "no_stats.pt"
    for bad in (base, {**base, "stats": None}, {**base, "stats": {}}):
        save_checkpoint(path, bad)
        with pytest.raises(ValueError, match="stats"):
            load_when(path, "cpu")
    save_checkpoint(path, {**base, "stats": STATS})
    assert load_when(path, "cpu").stats == STATS


def test_predict_counts_gap_and_lead_in(tmp_path):
    loaded = load_when(_checkpoint(tmp_path, "lights"), "cpu")
    rng = np.random.default_rng(0)
    mel = rng.normal(size=(int(31 * SAMPLE_RATE / HOP), 80)).astype(np.float32)
    counts = [12, 0, 5, 20, 3, 0, 0, 8, 1]
    notes = [Note(2.0 + 0.5 * k, 0.0, 1, 1, k % 2, 1) for k in range(50)]
    out = predict_layer_events(loaded, mel, notes, 3.0, counts, "cpu")
    assert len(out) == 9
    for c, (t, n) in enumerate(zip(out, counts)):
        assert t.dtype == np.float64 and abs(len(t) - n) <= 1
        if n == 0:
            assert len(t) == 0
        if len(t):
            assert t.min() >= 1.5 and (np.diff(t) >= 2 * HOP / SAMPLE_RATE - 1e-6).all()
            assert (np.diff(t) > 0).all()


@pytest.mark.parametrize("notes", [[], [Note(1.5 + 0.08 * k, 0.0, 1, 1, k % 2, 1)
                                        for k in range(400)]])
def test_predict_sparse_dense_notes_and_zero_counts(tmp_path, notes):
    loaded = load_when(_checkpoint(tmp_path, "decor", {"bomb_masks": [1, 2, 3]}), "cpu")
    mel = np.zeros((int(31 * SAMPLE_RATE / HOP), 80), np.float32)
    out = predict_layer_events(loaded, mel, notes, 6.0, [10, 10, 10], "cpu")
    assert len(out) == 3 and all(t.min(initial=2.0) >= 1.5 for t in out)
    zero = predict_layer_events(loaded, mel, notes, 6.0, [0, 0, 0], "cpu")
    assert [len(t) for t in zero] == [0, 0, 0]
    assert predict_layer_events(loaded, mel[:0], notes, 6.0, [3, 3, 3], "cpu")[0].size == 0
    short = predict_layer_events(loaded, mel[:50], notes, 6.0, [3, 3, 3], "cpu")
    assert all(len(t) == 0 for t in short)  # shorter than the lead-in
    with pytest.raises(ValueError):
        predict_layer_events(loaded, mel, notes, 6.0, [1], "cpu")
