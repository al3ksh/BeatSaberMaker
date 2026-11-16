from dataclasses import asdict

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from bsmaker.data.events_cache import Entry
from bsmaker.formats.model import Note
from bsmaker.formats.tokens import NONE_TOKEN
from bsmaker.models.common import save_checkpoint
from bsmaker.models.layers.spec import LIGHTS, LayerSpec, channel_mask, decor_spec
from bsmaker.models.layers.what import (
    WhatConfig, WhatModel, WhatSequences, load_what, note_context, sample_layer_values,
    what_better, what_eval, what_step)

STATS = {"mel_mean": 0.0, "mel_std": 1.0}
N_FRAMES = 1300
TINY = dict(d_model=32, n_layers=2, n_heads=2, context=16, dropout=0.0)
TOY = LayerSpec("toy", 2, 6, ((0, 1, 2), (3, 4, 5)), (2, 2), ("a", "b"), lambda m: True,
                lambda c, i: c.per[i])


class FakeLayers:
    def __init__(self, per):
        self.per = per

    def meta(self, i):
        return {}


class FakeEvents:
    def __init__(self, per):
        self.per = per

    def events(self, i):
        return self.per[i]


def _toy(tmp_path, n_songs, seed, split="train", base=0, n_events=40):
    """Alternating channels; each channel's token is the previous one of that channel + 1."""
    rng = np.random.default_rng(seed)
    layers, events, entries = {}, {}, []
    for k in range(n_songs):
        i = base + k
        np.save(tmp_path / f"s{i}.npy", np.zeros((N_FRAMES, 80), np.float16))
        t = 1.0 + 0.3 * np.arange(n_events)
        ch = np.arange(n_events) % 2
        tok = np.zeros(n_events, int)
        last = [int(rng.integers(0, 3)), int(rng.integers(0, 3))]
        for j in range(n_events):
            last[ch[j]] = (last[ch[j]] + 1) % 3
            tok[j] = last[ch[j]] + 3 * ch[j]
        layers[i] = (t, ch, tok)
        nt = t[::2].astype(np.float32)
        events[i] = (nt, np.full(len(nt), 5, np.uint8), np.full(len(nt), NONE_TOKEN, np.uint8))
        entries.append(Entry(i, f"s{i}", "Standard", "Expert", split, 3.0, ("tech",), 18.0,
                             0.0, N_FRAMES))
    return layers, events, entries


def _model_ckpt(tmp_path, layer, spec, name=None, extra_stats=None):
    cfg = WhatConfig(n_channels=spec.n_channels, vocab_size=spec.vocab_size, **TINY)
    torch.manual_seed(0)
    model = WhatModel(cfg, channel_mask(spec))
    path = tmp_path / f"{name or layer}.pt"
    save_checkpoint(path, {"model": model.state_dict(), "model_config": asdict(cfg),
                           "stats": dict(STATS, **(extra_stats or {})),
                           "train_options": {"layer": layer}})
    return path


def _mel(n=900):
    return np.random.default_rng(0).normal(size=(n, 80)).astype(np.float32)


def test_note_context_counts_within_a_quarter_second():
    ctx = note_context(np.array([1.0, 5.0]), np.array([0.8, 1.2, 1.3]), np.array([]))
    assert ctx.shape == (2, 2) and ctx.dtype == np.float32
    assert ctx[0, 0] == pytest.approx(np.log1p(2)) and ctx[0, 1] == 0 and ctx[1].sum() == 0


def test_dataset_item_shapes_and_padding(tmp_path):
    layers, events, entries = _toy(tmp_path, 1, 0, n_events=10)
    ds = WhatSequences(FakeLayers(layers), FakeEvents(events), entries, tmp_path, STATS, TOY,
                       16, seed=0)
    it = ds[0]
    assert it["mel_slices"].shape == (16, 35, 80) and it["gaps"].shape == (16, 5)
    assert it["valid"].sum() == 10 and (it["target"][10:] == -100).all()
    assert it["prev"][0] == 6 and it["prev"][1] == it["target"][0]  # BOS first
    model = WhatModel(WhatConfig(2, 6, **TINY), channel_mask(TOY))
    batch = {k: v.unsqueeze(0) for k, v in it.items()}
    assert torch.isfinite(what_step(model, batch))
    empty = WhatSequences(FakeLayers({0: (np.zeros(0), np.zeros(0, int), np.zeros(0, int))}),
                          FakeEvents({0: (np.zeros(0, np.float32),) * 3}), entries, tmp_path,
                          STATS, TOY, 16, seed=0)[0]
    assert not empty["valid"].any()
    assert what_step(model, {k: v.unsqueeze(0) for k, v in empty.items()}).item() == 0.0


def test_learns_same_channel_rule(tmp_path):
    torch.set_num_threads(1)  # tiny CPU models run far faster single-threaded
    torch.manual_seed(0)
    layers, events, entries = _toy(tmp_path, 12, 1)
    vl, ve, vent = _toy(tmp_path, 6, 2, split="val", base=20)
    ds = WhatSequences(FakeLayers(layers), FakeEvents(events), entries, tmp_path, STATS, TOY,
                       16, seed=0)
    val = WhatSequences(FakeLayers(vl), FakeEvents(ve), vent, tmp_path, STATS, TOY, 16, seed=1,
                        fixed=True)
    model = WhatModel(WhatConfig(2, 6, **TINY), channel_mask(TOY))
    before = what_eval(model, DataLoader(val, batch_size=6))["ce"]
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    for _ in range(75):  # 4 steps per pass over the 12 songs
        for batch in DataLoader(ds, batch_size=4, shuffle=True, drop_last=True):
            loss = what_step(model, batch)
            opt.zero_grad()
            loss.backward()
            opt.step()
    model.eval()
    after = what_eval(model, DataLoader(val, batch_size=6))["ce"]
    assert after < 0.4 < before, (before, after)
    assert what_better({"ce": 0.1}, {"ce": 0.2}) and not what_better({"ce": 0.3}, {"ce": 0.2})
    assert what_better({"ce": 1.0}, None)


def test_tokens_stay_in_their_channel(tmp_path):
    loaded = load_what(_model_ckpt(tmp_path, "lights", LIGHTS), "cpu")
    times = np.repeat(np.arange(1.5, 21.5, 0.5), 9)[:360]
    chans = np.tile(np.arange(9), 40)
    for seed in range(3):
        tok, skipped = sample_layer_values(loaded, times, chans, _mel(), [], 3.0, seed, "cpu",
                                           temperature=3.0, top_p=1.0)
        assert skipped == 0 and tok.shape == (360,)
        for c in range(9):
            assert set(tok[chans == c].tolist()) <= set(LIGHTS.channel_tokens[c])
    assert set(tok[chans == 5].tolist()) == {17}


def test_decor_bomb_vocabulary_travels_with_checkpoint(tmp_path):
    spec = decor_spec({"bomb_vocab": [5, 6, 7]})
    loaded = load_what(_model_ckpt(tmp_path, "decor", spec, extra_stats={"bomb_masks": [5, 6, 7]}),
                       "cpu")
    assert loaded.bomb_masks == [5, 6, 7] and loaded.spec.vocab_size == 27
    assert loaded.config.vocab_size == 27 and loaded.config.n_channels == 3
    chans = np.tile(np.arange(3), 20)
    tok, _ = sample_layer_values(loaded, np.linspace(2, 20, 60), chans, _mel(), [], 2.0, 0,
                                 "cpu", temperature=2.0)
    assert (tok[chans == 2] >= 24).all() and (tok[chans < 2] < 24).all()


def test_allowed_is_honoured_and_skips_are_reported(tmp_path):
    loaded = load_what(_model_ckpt(tmp_path, "lights", LIGHTS), "cpu")
    times = np.linspace(2.0, 20.0, 30)
    chans = np.zeros(30, dtype=int)
    seen = []

    def allowed(i, t, c):
        seen.append((i, round(t, 6), c))
        m = np.ones(18, dtype=bool)
        m[0] = False
        if i in (4, 11):
            m[:] = False
        return m

    tok, skipped = sample_layer_values(loaded, times, chans, _mel(), [], 3.0, 0, "cpu",
                                       allowed=allowed, temperature=2.0)
    assert skipped == 2 and tok[4] == -1 and tok[11] == -1
    assert (np.delete(tok, [4, 11]) > 0).all()
    assert [s[0] for s in seen] == list(range(30)) and seen[3] == (3, round(times[3], 6), 0)


def test_determinism_order_and_empty_input(tmp_path):
    loaded = load_what(_model_ckpt(tmp_path, "lights", LIGHTS), "cpu")
    times = np.repeat(np.arange(2.0, 14.0, 0.4), 3)
    chans = np.tile([0, 1, 7], len(times) // 3)
    notes = [Note(2.0 + 0.3 * k, 0.0, 1, 1, k % 2, 1) for k in range(30)]
    kw = dict(mel=_mel(), notes=notes, nps=4.0, device="cpu", temperature=2.0)
    a, _ = sample_layer_values(loaded, times, chans, seed=5, **kw)
    b, _ = sample_layer_values(loaded, times, chans, seed=5, **kw)
    c, _ = sample_layer_values(loaded, times, chans, seed=6, **kw)
    assert (a == b).all() and not (a == c).all()
    rev, _ = sample_layer_values(loaded, times[::-1], chans[::-1], seed=5, **kw)
    assert (rev[::-1] == a).all()  # tokens come back in the order the events were given
    tok, skipped = sample_layer_values(loaded, [], [], _mel(), [], 3.0, 0, "cpu")
    assert tok.shape == (0,) and skipped == 0
    with pytest.raises(ValueError):
        sample_layer_values(loaded, [1.0], [9], _mel(), [], 3.0, 0, "cpu")


@pytest.mark.parametrize("notes", [[], [Note(1.5 + 0.05 * k, 0.0, 1, 1, k % 2, 1)
                                        for k in range(600)]])
def test_short_song_sparse_and_dense_notes(tmp_path, notes):
    loaded = load_what(_model_ckpt(tmp_path, "lights", LIGHTS), "cpu")
    mel = np.zeros((int(31 * 86.13), 80), np.float32)
    times = np.linspace(1.6, 30.0, 40)
    tok, skipped = sample_layer_values(loaded, times, np.arange(40) % 9, mel, notes, 6.0, 1,
                                       "cpu")
    assert skipped == 0 and (tok >= 0).all()
