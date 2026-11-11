import pickle

import numpy as np
import pytest
import torch

from bsmaker.data.events_cache import Entry
from bsmaker.models.common import STYLE_DIM
from bsmaker.models.timing.dataset import TimingWindows, frame_targets
from bsmaker.models.timing.model import TimingConfig, TimingModel, timing_loss, timing_step

FPS = 22050 / 256
STATS = {"mel_mean": 0.0, "mel_std": 1.0}
TINY = TimingConfig(d_model=32, n_layers=2, n_heads=4, window=64, dropout=0.0)


class FakeCache:
    def __init__(self, times):
        self.times = times

    def events(self, i):
        return self.times[i % len(self.times)], None, None


def _entry(i, key="m", styles=("tech",), n_frames=200, nps=3.0):
    return Entry(i, key, "Standard", "Expert", "train", nps, styles, 18.0, 0.0, n_frames)


def _feats(tmp_path, key="m", n=200):
    np.save(tmp_path / f"{key}.npy", np.zeros((n, 80), np.float16))


def test_timing_forward_shape():
    out = TimingModel(TINY)(torch.zeros(2, 128, 80), torch.zeros(2), torch.zeros(2, STYLE_DIM))
    assert out.shape == (2, 128)


def test_frame_targets_fuzzy():
    t = frame_targets(np.array([10 / FPS]), 20)
    assert t.dtype == np.float32
    assert t[9:12].tolist() == [0.25, 1.0, 0.25] and t.sum() == pytest.approx(1.5)
    t = frame_targets(np.array([10 / FPS, 11 / FPS]), 20)
    assert t[9:13].tolist() == [0.25, 1.0, 1.0, 0.25]


def test_window_padding_masked_in_loss(tmp_path):
    _feats(tmp_path, n=30)
    ds = TimingWindows(FakeCache([np.array([5 / FPS])]), [_entry(0, n_frames=30)], tmp_path,
                       STATS, window=64, seed=0)
    item = ds[0]
    assert item["valid"].sum() == 30 and item["mel"][30:].abs().sum() == 0
    logits = torch.zeros(1, 64)
    junk = logits.clone()
    junk[0, 30:] = 50.0  # only padded frames differ
    v = item["valid"][None]
    a = timing_loss(logits, item["target"][None], valid=v)
    b = timing_loss(junk, item["target"][None], valid=v)
    assert torch.equal(a, b)
    assert not torch.equal(timing_loss(logits, item["target"][None]),
                           timing_loss(junk, item["target"][None]))


def test_style_dropout_rate(tmp_path):
    _feats(tmp_path)
    n = 2000
    ds = TimingWindows(FakeCache([np.array([1.0])]), [_entry(i) for i in range(n)],
                       tmp_path, STATS, window=64, seed=7, style_dropout=0.15)
    unknown = sum(float(ds[i]["styles"][-1]) for i in range(n))
    assert 0.12 < unknown / n < 0.18
    unstyled = TimingWindows(FakeCache([np.array([1.0])]), [_entry(0, styles=("unknown",))],
                             tmp_path, STATS, window=64, seed=7, style_dropout=0.15)
    assert float(unstyled[0]["styles"][-1]) == 1.0


def test_window_randomness_and_fixed(tmp_path):
    _feats(tmp_path)
    cache, entries = FakeCache([np.arange(1, 150) / FPS * 1.0]), [_entry(0)]
    a = TimingWindows(cache, entries, tmp_path, STATS, 64, seed=1)
    b = TimingWindows(cache, entries, tmp_path, STATS, 64, seed=1)
    seq_a = [a[0]["target"] for _ in range(5)]
    seq_b = [b[0]["target"] for _ in range(5)]
    assert all(torch.equal(x, y) for x, y in zip(seq_a, seq_b))  # reproducible from the seed
    c = TimingWindows(cache, entries, tmp_path, STATS, 64, seed=2)
    assert not all(torch.equal(x, c[0]["target"]) for x in seq_a)
    f = TimingWindows(cache, entries, tmp_path, STATS, 64, seed=1, fixed=True)
    first = f[0]["target"]
    assert all(torch.equal(first, f[0]["target"]) for _ in range(5))


def test_window_targets_follow_the_window(tmp_path):
    np.save(tmp_path / "m.npy", np.arange(200, dtype=np.float16)[:, None].repeat(80, 1))
    ds = TimingWindows(FakeCache([np.array([100 / FPS])]), [_entry(0)], tmp_path, STATS,
                       window=64, seed=3, fixed=True)
    item = ds[0]
    start = int(item["mel"][0, 0])
    expected = 100 - start
    if 0 <= expected < 64:
        assert item["target"][expected] == 1.0


def test_dataset_pickles_without_cache(tmp_path):
    _feats(tmp_path)
    ds = TimingWindows(FakeCache([np.array([1.0])]), [_entry(0)], tmp_path, STATS, 64, seed=0,
                       events_dir=tmp_path / "events")
    assert pickle.loads(pickle.dumps(ds))._cache is None


def test_overfit_two_maps(tmp_path):
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    times, entries = [], []
    for i, frames in enumerate([[10, 40, 90], [20, 55, 100, 120]]):
        mel = rng.normal(0, 0.1, (128, 80)).astype(np.float16)
        mel[frames] += 3.0
        np.save(tmp_path / f"m{i}.npy", mel)
        times.append(np.array(frames) / FPS)
        entries.append(_entry(i, key=f"m{i}", n_frames=128))
    ds = TimingWindows(FakeCache(times), entries, tmp_path, STATS, window=128, seed=0,
                       style_dropout=0.0)
    items = [ds[0], ds[1]]
    batch = {k: torch.stack([it[k] for it in items]) for k in items[0]}
    model = TimingModel(TimingConfig(d_model=32, n_layers=2, n_heads=4, window=128, dropout=0.0))
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    loss = None
    for _ in range(300):
        loss = timing_step(model, batch)
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < 0.05
