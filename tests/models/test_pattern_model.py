import numpy as np
import pytest
import torch

from bsmaker.data.events_cache import Entry
from bsmaker.models.common import STYLE_DIM
from bsmaker.models.pattern.dataset import PatternSequences, event_inputs
from bsmaker.models.pattern.model import (
    BOS_TOKEN, PatternConfig, PatternModel, pattern_loss)

FPS = 22050 / 256
STATS = {"mel_mean": 0.0, "mel_std": 1.0}
TINY = PatternConfig(d_model=32, n_layers=2, n_heads=4, context=16, dropout=0.0)


def _inputs(b=2, n=8, seed=0):
    g = torch.Generator().manual_seed(seed)
    return dict(
        mel_slices=torch.randn(b, n, 35, 80, generator=g), gaps=torch.rand(b, n, 5, generator=g),
        prev_left=torch.randint(0, 109, (b, n), generator=g),
        prev_right=torch.randint(0, 109, (b, n), generator=g),
        log_nps=torch.rand(b, generator=g), styles=torch.rand(b, STYLE_DIM, generator=g))


class FakeCache:
    def __init__(self, ev):
        self.ev = ev

    def events(self, i):
        return self.ev


def _entry(n_frames=400):
    return Entry(0, "m", "Standard", "Expert", "train", 3.0, ("tech",), 18.0, 0.0, n_frames)


def _ds(tmp_path, times, left, right, context=4, n_frames=400, **kw):
    np.save(tmp_path / "m.npy", np.random.default_rng(0).normal(size=(n_frames, 80))
            .astype(np.float16))
    ev = (np.array(times, np.float32), np.array(left, np.uint8), np.array(right, np.uint8))
    return PatternSequences(FakeCache(ev), [_entry(n_frames)], tmp_path, STATS, context,
                            seed=0, fixed=True, **kw)


def test_pattern_shapes():
    m = PatternModel(TINY).eval()
    x = _inputs()
    h = m.encode(**x)
    assert h.shape == (2, 8, 32)
    assert m.left_logits(h).shape == (2, 8, 109)
    assert m.right_logits(h, torch.zeros(2, 8, dtype=torch.long)).shape == (2, 8, 109)


def test_encode_is_causal():
    m = PatternModel(TINY).eval()
    x = _inputs()
    base = m.encode(**x)
    y = {k: v.clone() for k, v in x.items()}
    y["mel_slices"][:, 5] += 1.0
    y["gaps"][:, 5] += 1.0
    y["prev_left"][:, 5] = (y["prev_left"][:, 5] + 1) % 109
    y["prev_right"][:, 5] = (y["prev_right"][:, 5] + 1) % 109
    out = m.encode(**y)
    assert torch.allclose(base[:, :5], out[:, :5], atol=1e-6)
    assert not torch.allclose(base[:, 5:], out[:, 5:], atol=1e-6)


def test_right_depends_on_left():
    m = PatternModel(TINY).eval()
    h = m.encode(**_inputs())
    a = m.right_logits(h, torch.zeros(2, 8, dtype=torch.long))
    b = m.right_logits(h, torch.full((2, 8), 5))
    assert not torch.allclose(a, b)


def test_gaps_layout():
    _, g = event_inputs(np.array([1.0, 1.5, 2.5]), np.zeros((300, 80), np.float16), STATS)
    assert g.shape == (3, 5)
    assert g[0, 0] == 0
    assert g[1, 0] == pytest.approx(np.log1p(0.5))
    assert g[0, 1] == pytest.approx(np.log1p(0.5)) and g[0, 2] == pytest.approx(np.log1p(1.5))
    assert (g[0, 3:] == 0).all()
    assert (g[2, 1:] == 0).all()


def test_slices_padded_at_edges():
    mel = np.ones((100, 80), np.float16)
    s, _ = event_inputs(np.array([0.0, 99 / FPS]), mel, STATS)
    assert s.shape == (2, 35, 80)
    assert (s[0, :17] == 0).all() and (s[0, 17:] == 1).all()
    assert (s[1, :18] == 1).all() and (s[1, 18:] == 0).all()


def test_prev_tokens_are_shifted_targets(tmp_path):
    ds = _ds(tmp_path, [1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12])
    it = ds[0]  # run starts at the first event of the song
    assert it["prev_left"].tolist() == [BOS_TOKEN, 5, 6, 7]
    assert it["prev_right"].tolist() == [BOS_TOKEN, 9, 10, 11]
    assert it["left"].tolist() == [5, 6, 7, 8]
    assert it["gaps"][0, 0] == 0


class _Rng:
    """Stand-in generator: forces the run start."""

    def __init__(self, start):
        self.start = start

    def integers(self, lo, hi):
        assert lo <= self.start < hi
        return self.start

    def random(self):
        return 1.0


def test_mid_song_run_uses_true_neighbours(tmp_path, monkeypatch):
    times = [1.0, 1.5, 2.5, 2.75, 4.0, 4.5, 6.0, 6.25, 7.0]
    left = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    right = [11, 12, 13, 14, 15, 16, 17, 18, 19]
    ds = _ds(tmp_path, times, left, right, context=4)
    monkeypatch.setattr("bsmaker.models.pattern.dataset.np.random.default_rng",
                        lambda *_: _Rng(3))
    it = ds[0]  # events 3..6 of the song
    assert it["left"].tolist() == [4, 5, 6, 7]
    assert it["prev_left"].tolist() == [3, 4, 5, 6]
    assert it["prev_right"].tolist() == [13, 14, 15, 16]
    g = it["gaps"]
    assert g[0, 0].item() == pytest.approx(np.log1p(times[3] - times[2]))
    last = 6  # song index of the last position; events 7 and 8 follow, then none
    assert g[3, 1].item() == pytest.approx(np.log1p(times[7] - times[last]))
    assert g[3, 2].item() == pytest.approx(np.log1p(times[8] - times[last]))
    assert g[3, 1] > 0 and g[3, 2] > 0
    assert (g[3, 3:] == 0).all()
    assert g[0, 4].item() == pytest.approx(np.log1p(times[7] - times[3]))


def test_no_event_with_both_hands_empty_in_dataset(tmp_path):
    ds = _ds(tmp_path, [1, 2, 3, 4], [0, 5, 0, 7], [0, 0, 0, 2])
    it = ds[0]
    assert it["valid"].sum() == 2
    assert it["left"][:2].tolist() == [5, 7] and it["right"][:2].tolist() == [0, 2]
    assert it["prev_left"][1] == 5


def test_overfit_two_sequences():
    torch.manual_seed(0)
    m = PatternModel(TINY)
    x = _inputs(b=2, n=8)
    g = torch.Generator().manual_seed(1)
    batch = dict(x, left=torch.randint(0, 109, (2, 8), generator=g),
                 right=torch.randint(0, 109, (2, 8), generator=g),
                 valid=torch.ones(2, 8, dtype=torch.bool))
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3)
    for _ in range(400):
        loss, parts = pattern_loss(m, batch)
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < 0.1
    assert set(parts) == {"ce_left", "ce_right"}
