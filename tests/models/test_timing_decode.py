import numpy as np
import pytest
import torch

from bsmaker.data.events_cache import Entry
from bsmaker.models.common import save_checkpoint
from bsmaker.models.timing.decode import (
    MIN_FIRST_EVENT_S, frame_probs, load_timing, make_timing_eval, pick_peaks, predict_events,
    threshold_for_count, timing_better)
from bsmaker.models.timing.metrics import onset_f1
from bsmaker.models.timing.model import TimingConfig, TimingModel, timing_eval

FPS = 22050 / 256
TINY = TimingConfig(d_model=32, n_layers=2, n_heads=4, window=64, dropout=0.0)
STATS = {"mel_mean": 0.0, "mel_std": 1.0, "coverage_median": 0.8, "nps_bins": [0.0, 100.0],
         "double_share_by_bin": [0.25, 0.25]}


def _peaks_signal(n_peaks=40, spacing=10):
    p = np.full(n_peaks * spacing + 10, 0.01, dtype=np.float32)
    for k in range(n_peaks):
        p[5 + k * spacing] = 0.3 + 0.6 * (k + 1) / n_peaks
    return p


def test_pick_peaks_local_maxima_and_gap():
    p = np.array([0, .5, 0, 0, .9, 0, .6, 0, 0, 0, .7, 0], dtype=np.float32)
    assert pick_peaks(p, 0.4, min_gap_frames=3).tolist() == [1, 4, 10]  # 4 and 6 are 2 apart
    assert pick_peaks(p, 0.4, min_gap_frames=2).tolist() == [1, 4, 6, 10]
    assert pick_peaks(p, 0.8).tolist() == [4]


def test_threshold_matches_count():
    p = _peaks_signal()
    for n in (10, 40):
        assert len(pick_peaks(p, threshold_for_count(p, n))) == n


def test_threshold_when_fewer_peaks_than_asked():
    p = _peaks_signal(n_peaks=5)
    t = threshold_for_count(p, 50)
    assert 0.0 <= t < 1.0
    assert t < 0.3  # every clear peak is kept
    assert len(pick_peaks(p, t)) >= 5


def test_onset_f1_perfect_and_shifted():
    true = [1.0, 2.0, 3.0]
    assert onset_f1([t + 0.020 for t in true], true) == (1.0, 1.0, 1.0)
    assert onset_f1([t + 0.030 for t in true], true) == (0.0, 0.0, 0.0)
    assert onset_f1([], true) == (0.0, 0.0, 0.0)


def test_onset_f1_one_to_one():
    p, r, f = onset_f1([1.00, 1.01], [1.0])
    assert (p, r) == (0.5, 1.0)
    assert f == pytest.approx(2 / 3)


def _model():
    torch.manual_seed(0)
    return TimingModel(TINY).eval()


def test_frame_probs_overlap_covers_tail():
    m = _model()
    for n in (200, 64, 40, 97):  # not multiples of the hop; shorter than a window
        mel = np.random.default_rng(0).normal(size=(n, 80)).astype(np.float16)
        pr = frame_probs(m, mel, 3.0, ("tech",), STATS, "cpu")
        assert pr.shape == (n,) and pr.dtype == np.float32
        assert np.isfinite(pr).all() and (pr > 0).all() and (pr < 1).all()


def test_frame_probs_single_window_matches_model():
    m = _model()
    mel = np.random.default_rng(1).normal(size=(64, 80)).astype(np.float32)
    from bsmaker.models.common import log_nps_value, style_vector
    with torch.inference_mode():
        ref = torch.sigmoid(m(torch.from_numpy(mel)[None], torch.tensor([log_nps_value(3.0)]),
                              torch.from_numpy(style_vector(("tech",)))[None]))[0].numpy()
    pr = frame_probs(m, mel, 3.0, ("tech",), STATS, "cpu")
    np.testing.assert_allclose(pr, ref, atol=1e-5)


def _loaded(tmp_path):
    m = _model()
    from dataclasses import asdict
    save_checkpoint(tmp_path / "c.pt", {"model": m.state_dict(), "model_config": asdict(TINY),
                                        "stats": STATS, "step": 1, "best": None})
    return load_timing(tmp_path / "c.pt", "cpu")


def test_load_timing_roundtrip(tmp_path):
    ld = _loaded(tmp_path)
    assert ld.config == TINY and not ld.model.training and ld.stats == STATS


def test_predict_drops_events_before_lead_in(tmp_path):
    ld = _loaded(tmp_path)
    mel = np.random.default_rng(2).normal(size=(int(20 * FPS), 80)).astype(np.float16)
    t = predict_events(ld, mel, 6.0, (), "cpu")
    assert t.dtype == np.float64 and len(t) > 0
    assert (t >= MIN_FIRST_EVENT_S).all() and (np.diff(t) > 0).all()


def test_predict_count_scale(tmp_path):
    ld = _loaded(tmp_path)
    mel = np.random.default_rng(3).normal(size=(int(30 * FPS), 80)).astype(np.float16)
    n1 = len(predict_events(ld, mel, 4.0, (), "cpu"))
    n2 = len(predict_events(ld, mel, 4.0, (), "cpu", count_scale=0.5))
    assert 0 < n2 < n1
    # 4 nps * 30 s * 0.8 / 1.25 = 76.8 events before the lead-in cut
    assert abs(n1 - 77) <= 10


def test_timing_better_old_checkpoints():
    assert timing_better({"loss": 1.0, "f1": 0.1}, None)
    assert timing_better({"loss": 1.0, "f1": 0.1}, {"loss": 0.1})
    assert not timing_better({"loss": 0.01}, {"loss": 1.0, "f1": 0.0})
    assert timing_better({"f1": 0.5}, {"f1": 0.4})
    assert not timing_better({"f1": 0.4}, {"f1": 0.4})


class FakeCache:
    def events(self, i):
        return np.arange(2.0, 12.0, 0.5), None, None


def test_eval_fn_runs_on_cpu(tmp_path):
    for k in range(3):
        np.save(tmp_path / f"m{k}.npy", np.zeros((int(15 * FPS), 80), np.float16))
    entries = [Entry(k, f"m{k}", "Standard", "Expert", "val", 3.0, ("tech",), 18.0, 0.0,
                     int(15 * FPS)) for k in range(3)]
    base = lambda model, loader: {"loss": 0.5}  # noqa: E731
    fn = make_timing_eval(FakeCache(), entries, tmp_path, STATS, base, n_songs=2)
    out = fn(_model(), None)
    assert out["loss"] == 0.5
    assert all(0.0 <= out[k] <= 1.0 for k in ("precision", "recall", "f1"))
