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


def _two_level_curve(spacing=9, per_half=300):
    """Loud half (peaks 0.9) then quiet but active half (peaks 0.3), floor 0.01."""
    n = 2 * per_half * spacing + 400
    p = np.full(n, 0.01, dtype=np.float32)
    jit = np.random.default_rng(5).uniform(0.6, 1.4, 2 * per_half).astype(np.float32)
    p[np.arange(per_half) * spacing + 200] = np.minimum(0.9 * jit[:per_half], 0.99)
    p[(per_half + np.arange(per_half)) * spacing + 200] = 0.3 * jit[per_half:]
    return p, per_half * spacing + 196


def test_pick_events_covers_quiet_active_half():
    from bsmaker.models.timing.decode import pick_events
    p, mid = _two_level_curve()
    want = 200
    old = pick_peaks(p, threshold_for_count(p, want))
    assert (old >= mid).sum() == 0  # the old global threshold leaves the quiet half empty
    got = pick_events(p, want)
    assert (got < mid).sum() > 30 and (got >= mid).sum() >= 3  # gamma < 1: a mild lift
    assert abs(len(got) - want) <= max(2, 0.03 * want)


def test_pick_events_silent_stretch_stays_empty():
    from bsmaker.models.timing.decode import pick_events
    rng = np.random.default_rng(0)
    n = 6000
    p = np.full(n, 0.01, dtype=np.float32) + rng.random(n).astype(np.float32) * 0.002
    p[np.arange(150) * 9 + 100] = 0.9           # active first 1400 frames
    p[np.arange(150) * 9 + 3000] = 0.3          # quiet but active middle
    got = pick_events(p, 250)
    assert ((got > 1600) & (got < 2900)).sum() == 0
    assert (got >= 3000).sum() > 50
    assert ((got > 4400)).sum() == 0            # silent tail


def test_pick_events_invariants():
    from bsmaker.models.timing.decode import pick_events
    p, _ = _two_level_curve()
    p[:int(round(MIN_FIRST_EVENT_S * FPS)) + 50] = 0.8  # noise inside the lead-in
    got = pick_events(p, 300)
    assert got.tolist() == sorted(set(got.tolist()))
    assert (np.diff(got) >= 3).all()
    assert abs(len(got) - 300) <= 9
    assert (got >= int(round(MIN_FIRST_EVENT_S * FPS))).all()


def test_pick_events_count_scale_monotonic():
    from bsmaker.models.timing.decode import pick_events
    p, _ = _two_level_curve()
    assert len(pick_events(p, 360)) > len(pick_events(p, 300)) > len(pick_events(p, 240))


def test_pick_events_degenerate_inputs():
    from bsmaker.models.timing.decode import pick_events
    assert len(pick_events(np.zeros(5000, dtype=np.float32), 100)) == 0
    assert len(pick_events(np.zeros(0, dtype=np.float32), 10)) == 0
    short = np.array([0, .5, 0], dtype=np.float32)
    assert len(pick_events(short, 1)) == 0  # entirely inside the lead-in
    assert len(pick_events(short, 0)) == 0



def _at_seconds(times_s, total_s=120.0):
    p = np.full(int(total_s * FPS), 0.01, dtype=np.float32)
    for t in times_s:
        p[int(round(t * FPS))] = 0.9
    return p


def _stray_times(first_group, gap_after, rest_from=None, rest_n=60, step=0.5):
    last = first_group[-1]
    start = rest_from if rest_from is not None else last + gap_after
    return list(first_group) + [start + k * step for k in range(rest_n)]


def test_pick_events_drops_stray_lead_in_before_long_gap():
    from bsmaker.models.timing.decode import pick_events
    times = _stray_times([2.3, 2.8, 3.4, 4.0], 8.3)
    got = pick_events(_at_seconds(times), len(times)) / FPS
    assert got[0] > 12.0 and len(got) == 60


def test_pick_events_keeps_normal_start():
    from bsmaker.models.timing.decode import pick_events
    times = [2.0 + 0.5 * k for k in range(60)]
    got = pick_events(_at_seconds(times), len(times)) / FPS
    assert len(got) == 60 and abs(got[0] - 2.0) < 0.1


def test_pick_events_keeps_group_larger_than_k():
    from bsmaker.models.timing.decode import STRAY_MAX_EVENTS, pick_events
    group = [2.0 + 0.5 * k for k in range(STRAY_MAX_EVENTS + 1)]
    times = _stray_times(group, 9.0)
    got = pick_events(_at_seconds(times), len(times)) / FPS
    assert len(got) == len(times) and abs(got[0] - 2.0) < 0.1


def test_pick_events_long_gap_after_30s_does_not_trigger():
    from bsmaker.models.timing.decode import pick_events
    times = [2.0 + 0.5 * k for k in range(70)]            # dense until 36.5 s
    times += [36.5 + 9.0 + 0.5 * k for k in range(30)]
    got = pick_events(_at_seconds(times, 150.0), len(times)) / FPS
    assert len(got) == len(times) and abs(got[0] - 2.0) < 0.1
    # a small group that itself ends after 30 s is not a lead-in either
    t2 = [29.0, 30.5, 31.0] + [31.0 + 9.0 + 0.5 * k for k in range(60)]
    got2 = pick_events(_at_seconds(t2, 150.0), len(t2)) / FPS
    assert abs(got2[0] - 29.0) < 0.1


def test_trim_stray_lead_in_edge_cases():
    from bsmaker.models.timing.decode import trim_stray_lead_in
    for n in (0, 1, 2):
        assert len(trim_stray_lead_in(np.arange(n) * 10)) == n  # no gap after the group: kept
    assert len(trim_stray_lead_in(np.zeros(0, dtype=np.int64))) == 0
