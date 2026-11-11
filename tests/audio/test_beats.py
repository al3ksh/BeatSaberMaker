import numpy as np
import pytest

from bsmaker.audio.beats import Grid, detect_grid, fit_grid, snap_times, to_beats
from bsmaker.audio.decode import SAMPLE_RATE


def test_fit_grid_recovers_bpm_and_offset():
    rng = np.random.default_rng(0)
    beat = 60 / 128
    t = 0.37 + np.arange(60) * beat + rng.uniform(-0.005, 0.005, 60)
    g = fit_grid(t)
    assert g.steady
    assert abs(g.bpm - 128) < 0.2
    assert abs(g.offset_s - 0.37) < 0.01


def test_fit_grid_normalises_offset():
    t = 2.0 + np.arange(40) * 0.5
    g = fit_grid(t)
    assert 0 <= g.offset_s < 0.5
    assert abs(g.offset_s) < 1e-6 or abs(g.offset_s - 0.5) < 1e-6


def test_fit_grid_unsteady_on_tempo_drift():
    t = np.cumsum(0.5 + 0.005 * np.arange(40))
    g = fit_grid(t)
    assert not g.steady
    assert g.offset_s == 0.0
    assert g.bpm > 0


def test_fit_grid_too_few_beats_not_steady():
    g = fit_grid(np.arange(15) * 0.5)
    assert not g.steady
    assert abs(g.bpm - 120) < 1e-6


@pytest.mark.parametrize("n", [0, 1])
def test_fit_grid_degenerate(n):
    assert fit_grid(np.arange(n) * 0.5) == Grid(120.0, 0.0, False)


def _clicks(bpm, seconds, offset=0.3):
    rng = np.random.default_rng(1)
    a = np.zeros(int(seconds * SAMPLE_RATE), dtype=np.float32)
    n = 2000
    burst = (rng.standard_normal(n) * np.exp(-np.arange(n) / 300)).astype(np.float32)
    t = offset
    while t < seconds - 0.2:
        i = int(t * SAMPLE_RATE)
        a[i:i + n] += burst
        t += 60 / bpm
    return a


def test_detect_grid_on_click_track():
    g = detect_grid(_clicks(120, 30))
    assert g.steady
    assert any(abs(g.bpm - b) < 1 for b in (120, 60, 240))
    assert np.isfinite(g.offset_s)


def test_detect_grid_on_silence():
    g = detect_grid(np.zeros(5 * SAMPLE_RATE, dtype=np.float32))
    assert g == Grid(120.0, 0.0, False)


GRID = Grid(120.0, 0.1, True)  # beat 0.5 s, 1/16 = 31.25 ms, 1/12 = 41.67 ms


def test_snap_moves_within_limit_only():
    out = snap_times(np.array([0.1 + 0.5 + 0.004, 0.1 + 1.0 + 0.0156]), GRID, 0.025)
    assert out[0] == pytest.approx(0.6)
    assert out[1] == pytest.approx(1.1)
    far = snap_times(np.array([0.6 + 0.0156]), GRID, 0.001)
    assert far[0] == pytest.approx(0.6156)


def test_snap_prefers_nearest_of_sixteenth_and_triplet():
    step12 = 0.5 / 12
    t = 0.1 + 5 * step12 + 0.002  # on a triplet position, 1/16 grid is further away
    assert snap_times(np.array([t]), GRID)[0] == pytest.approx(0.1 + 5 * step12)
    t2 = 0.1 + 3 * 0.5 / 16 + 0.002
    assert snap_times(np.array([t2]), GRID)[0] == pytest.approx(0.1 + 3 * 0.5 / 16)


def test_snap_keeps_times_strictly_increasing():
    g = Grid(240.0, 0.0, True)
    t = np.array([1.0, 1.012, 1.024, 1.036])
    out = snap_times(t, g)
    assert np.all(np.diff(out) > 0)
    assert np.all(np.isfinite(out))


def test_snap_rejects_non_increasing_input():
    with pytest.raises(ValueError):
        snap_times(np.array([1.0, 1.0]), GRID)


def test_snap_empty():
    assert snap_times(np.array([]), GRID).shape == (0,)


def test_unsteady_grid_leaves_times_alone():
    t = np.array([0.101, 0.5234, 0.9])
    np.testing.assert_array_equal(snap_times(t, Grid(120.0, 0.1, False)), t)


def test_to_beats():
    np.testing.assert_allclose(to_beats(np.array([0.0, 0.5, 1.0]), 120.0), [0.0, 1.0, 2.0])
    assert to_beats(np.array([]), 120.0).shape == (0,)


def test_snap_regression_forward_snap_past_next_event():
    out = snap_times(np.array([10.55809, 10.57170]), Grid(65.249, 0.1748, True))
    assert np.all(np.diff(out) > 0)


def test_snap_random_property():
    rng = np.random.default_rng(7)
    for _ in range(500):
        bpm = rng.uniform(60, 240)
        g = Grid(bpm, rng.uniform(0, 60 / bpm), True)
        t = rng.uniform(0, 5) + np.cumsum(rng.uniform(0.012, 2.0, rng.integers(2, 41)))
        out = snap_times(t, g)
        assert np.all(np.diff(out) > 0)
        assert np.all(np.abs(out - t) <= 0.025 + 1e-9)
        beat = 60 / bpm
        for o, orig in zip(out, t):
            if o == orig:
                continue
            ok = any(
                abs((o - g.offset_s) / (beat / d) - round((o - g.offset_s) / (beat / d))) * (beat / d) < 1e-6
                for d in (16, 12)
            )
            assert ok


def test_fit_grid_tolerates_missed_beat():
    t = np.arange(40) * 0.5
    g = fit_grid(np.delete(t, 20))
    assert g.steady
    assert abs(g.bpm - 120) < 0.5
