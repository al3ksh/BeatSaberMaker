import numpy as np
import pytest

from bsmaker.audio.beats import Grid, align_grid, detect_grid, fit_grid, snap_times, to_beats
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


def _circ(d, p):
    return (d + p / 2) % p - p / 2


def test_snap_moves_within_limit_only():
    out = snap_times(np.array([0.1 + 0.5 + 0.004, 0.1 + 1.0 + 0.0156]), GRID, 0.025)
    assert out[0] == pytest.approx(0.6)
    assert out[1] == pytest.approx(1.1)
    far = snap_times(np.array([0.6 + 0.0156]), GRID, 0.001)
    assert far[0] == pytest.approx(0.6156)


def test_snap_pulls_jittered_events_back_to_quarter_beat_grid():
    rng = np.random.default_rng(3)
    g = Grid(150.0, 0.07, True)  # beat 0.4 s, 1/4 beat = 100 ms
    k = np.unique(rng.integers(5, 400, 150))
    t = g.offset_s + k * 0.1
    out = snap_times(t + rng.uniform(-0.015, 0.015, len(t)), g)
    assert np.max(np.abs(out - t)) < 0.001


def test_snap_keeps_triplets_off_the_quarter_grid():
    step3 = 0.5 / 3
    t = 0.1 + np.array([4, 5, 7, 8]) * step3 + np.array([0.004, -0.005, 0.006, -0.003])
    out = snap_times(t, GRID)
    np.testing.assert_allclose(out, 0.1 + np.array([4, 5, 7, 8]) * step3, atol=1e-9)


def test_snap_leaves_events_far_from_any_position_alone():
    # 1/4 beat = 125 ms: half a step (62.5 ms) from the 1/4 points, farther than 45 ms
    t = np.array([0.1 + 0.0625, 0.1 + 1.0 + 0.0625])
    np.testing.assert_array_equal(snap_times(t, GRID), t)


def test_snap_two_events_competing_for_one_position_stay_ordered():
    t = np.array([0.6 - 0.010, 0.6 + 0.012])
    out = snap_times(t, GRID)
    assert np.all(np.diff(out) > 0)
    assert out[0] == pytest.approx(0.6)
    assert out[1] == t[1]


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
        assert np.all(np.abs(out - t) <= 0.045 + 1e-9)
        assert np.all(np.abs(out - t) <= 0.45 * 60 / bpm / 3 + 1e-9)
        beat = 60 / bpm
        for o, orig in zip(out, t):
            if o == orig:
                continue
            ok = any(
                abs((o - g.offset_s) / (beat / d) - round((o - g.offset_s) / (beat / d))) * (beat / d) < 1e-6
                for d in (4, 3)
            )
            assert ok


def test_fit_grid_tolerates_missed_beat():
    t = np.arange(40) * 0.5
    g = fit_grid(np.delete(t, 20))
    assert g.steady
    assert abs(g.bpm - 120) < 0.5


def _click_track(bpm, first_s, dur_s, seed=0):
    n = int(dur_s * SAMPLE_RATE)
    y = np.zeros(n, dtype=np.float32)
    rng = np.random.default_rng(seed)
    m = int(0.03 * SAMPLE_RATE)
    burst = (rng.standard_normal(m) * np.exp(-np.arange(m) / (0.005 * SAMPLE_RATE))).astype(np.float32)
    t = first_s
    while t < dur_s - 0.05:
        i = int(round(t * SAMPLE_RATE))
        y[i:i + m] += burst[: n - i]
        t += 60.0 / bpm
    return y


def _circ_err(a, b, period):
    return abs((a - b + period / 2) % period - period / 2)


@pytest.mark.parametrize("bpm", [128.0, 174.0, 95.5])
def test_detect_grid_click_track_steady(bpm):
    g = detect_grid(_click_track(bpm, 0.250, 60))
    assert g.steady
    ratios = [1.0, 0.5, 2.0]
    assert any(abs(g.bpm / (bpm * r) - 1) < 0.003 for r in ratios)
    if abs(g.bpm / bpm - 1) < 0.003:
        half = 60.0 / g.bpm / 2
        # the 30 ms envelope-lag correction is a median over real songs; a synthetic
        # click is ~18-20 ms off with it (consistent), hence 25 ms instead of 15
        assert _circ_err(g.offset_s, 0.250, half) < 0.025


def test_detect_grid_white_noise_not_steady():
    y = np.random.default_rng(1).standard_normal(60 * SAMPLE_RATE).astype(np.float32)
    assert not detect_grid(y).steady


def test_detect_grid_tempo_change_not_steady():
    a = _click_track(120, 0.25, 30)
    b = _click_track(150, 0.25, 30)
    assert not detect_grid(np.concatenate([a, b])).steady


def test_detect_grid_short_and_silent_not_steady():
    g = detect_grid(_click_track(120, 0.25, 10))
    assert not g.steady
    s = detect_grid(np.zeros(60 * SAMPLE_RATE, dtype=np.float32))
    assert not s.steady
    assert np.isfinite(s.bpm) and np.isfinite(s.offset_s)


def _jittered_quarter_events(g, n=200, seed=5, jitter=0.015):
    rng = np.random.default_rng(seed)
    k = np.unique(rng.integers(8, 1500, n))
    return g.offset_s + k * 60 / g.bpm / 4 + rng.uniform(-jitter, jitter, len(k)), k


def test_align_grid_recovers_offset_from_wrong_input_grid():
    true = Grid(128.0, 0.2, True)
    t, _ = _jittered_quarter_events(true)
    wrong = Grid(128.0, 0.2 + 0.060, True)
    out = align_grid(wrong, t)
    assert out.bpm == 128.0 and out.steady
    assert abs(_circ(out.offset_s - true.offset_s, 60 / 128.0 / 4)) < 0.003


def test_align_grid_picks_beat_origin_from_triplets():
    true = Grid(120.0, 0.13, True)  # beat 0.5 s
    rng = np.random.default_rng(9)
    m = np.arange(8, 120)[:, None]
    t = np.sort((true.offset_s + m * 0.5 + np.array([0.0, 0.5 / 3, 1.0 / 3, 0.25])).ravel())
    t = t[rng.random(len(t)) < 0.6]
    t = t + rng.uniform(-0.01, 0.01, len(t))
    t = np.sort(t)
    out = align_grid(Grid(120.0, 0.13 + 0.125, True), t)
    assert abs(_circ(out.offset_s - true.offset_s, 0.5)) < 0.004


def test_align_grid_leaves_unsteady_grid_and_empty_input_alone():
    g = Grid(120.0, 0.0, False)
    assert align_grid(g, np.array([1.0, 1.2, 1.4] * 10)) == g
    s = Grid(120.0, 0.1, True)
    assert align_grid(s, np.array([])) == s


def _tone_in_silence(lead_s=5.0, tone_s=20.0, tail_s=5.0):
    t = np.arange(int(tone_s * SAMPLE_RATE)) / SAMPLE_RATE
    tone = 0.5 * np.sin(2 * np.pi * 220 * t)
    return np.concatenate([
        np.zeros(int(lead_s * SAMPLE_RATE)), tone, np.zeros(int(tail_s * SAMPLE_RATE)),
    ]).astype(np.float32)


def test_audible_span_finds_tone_inside_silence():
    from bsmaker.audio.beats import audible_span

    start, end = audible_span(_tone_in_silence())
    assert 4.8 <= start <= 5.3
    assert 24.7 <= end <= 25.2


def test_gate_audible_drops_notes_in_silence_and_on_last_frame():
    from bsmaker.audio.beats import audible_span, gate_audible

    audio = _tone_in_silence()
    span = audible_span(audio)
    times = np.array([1.0, 2.2, 5.5, 12.0, 24.0, 26.0, 29.5, 30.0])
    kept = gate_audible(times, span, len(audio) / SAMPLE_RATE)
    assert kept.tolist() == [5.5, 12.0, 24.0]
    # a uniformly loud file keeps everything except the very last frame
    loud = np.full(30 * SAMPLE_RATE, 0.3, dtype=np.float32)
    span = audible_span(loud)
    kept = gate_audible(np.array([0.5, 10.0, 29.9, 30.0]), span, 30.0)
    assert kept.tolist() == [0.5, 10.0]
