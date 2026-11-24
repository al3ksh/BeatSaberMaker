import numpy as np



def test_pick_events_density_envelope_rests_and_loud_bars():
    from bsmaker.audio.mel import FRAMES_PER_SECOND as fps
    from bsmaker.models.timing import decode as d
    n = int(60 * fps)
    rng = np.random.default_rng(0)
    probs = np.zeros(n, dtype=np.float32)
    probs[::10] = 0.5 + 0.2 * rng.random(len(probs[::10]))
    mel = np.full((n, 8), -4.0, dtype=np.float32)
    for k in range(30):  # loudness changes every 2 s bar
        a = int((1.5 + 2.0 * k) * fps)
        mel[a:a + int(2 * fps)] = -4.0 + 3.0 * ((k * 7) % 10) / 10
    frames = d.pick_events(probs, 200, mel=mel, bar_s=2.0)
    edges = d.bar_edges(60.0, 2.0)
    counts = np.histogram(frames / fps, edges)[0]
    assert abs(len(frames) - 200) <= 60  # synthetic peaks are near-tied, counts jump in steps
    assert counts.std() / counts.mean() > 0.4
    assert 0 < (counts == 0).mean() <= 0.12


def test_bar_edges_are_on_the_map_bar_grid():
    from bsmaker.models.timing.decode import MIN_FIRST_EVENT_S, bar_edges
    bar = 4 * 60.0 / 140.0  # 4 beats at 140 bpm
    edges = bar_edges(60.0, bar)
    assert edges[0] == MIN_FIRST_EVENT_S and edges[-1] == 60.0
    inner = edges[1:-1]
    assert np.allclose(inner / bar, np.round(inner / bar))  # multiples of the bar from t=0
    assert np.allclose(np.diff(inner), bar)


def test_empty_bars_are_inside_the_event_span():
    from bsmaker.audio.mel import FRAMES_PER_SECOND as fps
    from bsmaker.models.timing import decode as d
    n = int(60 * fps)
    rng = np.random.default_rng(1)
    probs = np.zeros(n, dtype=np.float32)
    lo, hi = int(10 * fps), int(50 * fps)  # events only in 10-50 s; intro and outro are quiet
    probs[lo:hi:10] = 0.5 + 0.2 * rng.random(len(probs[lo:hi:10]))
    mel = np.full((n, 8), -6.0, dtype=np.float32)
    for k in range(30):
        a = int((1.5 + 2.0 * k) * fps)
        mel[a:a + int(2 * fps)] = -4.0 + 3.0 * ((k * 7) % 10) / 10
    frames = d.pick_events(probs, 150, mel=mel, bar_s=2.0) / fps
    edges = d.bar_edges(60.0, 2.0)
    counts = np.histogram(frames, edges)[0]
    inside = counts[np.searchsorted(edges, frames.min(), side="right") - 1:
                    np.searchsorted(edges, frames.max(), side="right")]
    assert (inside == 0).sum() >= 1


def _bar_mel(edges, levels, n_bins=8, flux=None):
    """Mel with one constant log-level per bar (`levels[i]`); `flux[i]` makes bar i zig-zag in
    bin groups with the same mean power as the constant level."""
    from bsmaker.audio.mel import FRAMES_PER_SECOND as fps
    n = int(edges[-1] * fps) + 1
    mel = np.full((n, n_bins), -12.0, dtype=np.float32)
    for i, lv in enumerate(levels):
        a, b = int(edges[i] * fps), int(edges[i + 1] * fps)
        mel[a:b] = lv
        if flux is not None and flux[i]:
            hi = np.log(2 * np.exp(lv) - np.exp(lv) * 0.02)  # keeps the bar power constant
            lo = np.log(np.exp(lv) * 0.02)
            for t in range(a, b):
                half = (t // 3) % 2
                mel[t, :n_bins // 2] = hi if half else lo
                mel[t, n_bins // 2:] = lo if half else hi
    return mel


def _bar_weights(mel, edges, w):
    from bsmaker.audio.mel import FRAMES_PER_SECOND as fps
    return np.array([w[min(int((edges[i] + edges[i + 1]) / 2 * fps), len(w) - 1)]
                     for i in range(len(edges) - 1)])


def test_density_weights_softens_section_boundary_jump():
    from bsmaker.models.timing import decode as d
    edges = d.bar_edges(66.0, 2.0)
    levels = [-4.0 if i < 16 else -4.0 + 1.386 for i in range(len(edges) - 1)]  # +6 dB at bar 16
    mel = _bar_mel(edges, levels)
    w, empty = d.density_weights(mel, edges)
    bw = _bar_weights(mel, edges, w)
    ratio = np.log2(bw[16:20].mean() / bw[12:16].mean())
    assert 0 < ratio < 1.6  # louder side denser, but not the old ~5x cliff


def test_density_weights_increases_intrasection_variation_from_onset_flux():
    from bsmaker.audio.mel import FRAMES_PER_SECOND as fps
    from bsmaker.models.timing import decode as d
    n = int(60 * fps)
    edges = d.bar_edges(60.0, 2.0)
    nb = len(edges) - 1
    # same mean power in every bar, but odd bars have strong frame-to-frame flux
    mel = _bar_mel(edges, [-3.0] * nb, flux=[i % 2 for i in range(nb)])
    mel = mel[:n]
    rng = np.random.default_rng(0)
    probs = np.zeros(n, dtype=np.float32)
    probs[::10] = 0.6 + 0.002 * rng.random(len(probs[::10]))
    frames = d.pick_events(probs, 200, mel=mel, bar_s=2.0)
    counts = np.histogram(frames / fps, edges)[0][1:-1]  # full-length bars only
    assert counts.std() / counts.mean() > 0.25
    odd, even = counts[0::2], counts[1::2]  # bar 1 is high-flux, bar 2 flat, ...
    assert odd.mean() > 1.4 * even.mean()


def test_density_weights_caps_consecutive_audible_empty_bars(monkeypatch):
    from bsmaker.models.timing import decode as d
    monkeypatch.setattr(d, "EMPTY_BAR_SHARE", 0.3)
    edges = d.bar_edges(66.0, 2.0)
    nb = len(edges) - 1
    levels = [-4.0] * nb
    for i in range(4, 10):
        levels[i] = -4.0 - 1.39  # about -6 dB, still audible
    for i in range(24, 28):
        levels[i] = -4.0 - 5.8  # about -25 dB, near silence
    mel = _bar_mel(edges, levels)
    w, empty = d.density_weights(mel, edges)
    run = best = 0
    for e in empty[:20]:
        run = run + 1 if e else 0
        best = max(best, run)
    assert best <= d.MAX_CONSECUTIVE_EMPTY
    assert empty[4:10].any()
    assert empty[24:28].all()  # true silence may empty consecutively
