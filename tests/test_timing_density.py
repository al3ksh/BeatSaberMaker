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
