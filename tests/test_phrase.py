import numpy as np

from bsmaker.audio.bar_similarity import (
    MAX_CONSECUTIVE_REPEAT_BARS,
    MAX_COPIES_PER_SOURCE,
    MAX_REPEAT_LOUD_DIFF_DB,
    MAX_RHYTHM_REPEAT_SHARE,
    MIN_BAR_SIM,
    bar_features,
    bar_repeats,
    features_from_frames,
    find_repeats,
)
from bsmaker.audio.decode import SAMPLE_RATE


def _synth_bar(bar_s: float, freqs: tuple[float, ...], seed: int, amp: float = 0.25) -> np.ndarray:
    n = int(round(bar_s * SAMPLE_RATE))
    t = np.arange(n, dtype=np.float32) / SAMPLE_RATE
    rng = np.random.default_rng(seed)
    sig = np.zeros(n, dtype=np.float32)
    for k, f in enumerate(freqs):
        env = np.exp(-3.0 * ((t * (k + 1)) % 0.5))
        sig += np.sin(2.0 * np.pi * f * t) * env
    sig += 0.02 * rng.standard_normal(n).astype(np.float32)
    peak = float(np.max(np.abs(sig))) or 1.0
    return (sig / peak * amp).astype(np.float32)


def _distinct_bars(n_bars: int, bar_s: float, seed_offset: int = 0) -> list[np.ndarray]:
    # Distinct pitch sets, section transposition, and rhythmic envelopes across bars/sections.
    base_freqs = [
        (110.0, 165.0),
        (146.8, 220.0, 293.7),
        (196.0, 246.9, 392.0, 587.3),
        (261.6, 329.6, 523.3, 659.3, 784.0),
        (311.1, 466.2, 622.3),
        (349.2, 523.3, 698.5, 1046.5),
        (415.3, 622.3, 830.6),
        (493.9, 740.0, 987.8, 1480.0),
    ]
    section = seed_offset // 100
    semitone_shift = (section * 5) % 12
    pitch_scale = 2.0 ** (semitone_shift / 12.0)
    rot = (section * 3) % len(base_freqs)
    out = []
    for k in range(n_bars):
        chord = base_freqs[(k + rot) % len(base_freqs)]
        freqs = tuple(f * pitch_scale * (1.0 + 0.07 * (k // 8)) for f in chord)
        bar = _synth_bar(bar_s, freqs, seed=seed_offset + k)
        if k % 2 == 1:
            bar = bar * np.linspace(0.4, 1.0, len(bar), dtype=np.float32)
        out.append(bar)
    return out


def test_bar_similarity_prefers_8_and_4_bar_phrases_over_2_bar():
    bpm = 120.0
    bar_s = 4.0 * 60.0 / bpm
    rng = np.random.default_rng(42)

    bars_a = _distinct_bars(8, bar_s, seed_offset=100)
    bars_repeat = [
        b + rng.normal(0.0, 0.005, size=len(b)).astype(np.float32)
        for b in bars_a
    ]
    bars_b = _distinct_bars(8, bar_s, seed_offset=200)
    audio = np.concatenate(bars_a + bars_repeat + bars_b)

    rep = bar_repeats(audio, bpm=bpm)
    assert rep.source.shape == (24,)
    assert list(rep.source[8:15]) == list(range(7))
    assert rep.source[15] == -1
    assert np.all(rep.source[16:] == -1)
    assert rep.phrase_len[8] == 8
    assert np.all(rep.phrase_start[8:15] == 8)

    # 4-bar phrase matching when only 4 bars repeat.
    bars_4_src = _distinct_bars(8, bar_s, seed_offset=300)
    bars_4_rep = [
        b + rng.normal(0.0, 0.005, size=len(b)).astype(np.float32)
        for b in bars_4_src[4:8]
    ]
    bars_4_tail = _distinct_bars(4, bar_s, seed_offset=400)
    rep4 = bar_repeats(np.concatenate(bars_4_src + bars_4_rep + bars_4_tail), bpm=bpm)
    assert list(rep4.source[8:12]) == [4, 5, 6, 7]
    assert np.all(rep4.source[12:] == -1)
    assert rep4.phrase_len[8] == 4


def test_bar_similarity_enforces_min_sim_and_loudness_guard():
    rng = np.random.default_rng(7)
    bar_s = 2.0

    # Random uncorrelated unit-half vectors (max off-diag similarity < MIN_BAR_SIM).
    c = rng.standard_normal((16, 12))
    m = rng.standard_normal((16, 12))
    feats = np.concatenate([
        c / np.linalg.norm(c, axis=1, keepdims=True),
        m / np.linalg.norm(m, axis=1, keepdims=True),
    ], axis=1)
    sim = feats @ feats.T / 2.0
    np.fill_diagonal(sim, -1.0)
    assert sim.max() < MIN_BAR_SIM
    rep_uncorr = find_repeats(feats, np.full(16, -10.0), bar_s)
    assert np.all(rep_uncorr.source == -1)

    # Identical chroma+MFCC across pairs 0..7 -> 8..15, but 6 dB quieter (> MAX_REPEAT_LOUD_DIFF_DB).
    assert 6.0 > MAX_REPEAT_LOUD_DIFF_DB
    feats_8 = feats[:8]
    feats_loud_diff = np.vstack([feats_8, feats_8])
    loud_diff = np.concatenate([np.full(8, -10.0), np.full(8, -16.0)])
    rep_loud = find_repeats(feats_loud_diff, loud_diff, bar_s)
    assert np.all(rep_loud.source == -1)

    # Quiet bars (-80 dB) never match, whether mixed with loud bars or alone.
    feats_same = np.tile(feats[0], (16, 1))
    rep_all_quiet = find_repeats(feats_same, np.full(16, -80.0), bar_s)
    assert np.all(rep_all_quiet.source == -1)

    mixed_loud = np.concatenate([np.full(8, -10.0), np.full(8, -80.0)])
    rep_mixed_quiet = find_repeats(feats_same, mixed_loud, bar_s)
    assert np.all(rep_mixed_quiet.source[8:] == -1)
    assert all(s < 8 for s in rep_mixed_quiet.source if s >= 0)


def test_bar_similarity_loop_song_anti_monotony_caps():
    bpm = 120.0
    bar_s = 4.0 * 60.0 / bpm
    one_bar = _synth_bar(bar_s, (220.0, 330.0, 440.0), seed=99)
    four_bar = np.concatenate(_distinct_bars(4, bar_s, seed_offset=900))

    for audio in (np.tile(one_bar, 48), np.tile(four_bar, 12)):
        rep = bar_repeats(audio, bpm=bpm)
        assert rep.source.shape == (48,)
        n_linked = int((rep.source >= 0).sum())
        assert n_linked > 0
        assert n_linked / 48 <= MAX_RHYTHM_REPEAT_SHARE

        # Max consecutive run of source >= 0 is <= MAX_CONSECUTIVE_REPEAT_BARS.
        run = max_run = 0
        for s in rep.source:
            run = run + 1 if s >= 0 else 0
            max_run = max(max_run, run)
        assert max_run <= MAX_CONSECUTIVE_REPEAT_BARS

        # No canonical source copied > MAX_COPIES_PER_SOURCE times; zero copy chains.
        counts = np.bincount(rep.source[rep.source >= 0])
        assert int(counts.max()) <= MAX_COPIES_PER_SOURCE
        for s in rep.source:
            if s >= 0:
                assert rep.source[s] == -1


def test_bar_similarity_respects_offset_s():
    bpm = 120.0
    bar_s = 4.0 * 60.0 / bpm
    rng = np.random.default_rng(123)

    bars_a = _distinct_bars(8, bar_s, seed_offset=500)
    bars_repeat = [
        b + rng.normal(0.0, 0.005, size=len(b)).astype(np.float32)
        for b in bars_a
    ]
    bars_b = _distinct_bars(8, bar_s, seed_offset=600)
    audio = np.concatenate(bars_a + bars_repeat + bars_b)

    offset_s = 0.25 * bar_s
    # Spectrally distinct broadband prefix (similar RMS to bars_a) so ignoring offset_s
    # misaligns bars and breaks bar 0 matching without causing a +20 dB STFT boundary bleed.
    n_prefix = int(round(offset_s * SAMPLE_RATE))
    prefix = (0.08 * rng.standard_normal(n_prefix)).astype(np.float32)
    shifted = np.concatenate([prefix, audio])

    rep_base = bar_repeats(audio, bpm=bpm, offset_s=0.0)
    rep_shifted = bar_repeats(shifted, bpm=bpm, offset_s=offset_s)
    rep_wrong = bar_repeats(shifted, bpm=bpm, offset_s=0.0)

    assert (rep_base.source >= 0).any()
    assert np.array_equal(rep_shifted.source, rep_base.source)
    assert np.array_equal(rep_shifted.phrase_len, rep_base.phrase_len)
    assert not np.array_equal(rep_wrong.source, rep_base.source)

    feats_base, loud_base = bar_features(audio, bpm=bpm, offset_s=0.0)
    feats_shifted, loud_shifted = bar_features(shifted, bpm=bpm, offset_s=offset_s)
    feats_wrong, _ = bar_features(shifted, bpm=bpm, offset_s=0.0)
    cos_aligned = (feats_base * feats_shifted).sum(axis=1) / 2.0
    cos_wrong = (feats_base * feats_wrong[: len(feats_base)]).sum(axis=1) / 2.0
    assert float(cos_aligned.min()) > 0.99
    assert np.allclose(loud_shifted, loud_base, atol=0.25)
    assert float(cos_wrong.min()) < 0.96
    assert float(cos_wrong.mean()) < float(cos_aligned.mean()) - 0.02


def test_features_from_frames_handles_non_finite_params():
    chroma = np.ones((12, 64), dtype=np.float32)
    mfcc = np.ones((13, 64), dtype=np.float32)
    rms = np.ones(64, dtype=np.float32)
    for hop_s, bar_s, offset_s in [
        (float("nan"), 2.0, 0.0),
        (float("inf"), 2.0, 0.0),
        (0.02, 2.0, float("nan")),
        (0.02, 2.0, float("inf")),
        (0.02, float("nan"), 0.0),
    ]:
        feats, loud = features_from_frames(chroma, mfcc, rms, hop_s, bar_s, offset_s=offset_s)
        assert feats.shape == (0, 24)
        assert loud.shape == (0,)
