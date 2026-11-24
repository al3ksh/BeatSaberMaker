import numpy as np
import pytest
import torch

from bsmaker.audio.bar_similarity import (
    MAX_CONSECUTIVE_REPEAT_BARS,
    MAX_COPIES_PER_SOURCE,
    MAX_REPEAT_LOUD_DIFF_DB,
    MAX_RHYTHM_REPEAT_SHARE,
    MIN_BAR_SIM,
    BarRepeats,
    bar_features,
    bar_repeats,
    features_from_frames,
    find_repeats,
)
from bsmaker.audio.decode import SAMPLE_RATE
from bsmaker.audio.mel import FRAMES_PER_SECOND
from bsmaker.formats.events import to_events
from bsmaker.formats.tokens import NONE_TOKEN, VOCAB_SIZE
from bsmaker.models.pattern.model import PatternConfig, PatternModel
from bsmaker.models.pattern.sample import LoadedPattern, sample_notes
from bsmaker.models.timing.decode import TimingConfig, TimingCurve, LoadedTiming, pick_events, predict_events
from bsmaker.models.timing.model import TimingModel
from bsmaker.phrase import (
    HAND_LINK_MAX_DOUBLE_RUN,
    HAND_PRESENCE_BONUS,
    MIN_GAP_S,
    RESNAP_FRACTION,
    HandRhythmLinks,
    plan_rhythm,
    repeat_rhythm,
)
from bsmaker.playability.checker import check
from bsmaker.playability.config import PlayabilityConfig


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


def _make_repeats(n_bars: int, bar_s: float, links: dict[int, int]) -> BarRepeats:
    source = np.full(n_bars, -1, dtype=np.int64)
    sim = np.full((n_bars, n_bars), -1.0, dtype=np.float32)
    phrase_len = np.zeros(n_bars, dtype=np.int64)
    phrase_start = np.full(n_bars, -1, dtype=np.int64)
    for j, i in links.items():
        source[j] = i
        sim[j, i] = 0.95
        phrase_len[j] = 2
        phrase_start[j] = j
    return BarRepeats(
        bar_s=bar_s,
        source=source,
        sim=sim,
        phrase_len=phrase_len,
        phrase_start=phrase_start,
    )


def _set_support(curve: np.ndarray, times_s: list[float], val: float) -> None:
    for t in times_s:
        curve[int(round(t * FRAMES_PER_SECOND))] = val


def test_repeat_rhythm_all_or_nothing_no_refill():
    bar_s = 2.0
    n_bars = 9
    thr = 0.5
    curve = np.zeros(int(round((n_bars * bar_s + 1.0) * FRAMES_PER_SECOND)), dtype=np.float32)

    # Bar 0: 4-event source; Bar 1: early-entry source; Bar 7: 6-event source.
    bar0 = [0.25, 0.75, 1.25, 1.75]
    bar1 = [2.01, 2.50, 3.00, 3.50]
    bar2 = [4.10, 4.90, 5.50]                 # supported copy of bar 0 -> accepts all 4 offsets
    bar3 = [6.10, 6.60, 7.10, 7.60]           # 2 unsupported offsets -> stays unchanged
    bar4 = [8.50, 9.50]                       # rest bar (< 3 events) -> stays untouched
    bar5 = [10.20, 10.70, 11.20, 11.99]       # ends right before bar 6 boundary
    bar6 = [12.30, 12.80, 13.30, 13.80]       # copy of bar 1 violates MIN_GAP_S -> rejected
    bar7 = [14.15, 14.45, 14.75, 15.05, 15.35, 15.65]  # 6-event source
    bar8 = [16.20, 16.50, 16.80, 17.10, 17.40, 17.70]  # 5 strong + 1 weak (>= 0.25*thr) -> accepted

    planned = np.array(bar0 + bar1 + bar2 + bar3 + bar4 + bar5 + bar6 + bar7 + bar8, dtype=np.float64)
    repeats = _make_repeats(n_bars, bar_s, {2: 0, 3: 0, 4: 0, 6: 1, 8: 7})

    # Bar 2: all 4 shifted offsets well supported (> RESNAP_FRACTION * thr)
    _set_support(curve, [t + 4.0 for t in bar0], 0.8)
    # Bar 3: only 2 supported, 2 unsupported (0.0)
    _set_support(curve, [bar0[0] + 6.0, bar0[1] + 6.0], 0.8)
    # Bar 4: fully supported, but rest bar (< 3 events) must stay untouched
    _set_support(curve, [t + 8.0 for t in bar0], 0.8)
    # Bar 6: fully supported, but 12.01 - 11.99 = 0.02 < MIN_GAP_S
    assert (bar1[0] + 10.0) - bar5[-1] < MIN_GAP_S
    _set_support(curve, [t + 10.0 for t in bar1], 0.8)
    # Bar 8: 5 strong (>= 0.45*thr) + 1 weak in [0.25*thr, 0.45*thr)
    _set_support(curve, [t + 2.0 for t in bar7[:5]], 0.8)
    _set_support(curve, [bar7[5] + 2.0], 0.30 * thr)
    assert 0.25 * thr < 0.30 * thr < RESNAP_FRACTION * thr

    out = repeat_rhythm(planned, TimingCurve(curve, thr), repeats)

    def sl(k: int) -> np.ndarray:
        return out[(out >= k * bar_s) & (out < (k + 1) * bar_s)]

    np.testing.assert_allclose(sl(2), np.array(bar0) + 4.0, atol=1e-12)
    np.testing.assert_allclose(sl(3), np.array(bar3), atol=1e-12)
    np.testing.assert_allclose(sl(4), np.array(bar4), atol=1e-12)
    np.testing.assert_allclose(sl(6), np.array(bar6), atol=1e-12)
    np.testing.assert_allclose(sl(8), np.array(bar7) + 2.0, atol=1e-12)

    # With 2 weak offsets in a 6-event bar, even if both >= 0.25*thr, reject cleanly.
    curve2 = curve.copy()
    _set_support(curve2, [bar7[4] + 2.0], 0.30 * thr)
    out2 = repeat_rhythm(planned, TimingCurve(curve2, thr), repeats)
    np.testing.assert_allclose(out2[(out2 >= 8 * bar_s) & (out2 < 9 * bar_s)], np.array(bar8), atol=1e-12)


def test_hand_rhythm_links_bias_none_token_only():
    t_s1, t_t1 = 2.25, 10.25
    t_s2, t_t2 = 2.75, 10.75
    t_unlinked = 6.25

    links = HandRhythmLinks([(t_t1, t_s1), (t_t2, t_s2)])
    links.record(t_s1, has_left=True, has_right=False)
    links.record(t_s2, has_left=False, has_right=True)

    mask_all = np.ones(VOCAB_SIZE, dtype=bool)
    lg = torch.linspace(-2.0, 2.0, VOCAB_SIZE, dtype=torch.float32)

    # Target 1: source had left only -> left NONE down, right NONE up (when left played).
    out_l1 = links.bias_left(lg, t_t1, mask_all)
    assert torch.equal(out_l1[1:], lg[1:])
    assert out_l1[NONE_TOKEN].item() == pytest.approx(lg[NONE_TOKEN].item() - HAND_PRESENCE_BONUS)

    out_r1 = links.bias_right(lg, t_t1, mask_all, left=5, run=0)
    assert torch.equal(out_r1[1:], lg[1:])
    assert out_r1[NONE_TOKEN].item() == pytest.approx(lg[NONE_TOKEN].item() + HAND_PRESENCE_BONUS)

    # Never encourage right NONE when left is already NONE.
    out_r1_none_left = links.bias_right(lg, t_t1, mask_all, left=NONE_TOKEN, run=0)
    assert torch.equal(out_r1_none_left, lg)

    # Target 2: source had right only -> left NONE up, right NONE down.
    out_l2 = links.bias_left(lg, t_t2, mask_all)
    assert torch.equal(out_l2[1:], lg[1:])
    assert out_l2[NONE_TOKEN].item() == pytest.approx(lg[NONE_TOKEN].item() + HAND_PRESENCE_BONUS)

    out_r2 = links.bias_right(lg, t_t2, mask_all, left=5, run=0)
    assert torch.equal(out_r2[1:], lg[1:])
    assert out_r2[NONE_TOKEN].item() == pytest.approx(lg[NONE_TOKEN].item() - HAND_PRESENCE_BONUS)

    # Double-run cap stops encouraging doubles when run >= HAND_LINK_MAX_DOUBLE_RUN.
    out_r2_cap = links.bias_right(lg, t_t2, mask_all, left=5, run=HAND_LINK_MAX_DOUBLE_RUN)
    assert torch.equal(out_r2_cap, lg)

    # When left is NONE, encouraging right is a single note (not a double), so cap does not block it.
    out_r2_single = links.bias_right(lg, t_t2, mask_all, left=NONE_TOKEN, run=HAND_LINK_MAX_DOUBLE_RUN)
    assert torch.equal(out_r2_single[1:], lg[1:])
    assert out_r2_single[NONE_TOKEN].item() == pytest.approx(lg[NONE_TOKEN].item() - HAND_PRESENCE_BONUS)

    # Unlinked event is untouched.
    assert torch.equal(links.bias_left(lg, t_unlinked, mask_all), lg)
    assert torch.equal(links.bias_right(lg, t_unlinked, mask_all, left=5, run=0), lg)


def _hand_repeat_share(notes, bar_s: float) -> float:
    bars: dict[int, list] = {}
    for n in notes:
        b = int(n.time_s // bar_s)
        bars.setdefault(b, []).append(n)
    seen = set()
    rep = 0
    total = 0
    for b in sorted(bars):
        bn = bars[b]
        if len(bn) < 3:
            continue
        total += 1
        sig = tuple(sorted({(round(((n.time_s % bar_s) / bar_s) * 48) / 48, n.color) for n in bn}))
        if sig in seen:
            rep += 1
        seen.add(sig)
    return rep / max(total, 1)


def test_hand_rhythm_links_in_sampler_increases_hand_repeat_and_stays_playable():
    torch.manual_seed(0)
    pcfg = PatternConfig(d_model=32, n_layers=2, n_heads=2, context=16, dropout=0.0)
    model = PatternModel(pcfg).eval()
    # Real checkpoints have logit(NONE) ~= log(108) ~= 4.5 so single-hand notes and doubles compete.
    with torch.no_grad():
        model.left_head.bias[NONE_TOKEN] = 4.5
        model.right_head[-1].bias[NONE_TOKEN] = 4.5
    stats = {"mel_mean": 0.0, "mel_std": 1.0}
    loaded = LoadedPattern(model, pcfg, stats)
    cfg = PlayabilityConfig()

    bar_s = 2.0
    phrase = [
        np.array([0.20, 0.70, 1.20, 1.70], dtype=np.float64),
        np.array([0.25, 0.55, 1.05, 1.55], dtype=np.float64),
        np.array([0.15, 0.65, 1.15, 1.65], dtype=np.float64),
        np.array([0.30, 0.80, 1.30, 1.80], dtype=np.float64),
    ]
    n_bars = 17  # bars 1..16 active (4 repetitions of the 4-bar phrase)
    planned = np.concatenate([k * bar_s + phrase[(k - 1) % 4] for k in range(1, n_bars)])
    mel = np.zeros((int((n_bars * bar_s + 1.0) * FRAMES_PER_SECOND), 80), dtype=np.float16)

    curve = np.ones(len(mel), dtype=np.float32)
    links_map = {5 + k: 1 + k for k in range(4)}
    links_map.update({9 + k: 1 + k for k in range(4)})
    repeats = _make_repeats(n_bars, bar_s, links_map)
    planned2, hand_rhythm = plan_rhythm(planned, TimingCurve(curve, 0.5), repeats)
    assert hand_rhythm is not None and len(hand_rhythm) > 0
    np.testing.assert_allclose(planned2, planned, atol=1e-12)

    base_shares = []
    linked_shares = []
    for seed in (1, 2, 3):
        notes_base, _ = sample_notes(loaded, planned, mel, 2.0, ("balanced",), seed, cfg, "cpu")
        notes_link, _ = sample_notes(
            loaded, planned2, mel, 2.0, ("balanced",), seed, cfg, "cpu",
            hand_rhythm=hand_rhythm,
        )
        events, dropped = to_events(notes_link)
        assert dropped == 0
        assert check(events, cfg).hard == ()
        base_shares.append(_hand_repeat_share(notes_base, bar_s))
        linked_shares.append(_hand_repeat_share(notes_link, bar_s))

    assert float(np.mean(linked_shares)) > float(np.mean(base_shares))


def test_predict_events_emits_timing_curve():
    torch.manual_seed(0)
    tcfg = TimingConfig(d_model=32, n_layers=2, n_heads=4, window=64, dropout=0.0)
    stats = {
        "mel_mean": 0.0,
        "mel_std": 1.0,
        "coverage_median": 0.8,
        "nps_bins": [0.0, 100.0],
        "double_share_by_bin": [0.25, 0.25],
    }
    loaded = LoadedTiming(TimingModel(tcfg).eval(), tcfg, stats)
    mel = np.random.default_rng(0).normal(size=(int(20 * FRAMES_PER_SECOND), 80)).astype(np.float16)

    curve_out: list[TimingCurve] = []
    t_with = predict_events(loaded, mel, 3.0, ("balanced",), "cpu", bar_s=2.0, curve_out=curve_out)
    t_without = predict_events(loaded, mel, 3.0, ("balanced",), "cpu", bar_s=2.0)

    np.testing.assert_array_equal(t_with, t_without)
    assert len(curve_out) == 1
    assert isinstance(curve_out[0], TimingCurve)
    assert curve_out[0].curve.shape == (len(mel),)
    assert np.isfinite(curve_out[0].threshold) and curve_out[0].threshold > 0.0

