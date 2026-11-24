import random
import math

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
from bsmaker.flow_rules import _added_dot_tails, apply_flow, final_flow_ok
from bsmaker.formats.events import to_events
from bsmaker.formats.model import Note
from bsmaker.formats.tokens import NONE_TOKEN, VOCAB_SIZE, decode_token, encode_token
from bsmaker.models.pattern.model import PatternConfig, PatternModel
from bsmaker.models.pattern.sample import LoadedPattern, sample_notes
from bsmaker.models.timing.decode import TimingConfig, TimingCurve, LoadedTiming, pick_events, predict_events
from bsmaker.models.timing.model import TimingModel
from bsmaker.phrase import (
    HAND_LINK_MAX_DOUBLE_RUN,
    HAND_PRESENCE_BONUS,
    MAX_CONSECUTIVE_REPLAY_BARS,
    MIN_GAP_S,
    PATTERN_REPLAY_MAX_SHARE,
    RESNAP_FRACTION,
    HandRhythmLinks,
    mirror_note,
    mirror_token,
    plan_rhythm,
    repeat_rhythm,
    replay_patterns,
)
from bsmaker.playability.checker import check, check_exact
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


def _note(t: float, x: int, y: int, color: int, direction: int, bpm: float = 120.0) -> Note:
    return Note(time_s=t, beat=t * bpm / 60.0, x=x, y=y, color=color, direction=direction)


def _bar_notes_map(notes: list[Note], bar_s: float) -> dict[int, list[Note]]:
    out: dict[int, list[Note]] = {}
    for n in notes:
        out.setdefault(int(math.floor(n.time_s / bar_s)), []).append(n)
    return out


def test_mirror_token_and_note():
    assert mirror_token(NONE_TOKEN) == NONE_TOKEN
    # (x=0, y=1, dir=4[UL]) <-> (x=3, y=1, dir=5[UR])
    t_ul = encode_token(0, 1, 4)
    t_ur = mirror_token(t_ul)
    assert decode_token(t_ur) == (3, 1, 5)
    assert mirror_token(t_ur) == t_ul
    # (x=1, y=0, dir=2[L]) <-> (x=2, y=0, dir=3[R])
    assert decode_token(mirror_token(encode_token(1, 0, 2))) == (2, 0, 3)
    # Up, down, dot directions preserved; x flipped
    assert decode_token(mirror_token(encode_token(1, 2, 0))) == (2, 2, 0)
    assert decode_token(mirror_token(encode_token(0, 0, 8))) == (3, 0, 8)

    n = _note(1.25, 1, 0, 0, 7)  # left DR
    m = mirror_note(n)
    assert m == Note(time_s=1.25, beat=n.beat, x=2, y=0, color=1, direction=6)


def test_replay_patterns_exact_and_mirror_respect_budget_and_window_ok():
    bpm = 120.0
    bar_s = 2.0
    n_bars = 40
    cfg = PlayabilityConfig()

    # Bar 0 (source): asymmetric L/R with diagonal L and vertical R, valid entry/exit parity for both exact & mirror.
    # Bars 1..39: vertical L/R so exact and mirrored replays of bar 0 are both distinct and distinguishable.
    notes: list[Note] = []
    for k in range(n_bars):
        t0 = k * bar_s
        if k == 0:
            notes.extend([
                _note(t0 + 0.25, 0, 1, 0, 4, bpm),  # L UL
                _note(t0 + 0.75, 2, 2, 1, 0, bpm),  # R U
                _note(t0 + 1.25, 1, 0, 0, 7, bpm),  # L DR
                _note(t0 + 1.75, 2, 0, 1, 1, bpm),  # R D
            ])
        else:
            notes.extend([
                _note(t0 + 0.25, 1, 2, 0, 0, bpm),  # L U
                _note(t0 + 0.75, 2, 2, 1, 0, bpm),  # R U
                _note(t0 + 1.25, 1, 0, 0, 1, bpm),  # L D
                _note(t0 + 1.75, 2, 0, 1, 1, bpm),  # R D
            ])

    links = {j: 0 for j in range(8, n_bars)}
    repeats = _make_repeats(n_bars, bar_s, links)
    for j in range(8, n_bars):
        repeats.sim[j, 0] = 0.90 + 0.001 * (j % 5)

    saw_exact = False
    saw_mirror = False
    src_sig_exact = [(n.x, n.y, n.color, n.direction) for n in notes[:4]]
    src_sig_mirror = [(mirror_note(n).x, mirror_note(n).y, mirror_note(n).color, mirror_note(n).direction)
                      for n in notes[:4]]

    for seed in range(20):
        out, replayed = replay_patterns(notes, repeats=repeats, bpm=bpm, mask_cfg=cfg, rng=random.Random(seed))
        assert 0 < replayed <= PATTERN_REPLAY_MAX_SHARE * n_bars + 1
        assert final_flow_ok(notes, out, cfg)
        assert check_exact(out, cfg).hard == ()

        by_bar = _bar_notes_map(out, bar_s)
        replayed_bars = []
        for j in range(8, n_bars):
            sig = [(n.x, n.y, n.color, n.direction) for n in by_bar[j]]
            if sig == src_sig_exact:
                saw_exact = True
                replayed_bars.append(j)
            elif sig == src_sig_mirror:
                saw_mirror = True
                replayed_bars.append(j)

        assert len(replayed_bars) == replayed
        # Consecutive replayed bars <= MAX_CONSECUTIVE_REPLAY_BARS
        run = max_run = 0
        replayed_set = set(replayed_bars)
        for k in range(n_bars):
            run = run + 1 if k in replayed_set else 0
            max_run = max(max_run, run)
        assert max_run <= MAX_CONSECUTIVE_REPLAY_BARS

    assert saw_exact and saw_mirror


def test_replay_patterns_skips_flat_flip_sources_and_protects_dots():
    bpm = 120.0
    bar_s = 2.0
    n_bars = 20
    cfg = PlayabilityConfig()

    # 1) Source bar 0 contains a row-0 U/D flat flip -> must never be replayed.
    # Bar 0 matches bars 1..19's entry/exit parity (L U -> L D, R U -> R D).
    notes_flip: list[Note] = []
    for k in range(n_bars):
        t0 = k * bar_s
        if k == 0:
            notes_flip.extend([
                _note(t0 + 0.25, 0, 0, 0, 0, bpm),  # L U at y=0
                _note(t0 + 0.75, 2, 2, 1, 0, bpm),  # R U
                _note(t0 + 1.25, 1, 0, 0, 1, bpm),  # L D at y=0 -> flat row-0 flip!
                _note(t0 + 1.75, 2, 0, 1, 1, bpm),  # R D
            ])
        else:
            notes_flip.extend([
                _note(t0 + 0.25, 1, 2, 0, 0, bpm),
                _note(t0 + 0.75, 2, 2, 1, 0, bpm),
                _note(t0 + 1.25, 1, 0, 0, 1, bpm),
                _note(t0 + 1.75, 2, 0, 1, 1, bpm),
            ])
    rep_flip = _make_repeats(n_bars, bar_s, {j: 0 for j in range(4, n_bars)})
    out_flip, replayed_flip = replay_patterns(notes_flip, repeats=rep_flip, bpm=bpm, mask_cfg=cfg, rng=random.Random(0))
    assert replayed_flip == 0
    assert out_flip == notes_flip
    # Counterfactual: lifting y=0 -> y=2 on bar 0's L U note removes the flat flip and allows replay.
    notes_lifted = list(notes_flip)
    notes_lifted[0] = _note(0.25, 0, 2, 0, 0, bpm)
    out_lifted, replayed_lifted = replay_patterns(notes_lifted, repeats=rep_flip, bpm=bpm, mask_cfg=cfg, rng=random.Random(0))
    assert replayed_lifted >= 1
    assert final_flow_ok(notes_lifted, out_lifted, cfg)

    # 2) Source bar 1 starts with a standalone dot (after > 1 beat rest from bar 0) -> rejected when
    #    target bar 10's predecessor bar 9 ends < 1 beat (0.50 s) before it, accepted when >= 1 beat.
    notes_dot: list[Note] = []
    for k in range(n_bars):
        t0 = k * bar_s
        if k == 0:
            notes_dot.extend([
                _note(t0 + 0.25, 1, 2, 0, 0, bpm),
                _note(t0 + 0.75, 2, 2, 1, 0, bpm),
                _note(t0 + 1.25, 1, 0, 0, 1, bpm),
                _note(t0 + 1.50, 2, 0, 1, 1, bpm),  # gap to bar 1 dot (2.25) is >= 0.75 s >= 1 beat
            ])
        elif k == 1:
            notes_dot.extend([
                _note(t0 + 0.25, 0, 1, 0, 8, bpm),  # standalone dot on calm rest spot (flips D -> U)
                _note(t0 + 0.75, 2, 2, 1, 0, bpm),  # R U (after R D)
                _note(t0 + 1.25, 1, 0, 0, 1, bpm),  # L D (after L dot [U])
                _note(t0 + 1.75, 2, 0, 1, 1, bpm),  # R D (after R U)
            ])
        elif k == 9:
            notes_dot.extend([
                _note(t0 + 0.25, 1, 2, 0, 0, bpm),
                _note(t0 + 0.75, 2, 2, 1, 0, bpm),
                _note(t0 + 1.85, 1, 0, 0, 1, bpm),  # ends 0.40 s (< 0.50 s = 1 beat) before bar 10's 20.25
                _note(t0 + 1.90, 2, 0, 1, 1, bpm),  # also < 1 beat on right hand so mirror also fails guard
            ])
        else:
            notes_dot.extend([
                _note(t0 + 0.25, 1, 2, 0, 0, bpm),
                _note(t0 + 0.75, 2, 2, 1, 0, bpm),
                _note(t0 + 1.25, 1, 0, 0, 1, bpm),
                _note(t0 + 1.75, 2, 0, 1, 1, bpm),
            ])
    rep_dot = _make_repeats(n_bars, bar_s, {10: 1})
    out_dot, replayed_dot = replay_patterns(notes_dot, repeats=rep_dot, bpm=bpm, mask_cfg=cfg, rng=random.Random(0))
    assert replayed_dot == 0
    assert out_dot == notes_dot
    # Counterfactual: moving bar 9's final L/R notes earlier to leave >= 1 beat rest allows replay.
    notes_dot_calm = [
        _note(19.25, 1, 0, 0, 1, bpm) if abs(n.time_s - 19.85) < 1e-9 else
        _note(19.50, 2, 0, 1, 1, bpm) if abs(n.time_s - 19.90) < 1e-9 else n
        for n in notes_dot
    ]
    out_dot_calm, replayed_dot_calm = replay_patterns(notes_dot_calm, repeats=rep_dot, bpm=bpm, mask_cfg=cfg, rng=random.Random(0))
    assert replayed_dot_calm == 1
    assert final_flow_ok(notes_dot_calm, out_dot_calm, cfg)

    # 3) Mirrored same-time arrow stack: exact parity fails at bar 10 entry, so replay_patterns mirrors
    #    bar 0's left-hand stack onto bar 10's right hand (x >= 2) cleanly.
    notes_stack: list[Note] = []
    for k in range(n_bars):
        t0 = k * bar_s
        if k == 0:
            notes_stack.extend([
                _note(t0 + 0.25, 0, 0, 0, 1, bpm),  # L stack D (cell 1)
                _note(t0 + 0.25, 0, 1, 0, 1, bpm),  # L stack D (cell 2)
                _note(t0 + 0.75, 2, 2, 1, 0, bpm),  # R U
                _note(t0 + 1.25, 1, 2, 0, 0, bpm),  # L U
                _note(t0 + 1.75, 2, 0, 1, 1, bpm),  # R D
            ])
        elif k == 9:
            notes_stack.extend([
                _note(t0 + 0.25, 1, 2, 0, 0, bpm),
                _note(t0 + 0.75, 2, 0, 1, 1, bpm),
                _note(t0 + 1.25, 1, 0, 0, 1, bpm),  # L ends D -> blocks exact (L starts D), allows mirror (L starts U)
                _note(t0 + 1.75, 2, 2, 1, 0, bpm),  # R ends U -> blocks exact (R starts U), allows mirror (R starts D)
            ])
        elif k == 10:
            notes_stack.extend([
                _note(t0 + 0.25, 1, 2, 0, 0, bpm),
                _note(t0 + 0.75, 2, 0, 1, 1, bpm),
                _note(t0 + 1.25, 1, 0, 0, 1, bpm),
                _note(t0 + 1.75, 2, 2, 1, 0, bpm),
            ])
        else:
            notes_stack.extend([
                _note(t0 + 0.25, 1, 2, 0, 0, bpm),
                _note(t0 + 0.75, 2, 2, 1, 0, bpm),
                _note(t0 + 1.25, 1, 0, 0, 1, bpm),
                _note(t0 + 1.75, 2, 0, 1, 1, bpm),
            ])
    rep_stack = _make_repeats(n_bars, bar_s, {10: 0})
    out_stack, replayed_stack = replay_patterns(notes_stack, repeats=rep_stack, bpm=bpm, mask_cfg=cfg, rng=random.Random(0))
    assert replayed_stack == 1
    assert final_flow_ok(notes_stack, out_stack, cfg)
    assert check_exact(out_stack, cfg).hard == ()
    bar10_at_025 = [n for n in out_stack if abs(n.time_s - 20.25) < 1e-9]
    assert len(bar10_at_025) == 2
    assert all(n.color == 1 and n.x == 3 and n.direction == 1 for n in bar10_at_025)
    assert {n.y for n in bar10_at_025} == {0, 1}

    # 4) Source bar containing a same-hand soft reset (angle_diff < 90 deg after >= 0.8s rest) is skipped.
    notes_reset: list[Note] = []
    for k in range(n_bars):
        t0 = k * bar_s
        if k == 0:
            notes_reset.extend([
                _note(t0 + 0.10, 1, 2, 0, 0, bpm),  # L U
                _note(t0 + 0.50, 2, 2, 1, 0, bpm),  # R U
                _note(t0 + 1.10, 1, 2, 0, 0, bpm),  # L U again after 1.0 s -> soft reset (0 deg < 90 deg)
                _note(t0 + 1.60, 2, 0, 1, 1, bpm),  # R D
            ])
        else:
            notes_reset.extend([
                _note(t0 + 0.10, 1, 2, 0, 0, bpm),
                _note(t0 + 0.50, 2, 2, 1, 0, bpm),
                _note(t0 + 1.10, 1, 0, 0, 1, bpm),
                _note(t0 + 1.60, 2, 0, 1, 1, bpm),
            ])
    rep_reset = _make_repeats(n_bars, bar_s, {10: 0})
    out_reset, replayed_reset = replay_patterns(notes_reset, repeats=rep_reset, bpm=bpm, mask_cfg=cfg, rng=random.Random(0))
    assert replayed_reset == 0
    assert out_reset == notes_reset


def test_replay_patterns_cross_bar_tech_dot_tails_stay_final_flow_ok():
    bpm = 120.0
    bar_s = 2.0
    n_bars = 20
    cfg = PlayabilityConfig()

    # Case A: bar 0 ends at 1.97 s with L D, and add_dot_stacks places an added dot tail at 2.0325 s
    # (inside bar 1!). Copying bar 1 to bar 12 must NOT copy an orphaned dot tail.
    # Case B: bar 9 ends at 19.97 s with L D and its added dot tail is at 20.0325 s (inside bar 10!).
    # Replacing/mirroring bar 9 from bar 3 (which also has a note at +1.97 s) must NOT orphan bar 10's dot tail.
    pre_flow: list[Note] = []
    for k in range(n_bars):
        t0 = k * bar_s
        if k in (0, 9):
            pre_flow.extend([
                _note(t0 + 0.25, 1, 2, 0, 0, bpm),
                _note(t0 + 0.75, 2, 2, 1, 0, bpm),
                _note(t0 + 1.25, 2, 0, 1, 1, bpm),
                _note(t0 + 1.97, 1, 1, 0, 1, bpm),  # L D near bar end; dot tail spills into k+1
            ])
        elif k == 3:
            pre_flow.extend([
                _note(t0 + 0.25, 1, 2, 0, 0, bpm),
                _note(t0 + 0.75, 2, 2, 1, 0, bpm),
                _note(t0 + 1.25, 2, 0, 1, 1, bpm),
                _note(t0 + 1.97, 0, 1, 0, 7, bpm),  # L DR at +1.97; matches bar 9 offsets
            ])
        else:
            pre_flow.extend([
                _note(t0 + 0.25, 1, 2, 0, 0, bpm),
                _note(t0 + 0.75, 2, 2, 1, 0, bpm),
                _note(t0 + 1.25, 1, 0, 0, 1, bpm),
                _note(t0 + 1.75, 2, 0, 1, 1, bpm),
            ])

    # Add TECH dot tails: two cross-bar tails (bar 0 -> bar 1 and bar 9 -> bar 10) plus
    # an in-bar tail in bar 2 at 4.3125 (head 4.25 L U).
    tail_bar1 = _note(2.0325, 1, 0, 0, 8, bpm)
    tail_bar2 = _note(4.3125, 1, 2, 0, 8, bpm)
    tail_bar10 = _note(20.0325, 1, 0, 0, 8, bpm)
    post_stacks = sorted(
        pre_flow + [tail_bar1, tail_bar2, tail_bar10],
        key=lambda n: (n.time_s, n.color, n.x, n.y),
    )
    added = _added_dot_tails(pre_flow, post_stacks)
    assert added == frozenset({(2.0325, 0), (4.3125, 0), (20.0325, 0)})
    assert final_flow_ok(pre_flow, post_stacks, cfg)

    repeats = _make_repeats(n_bars, bar_s, {12: 1, 9: 3, 14: 2})
    repeats.sim[12, 1] = 0.98
    repeats.sim[9, 3] = 0.95
    repeats.sim[14, 2] = 0.90
    out, replayed = replay_patterns(post_stacks, repeats=repeats, bpm=bpm, mask_cfg=cfg, rng=random.Random(0), added=added)
    assert final_flow_ok(pre_flow, out, cfg)
    out_added = _added_dot_tails(pre_flow, out)
    assert check_exact([n for n in out if (n.time_s, n.color) not in out_added], cfg).hard == ()
    by_bar = _bar_notes_map(out, bar_s)
    assert not any(n.direction == 8 for n in by_bar[12])
    assert any(abs(n.time_s - 19.97) < 1e-9 for n in by_bar[9])
    assert replayed == 1
    assert any(n.direction == 8 for n in by_bar[14])

    # End-to-end apply_flow(level=2) with a cross-bar candidate: final_flow_ok must hold.
    env = np.ones(int((n_bars * bar_s + 2.0) / (512 / SAMPLE_RATE)), dtype=np.float32)
    out_flow, _ = apply_flow(
        pre_flow, bpm=bpm, onset_env=env, onset_hop_s=512 / SAMPLE_RATE,
        level=2, seed=0, mask_cfg=cfg, repeats=repeats,
    )
    assert final_flow_ok(pre_flow, out_flow, cfg)


def test_replay_patterns_float_bar_boundary_consistent_with_bisect():
    # Regression for float64 (j * bar_s) / bar_s < j (e.g. bpm=81, j=11).
    bpm = 81.0
    bar_s = 240.0 / bpm
    j_bad = 11
    assert math.floor((j_bad * bar_s) / bar_s) < j_bad

    n_bars = 24
    cfg = PlayabilityConfig()
    notes: list[Note] = []
    for k in range(n_bars):
        t0 = k * bar_s
        if k == 0:
            notes.extend([
                _note(t0 + 0.0, 0, 1, 0, 4, bpm),
                _note(t0 + 0.5, 2, 2, 1, 0, bpm),
                _note(t0 + 1.0, 1, 0, 0, 7, bpm),
                _note(t0 + 1.5, 2, 0, 1, 1, bpm),
            ])
        else:
            notes.extend([
                _note(t0 + 0.0, 1, 2, 0, 0, bpm),
                _note(t0 + 0.5, 2, 2, 1, 0, bpm),
                _note(t0 + 1.0, 1, 0, 0, 1, bpm),
                _note(t0 + 1.5, 2, 0, 1, 1, bpm),
            ])

    repeats = _make_repeats(n_bars, bar_s, {j_bad: 0})
    out, replayed = replay_patterns(notes, repeats=repeats, bpm=bpm, mask_cfg=cfg, rng=random.Random(0))
    assert replayed == 1
    assert final_flow_ok(notes, out, cfg)
    assert len(out) == len(notes)
    bar11 = [n for n in out if j_bad * bar_s <= n.time_s < (j_bad + 1) * bar_s]
    assert len(bar11) == 4


def test_replay_patterns_rejects_new_exit_flip_even_when_entry_already_flips():
    bpm = 120.0
    bar_s = 2.0
    n_bars = 20
    cfg = PlayabilityConfig()

    # Bar 9 ends with L D(0), R D(0) -> bar 10 enters L U(0), R U(0) (existing entry flip on both hands),
    # then ends with L D(2), R D(2) -> bar 11 starts L U(0), R U(0) (NO exit flip).
    # Source bar 0 does U(0)->D(2)->U(2)->D(0) on both hands (clean internally, keeps entry flip,
    # and would introduce a NEW exit flip against bar 11 on both exact and mirrored polarities).
    notes: list[Note] = []
    for k in range(n_bars):
        t0 = k * bar_s
        if k == 0:
            notes.extend([
                _note(t0 + 0.15, 1, 0, 0, 0, bpm),  # L U(0)
                _note(t0 + 0.35, 2, 0, 1, 0, bpm),  # R U(0)
                _note(t0 + 0.65, 1, 2, 0, 1, bpm),  # L D(2)
                _note(t0 + 0.85, 2, 2, 1, 1, bpm),  # R D(2)
                _note(t0 + 1.15, 1, 2, 0, 0, bpm),  # L U(2)
                _note(t0 + 1.35, 2, 2, 1, 0, bpm),  # R U(2)
                _note(t0 + 1.65, 1, 0, 0, 1, bpm),  # L D(0) -> exit at y=0
                _note(t0 + 1.85, 2, 0, 1, 1, bpm),  # R D(0) -> exit at y=0
            ])
        elif k == 9:
            notes.extend([
                _note(t0 + 0.15, 1, 2, 0, 0, bpm),
                _note(t0 + 0.35, 2, 2, 1, 0, bpm),
                _note(t0 + 0.65, 1, 0, 0, 1, bpm),
                _note(t0 + 0.85, 2, 0, 1, 1, bpm),
                _note(t0 + 1.15, 1, 2, 0, 0, bpm),
                _note(t0 + 1.35, 2, 2, 1, 0, bpm),
                _note(t0 + 1.65, 1, 0, 0, 1, bpm),  # L D(0) -> entry flip with bar 10
                _note(t0 + 1.85, 2, 0, 1, 1, bpm),  # R D(0) -> entry flip with bar 10
            ])
        elif k == 10:
            notes.extend([
                _note(t0 + 0.15, 1, 0, 0, 0, bpm),  # L U(0) (entry flip)
                _note(t0 + 0.35, 2, 0, 1, 0, bpm),  # R U(0) (entry flip)
                _note(t0 + 0.65, 1, 2, 0, 1, bpm),
                _note(t0 + 0.85, 2, 2, 1, 1, bpm),
                _note(t0 + 1.15, 1, 2, 0, 0, bpm),
                _note(t0 + 1.35, 2, 2, 1, 0, bpm),
                _note(t0 + 1.65, 1, 2, 0, 1, bpm),  # L D(2) (no exit flip)
                _note(t0 + 1.85, 2, 2, 1, 1, bpm),  # R D(2) (no exit flip)
            ])
        elif k == 11:
            notes.extend([
                _note(t0 + 0.15, 1, 0, 0, 0, bpm),  # L U(0)
                _note(t0 + 0.35, 2, 0, 1, 0, bpm),  # R U(0)
                _note(t0 + 0.65, 1, 2, 0, 1, bpm),
                _note(t0 + 0.85, 2, 2, 1, 1, bpm),
                _note(t0 + 1.15, 1, 2, 0, 0, bpm),
                _note(t0 + 1.35, 2, 2, 1, 0, bpm),
                _note(t0 + 1.65, 1, 2, 0, 1, bpm),
                _note(t0 + 1.85, 2, 2, 1, 1, bpm),
            ])
        else:
            notes.extend([
                _note(t0 + 0.15, 1, 2, 0, 0, bpm),
                _note(t0 + 0.35, 2, 2, 1, 0, bpm),
                _note(t0 + 0.65, 1, 0, 0, 1, bpm),
                _note(t0 + 0.85, 2, 0, 1, 1, bpm),
                _note(t0 + 1.15, 1, 2, 0, 0, bpm),
                _note(t0 + 1.35, 2, 2, 1, 0, bpm),
                _note(t0 + 1.65, 1, 0, 0, 1, bpm),
                _note(t0 + 1.85, 2, 0, 1, 1, bpm),
            ])

    rep = _make_repeats(n_bars, bar_s, {10: 0})
    out, replayed = replay_patterns(notes, repeats=rep, bpm=bpm, mask_cfg=cfg, rng=random.Random(0))
    assert replayed == 0
    assert out == notes


def test_replay_patterns_chained_source_not_overwritten():
    bpm = 120.0
    bar_s = 2.0
    n_bars = 40
    cfg = PlayabilityConfig()

    notes: list[Note] = []
    for k in range(n_bars):
        t0 = k * bar_s
        if k == 10:
            notes.extend([
                _note(t0 + 0.25, 0, 1, 0, 4, bpm),
                _note(t0 + 0.75, 2, 2, 1, 0, bpm),
                _note(t0 + 1.25, 1, 0, 0, 7, bpm),
                _note(t0 + 1.75, 2, 0, 1, 1, bpm),
            ])
        else:
            notes.extend([
                _note(t0 + 0.25, 1, 2, 0, 0, bpm),
                _note(t0 + 0.75, 2, 2, 1, 0, bpm),
                _note(t0 + 1.25, 1, 0, 0, 1, bpm),
                _note(t0 + 1.75, 2, 0, 1, 1, bpm),
            ])

    rep = _make_repeats(n_bars, bar_s, {18: 10, 10: 0})
    rep.sim[18, 10] = 0.95
    rep.sim[10, 0] = 0.85
    out, replayed = replay_patterns(notes, repeats=rep, bpm=bpm, mask_cfg=cfg, rng=random.Random(0))
    assert replayed == 1
    by_bar = _bar_notes_map(out, bar_s)
    assert [(n.x, n.y, n.color, n.direction) for n in by_bar[10]] == [
        (0, 1, 0, 4), (2, 2, 1, 0), (1, 0, 0, 7), (2, 0, 1, 1),
    ]



def test_replay_copied_dot_tail_on_pre_flow_key_never_breaks_final_flow_ok():
    bpm = 120.0
    bar_s = 2.0
    n_bars = 16
    cfg = PlayabilityConfig()
    src_bar, tgt_bar = 3, 12

    pre_flow: list[Note] = []
    for k in range(n_bars):
        t0 = k * bar_s
        if k == tgt_bar:
            # Same onsets as the source bar but the hands swapped at the tail slot: the pre-flow
            # L note sits on exactly the (time, color) key the copied dot tail will land on.
            pre_flow.extend([
                _note(t0 + 0.25, 1, 2, 0, 0, bpm),
                _note(t0 + 0.75, 2, 2, 1, 0, bpm),
                _note(t0 + 1.25, 2, 0, 1, 1, bpm),
                _note(t0 + 1.3125, 1, 0, 0, 1, bpm),
            ])
        else:
            pre_flow.extend([
                _note(t0 + 0.25, 1, 2, 0, 0, bpm),
                _note(t0 + 0.75, 2, 2, 1, 0, bpm),
                _note(t0 + 1.25, 1, 0, 0, 1, bpm),
                _note(t0 + 1.3125, 2, 0, 1, 1, bpm),
            ])

    # Flow-added dot tail of the source bar's L head (same onset slot as the R note there).
    tail = _note(src_bar * bar_s + 1.3125, 1, 1, 0, 8, bpm)
    post = sorted(pre_flow + [tail], key=lambda n: (n.time_s, n.color, n.x, n.y))
    added = _added_dot_tails(pre_flow, post)
    assert added == frozenset({(src_bar * bar_s + 1.3125, 0)})
    assert final_flow_ok(pre_flow, post, cfg)

    repeats = _make_repeats(n_bars, bar_s, {tgt_bar: src_bar})
    repeats.sim[tgt_bar, src_bar] = 0.98
    for seed in range(6):
        out, _ = replay_patterns(
            post, repeats=repeats, bpm=bpm, mask_cfg=cfg, rng=random.Random(seed), added=added,
            pre_keys={(n.time_s, n.color) for n in pre_flow},
        )
        assert final_flow_ok(pre_flow, out, cfg)
