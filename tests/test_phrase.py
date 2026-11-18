import numpy as np
import pytest
import torch

from bsmaker import phrase as P
from bsmaker.audio.bar_similarity import BarRepeats, features_from_frames, find_repeats
from bsmaker.formats.events import FRAMES_PER_SECOND, to_events
from bsmaker.formats.tokens import NONE_TOKEN, VOCAB_SIZE, encode_token
from bsmaker.models.pattern.model import PatternConfig, PatternModel
from bsmaker.models.pattern.sample import LoadedPattern, sample_notes
from bsmaker.playability.checker import check
from bsmaker.playability.config import PlayabilityConfig

FPS = FRAMES_PER_SECOND
BAR = 2.0  # 120 bpm


def _rep(source) -> BarRepeats:
    return BarRepeats(BAR, np.asarray(source, dtype=np.int64))


# ---------------------------------------------------------------- bar similarity
def test_repeated_section_finds_its_earlier_bars() -> None:
    rng = np.random.default_rng(0)
    base = rng.normal(size=(16, 24))
    feats = np.concatenate([base, base + rng.normal(scale=0.05, size=base.shape)])
    feats /= np.linalg.norm(feats, axis=1, keepdims=True) / np.sqrt(2)
    rep = find_repeats(feats, np.zeros(32), BAR)
    assert list(rep.source[16:]) == list(range(16))


def test_quiet_bars_never_repeat() -> None:
    feats = np.ones((12, 24))
    loud = np.zeros(12)
    loud[[3, 9]] = -80.0
    rep = find_repeats(feats, loud, BAR)
    assert rep.source[3] == -1 and rep.source[9] == -1
    assert all(rep.source[j] not in (3, 9) for j in range(12))


def test_features_from_frames_are_unit_halves() -> None:
    rng = np.random.default_rng(1)
    chroma, mfcc = rng.random((12, 400)), rng.normal(size=(13, 400))
    rms = np.full(400, 0.1)
    feats, loud = features_from_frames(chroma, mfcc, rms, 0.02, BAR)
    assert feats.shape == (4, 24)
    assert np.allclose(np.linalg.norm(feats[:, :12], axis=1), 1.0)
    assert np.allclose(np.linalg.norm(feats[:, 12:], axis=1), 1.0)
    assert np.allclose(loud, -20.0)


# ---------------------------------------------------------------- timing
def test_repeat_rhythm_copies_offsets_where_the_curve_allows() -> None:
    a = [0.0, 0.5, 0.75, 1.5]
    b = [0.25, 0.5, 1.0, 1.25]
    times = np.array([BAR * 1 + o for o in a] + [BAR * 3 + o for o in b])
    curve = np.ones(int(10 * BAR * FPS), np.float32)
    out, n = P.repeat_rhythm(times, curve, 0.8, _rep([-1, -1, -1, 1, -1]), FPS)
    assert n == 1
    assert np.allclose(out, [BAR + o for o in a] + [3 * BAR + o for o in a])


def test_repeat_rhythm_drops_unsupported_offsets_and_keeps_bar_if_too_few() -> None:
    a = [0.0, 0.5, 1.0, 1.5]
    times = np.array([BAR + o for o in a] + [3 * BAR + o for o in (0.25, 0.75, 1.25)])
    curve = np.ones(int(10 * BAR * FPS), np.float32)
    # one offset (3 bars + 0.5 s) without curve support: 3 of 4 survive (>= 75%)
    f = int(round((3 * BAR + 0.5) * FPS))
    curve[f - 8:f + 9] = 0.1
    out, n = P.repeat_rhythm(times, curve, 0.8, _rep([-1, -1, -1, 1]), FPS)
    assert n == 1 and np.allclose(out[4:], [3 * BAR + o for o in (0.0, 1.0, 1.5)])
    # two offsets without support: the bar keeps its own rhythm
    g = int(round((3 * BAR + 1.0) * FPS))
    curve[g - 8:g + 9] = 0.1
    out, n = P.repeat_rhythm(times, curve, 0.8, _rep([-1, -1, -1, 1]), FPS)
    assert n == 0 and np.allclose(out, times)


def test_repeat_rhythm_leaves_rest_bars_alone_and_keeps_min_gap() -> None:
    curve = np.ones(int(10 * BAR * FPS), np.float32)
    times = np.array([BAR + o for o in (0.0, 0.5, 1.0, 1.5)] + [3 * BAR + 0.5])
    out, n = P.repeat_rhythm(times, curve, 0.8, _rep([-1, -1, -1, 1]), FPS)
    assert n == 0 and np.allclose(out, times)  # bar 3 is a rest (1 event)
    # an event of the previous bar sits 10 ms before the copied first offset: that one is dropped
    times = np.array([BAR + o for o in (0.0, 0.5, 1.0, 1.5)] + [3 * BAR - 0.01]
                     + [3 * BAR + o for o in (0.25, 0.75, 1.25, 1.75)])
    a = [0.0, 0.5, 1.0, 1.5]
    out, n = P.repeat_rhythm(times, curve, 0.8, _rep([-1, -1, -1, 1]), FPS)
    assert n == 1 and np.allclose(out[5:], [3 * BAR + o for o in a[1:]])
    assert np.all(np.diff(out) >= P.MIN_GAP_S - 1e-9)


# ---------------------------------------------------------------- pattern links
def test_links_need_matching_rhythm_and_no_copy_of_a_copy() -> None:
    off = [0.0, 0.5, 1.0]
    times = np.array([k * BAR + o for k in (0, 1, 2) for o in off] + [3 * BAR + o for o in (0.0, 0.75)])
    links = P.pattern_links(times, _rep([-1, 0, 1, 2]), seed=0)
    assert links.n_bars == 1  # bar 1 links to 0; bar 2's source is a copy; bar 3's rhythm differs
    assert {k for k in links.link} == {P.PatternLinks.key(BAR + o) for o in off}
    assert links.link[P.PatternLinks.key(BAR + 0.5)][0] == P.PatternLinks.key(0.5)


def test_bias_favours_source_tokens_and_mirror() -> None:
    lt, rt = encode_token(1, 0, 6), encode_token(2, 0, 1)
    links = P.PatternLinks({P.PatternLinks.key(2.5): (P.PatternLinks.key(0.5), False),
                            P.PatternLinks.key(4.5): (P.PatternLinks.key(0.5), True)})
    links.record(0.5, lt, rt)
    z = torch.zeros(VOCAB_SIZE)
    assert int(torch.argmax(links.bias(z, 2.5, 0))) == lt
    assert int(torch.argmax(links.bias(z, 2.5, 1))) == rt
    assert links.wanted(4.5, 0) == encode_token(1, 0, 1)   # mirror of right D at x=2
    assert links.wanted(4.5, 1) == encode_token(2, 0, 7)   # mirror of left DL at x=1 -> DR
    assert links.wanted(3.0, 0) is None
    assert torch.equal(links.bias(z, 3.0, 0), z)


def test_mirror_token() -> None:
    assert P.mirror_token(NONE_TOKEN) == NONE_TOKEN
    assert P.mirror_token(encode_token(0, 2, 4)) == encode_token(3, 2, 5)
    assert P.mirror_token(P.mirror_token(encode_token(1, 1, 2))) == encode_token(1, 1, 2)


# ---------------------------------------------------------------- sampler hook
@pytest.fixture(scope="module")
def loaded() -> LoadedPattern:
    torch.manual_seed(0)
    cfg = PatternConfig(d_model=32, n_layers=2, n_heads=2, context=16, dropout=0.0)
    return LoadedPattern(PatternModel(cfg).eval(), cfg, {"mel_mean": 0.0, "mel_std": 1.0})


def _bars_alike(notes, b0, b1) -> float:
    def sig(k):
        return {(round(n.time_s - k * BAR, 3), n.color, n.x, n.y, n.direction)
                for n in notes if k * BAR <= n.time_s < (k + 1) * BAR}
    s0, s1 = sig(b0), sig(b1)
    return len(s0 & s1) / max(len(s0 | s1), 1)


def test_sampler_copies_linked_bars_and_stays_playable(loaded) -> None:
    cfg = PlayabilityConfig()
    off = [0.0, 0.5, 0.75, 1.0, 1.5]
    n_bars = 12
    times = np.array([k * BAR + o for k in range(1, n_bars) for o in off])
    mel = np.zeros((int((n_bars + 1) * BAR * FPS) + 10, 80), np.float16)
    src = [-1] * n_bars
    for k in range(5, n_bars):
        src[k] = k - 4
    links = P.pattern_links(times, _rep(src), seed=0, bonus=8.0)
    notes, _ = sample_notes(loaded, times, mel, 4.0, ("tech",), 1, cfg, "cpu", phrase=links)
    plain, _ = sample_notes(loaded, times, mel, 4.0, ("tech",), 1, cfg, "cpu")
    events, dropped = to_events(notes)
    assert dropped == 0 and check(events, cfg).hard == ()
    linked = [k for k in range(5, n_bars) if P.PatternLinks.key(k * BAR) in links.link
              and not links.link[P.PatternLinks.key(k * BAR)][1]]
    assert linked
    copied = np.mean([_bars_alike(notes, k - 4, k) for k in linked])
    base = np.mean([_bars_alike(plain, k - 4, k) for k in linked])
    assert copied > base and copied > 0.5
