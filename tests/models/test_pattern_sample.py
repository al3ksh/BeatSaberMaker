import numpy as np
import pytest
import torch

from bsmaker.formats.events import FRAMES_PER_SECOND, to_events
from bsmaker.formats.tokens import NONE_TOKEN, VOCAB_SIZE
from bsmaker.models.pattern import sample as S
from bsmaker.models.pattern.model import PatternConfig, PatternModel
from bsmaker.models.pattern.sample import (
    LoadedPattern, _top_p_probs, load_pattern, sample_notes, window_gaps)
from bsmaker.playability.checker import check
from bsmaker.playability.config import PlayabilityConfig

CFG = PlayabilityConfig()
STATS = {"mel_mean": 0.0, "mel_std": 1.0}


@pytest.fixture(scope="module")
def loaded() -> LoadedPattern:
    torch.manual_seed(0)
    cfg = PatternConfig(d_model=32, n_layers=2, n_heads=2, context=16, dropout=0.0)
    return LoadedPattern(PatternModel(cfg).eval(), cfg, STATS)


def _mel(seconds: float) -> np.ndarray:
    return np.zeros((int(seconds * FRAMES_PER_SECOND) + 10, 80), np.float16)


def _times(n: int, seed: int = 0) -> np.ndarray:
    gaps = np.random.default_rng(seed).choice([0.12, 0.2, 0.3, 0.5, 1.0], size=n)
    return 2.0 + np.cumsum(gaps)


def _run(loaded, times, seed=1, cfg=CFG, **kw):
    return sample_notes(loaded, times, _mel(times[-1] + 1), 4.0, ("tech",), seed, cfg, "cpu", **kw)


def test_sampled_map_has_zero_hard_violations(loaded) -> None:
    times = _times(500)
    notes, rep = _run(loaded, times)
    events, dropped = to_events(notes)
    assert dropped == 0
    assert check(events, CFG).hard == ()
    assert len(notes) > 100
    assert rep.n_events == 500


def test_close_events_stay_valid(loaded) -> None:
    times = 2.0 + np.arange(300) * 0.02
    notes, rep = _run(loaded, times)
    events, dropped = to_events(notes)
    assert dropped == 0
    assert check(events, CFG).hard == ()
    assert rep.n_skipped > 0


def test_never_returns_masked_token(loaded, monkeypatch) -> None:
    def fake(state, t, hand, cfg, other_token=NONE_TOKEN):
        m = np.zeros(VOCAB_SIZE, bool)
        m[[NONE_TOKEN, 10 + 50 * hand]] = True
        return m
    monkeypatch.setattr(S, "allowed", fake)
    notes, _ = _run(loaded, _times(60))
    assert notes
    assert {(n.color, n.x, n.y, n.direction) for n in notes} <= {
        *{(c, *S.decode_token(10 + 50 * c)) for c in (0, 1)}}


def test_all_blocked_events_are_skipped_and_terminate(loaded, monkeypatch) -> None:
    def only_none(state, t, hand, cfg, other_token=NONE_TOKEN):
        m = np.zeros(VOCAB_SIZE, bool)
        m[NONE_TOKEN] = True
        return m
    monkeypatch.setattr(S, "allowed", only_none)
    notes, rep = _run(loaded, _times(50))
    assert notes == []
    assert rep.n_skipped == 50 and rep.n_events == 50


def test_same_seed_same_notes(loaded) -> None:
    t = _times(80)
    assert _run(loaded, t, seed=3)[0] == _run(loaded, t, seed=3)[0]


def test_different_seed_differs(loaded) -> None:
    t = _times(80)
    assert _run(loaded, t, seed=3)[0] != _run(loaded, t, seed=4)[0]


def test_no_event_with_both_hands_empty(loaded) -> None:
    notes, rep = _run(loaded, _times(200))
    events, _ = to_events(notes)
    assert all(e.left != NONE_TOKEN or e.right != NONE_TOKEN for e in events)
    assert len(events) == rep.n_events - rep.n_skipped


def test_history_excludes_skipped_events() -> None:
    kept = [1.0, 1.5, 2.0, 3.0]
    future = [4.0, 4.5, 5.0, 5.5, 6.0]
    g = window_gaps(kept, 3.5, future, context=16)
    assert g.shape == (5, 5)
    np.testing.assert_allclose(g[:, 0], np.log1p([0, 0.5, 0.5, 1.0, 0.5]), atol=1e-6)
    np.testing.assert_allclose(g[-1, 1:], np.log1p([0.5, 1.0, 1.5, 2.0]), atol=1e-6)
    # Truncated window: the first row still measures its gap to the kept event before it.
    g2 = window_gaps(kept, 3.5, future, context=3)
    assert g2.shape == (3, 5)
    np.testing.assert_allclose(g2[0, 0], np.log1p(0.5), atol=1e-6)
    # no kept events: previous gap is 0
    assert window_gaps([], 2.0, [2.5], context=4)[0, 0] == 0.0


def test_top_p_keeps_at_least_one_token() -> None:
    p = torch.tensor([0.7, 0.2, 0.1])
    out = _top_p_probs(p, 0.0)
    assert out.tolist() == [1.0, 0.0, 0.0]
    out = _top_p_probs(p, 0.8)
    assert out[2] == 0 and out[0] > 0 and out[1] > 0
    assert float(out.sum()) == pytest.approx(1.0)


def test_load_pattern_requires_stats(loaded, tmp_path) -> None:
    from dataclasses import asdict
    payload = {"model": loaded.model.state_dict(), "model_config": asdict(loaded.config)}
    path = tmp_path / "c.pt"
    torch.save(payload, path)
    with pytest.raises(ValueError, match="stats"):
        load_pattern(path, "cpu")
    torch.save({**payload, "stats": STATS}, path)
    assert load_pattern(path, "cpu").config == loaded.config
