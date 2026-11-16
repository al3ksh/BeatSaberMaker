import numpy as np
import pytest
import torch

from bsmaker.formats.events import FRAMES_PER_SECOND, to_events
from bsmaker.formats.tokens import NONE_TOKEN, VOCAB_SIZE
from bsmaker.models.pattern import sample as S
from bsmaker.models.pattern.model import PatternConfig, PatternModel
from bsmaker.models.pattern.sample import (
    DEFAULT_DOT_BRAKE, DEFAULT_DOUBLE_BIAS, DOT_TOKENS, DotBrake, DoubleBias, LoadedPattern, _top_p_probs, load_pattern, sample_notes,
    window_gaps)
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
    assert rep.n_skipped + rep.n_frame_collisions > 0


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
    assert len(events) == rep.n_events - rep.n_skipped - rep.n_frame_collisions


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


def _first_allowed(logits, mask, temperature, top_p, gen, masses):
    allowed_tokens = [i for i in np.flatnonzero(mask) if i != NONE_TOKEN]
    return int(allowed_tokens[0]) if allowed_tokens else NONE_TOKEN


def test_exact_time_gap_below_min_hand_gap_is_masked(loaded, monkeypatch) -> None:
    # 3 sixteenths at 128 BPM = 87.9 ms (< min_hand_gap_s 0.09) but 8 frames = 92.9 ms.
    monkeypatch.setattr(S, "_pick", _first_allowed)
    t0 = 2.0
    times = np.array([t0, t0 + 3 * 60 / 128 / 16])
    notes, rep = _run(loaded, times)
    assert {n.time_s for n in notes} == {t0}
    assert rep.n_skipped == 1


def test_exact_times_on_128_bpm_grid_have_no_hard_violations(loaded) -> None:
    from bsmaker.generate import check_exact
    step = 60 / 128 / 16
    rng = np.random.default_rng(0)
    times = 2.0 + np.cumsum(rng.choice([1, 1, 2, 3], size=400)) * step
    hard = n_skipped = n_events = 0
    for seed in range(4):
        notes, rep = _run(loaded, times, seed=seed)
        hard += len(check_exact(notes, CFG).hard)
        n_skipped += rep.n_skipped
        n_events += rep.n_events
        assert {n.time_s for n in notes} <= set(times.tolist())
    print(f"EXACT hard={hard} skipped={n_skipped}/{n_events}")
    assert hard == 0


def test_frame_collisions_are_not_skips(loaded) -> None:
    t = 2.0 + np.arange(100) * 0.01   # many events per 11.6 ms frame
    _, rep = _run(loaded, t)
    assert rep.n_frame_collisions > 0
    assert rep.n_frame_collisions + rep.n_skipped <= rep.n_events


def _doubles(notes) -> int:
    events, _ = to_events(notes)
    return sum(e.left != NONE_TOKEN and e.right != NONE_TOKEN for e in events)


def test_double_bias_off_reproduces_unbiased_output(loaded) -> None:
    t = _times(300)
    plain = _run(loaded, t, seed=5, double_bias=None)[0]
    assert _run(loaded, t, seed=5, double_bias=DoubleBias(0.0, 0.0, 0.0, 0.0, 4))[0] == plain
    assert _doubles(plain) > 20


def test_default_double_bias_is_active_and_reduces_doubles(loaded) -> None:
    t = _times(600)
    plain = _run(loaded, t, seed=2, double_bias=None)[0]
    default = _run(loaded, t, seed=2)[0]
    assert DEFAULT_DOUBLE_BIAS != DoubleBias(0.0, 0.0, 0.0, 0.0, 4)
    assert _doubles(default) < _doubles(plain)


def test_large_bias_removes_doubles_when_singles_are_allowed(loaded) -> None:
    notes, _ = _run(loaded, _times(400), double_bias=DoubleBias(60.0, 0.0, 0.0, 0.0, 4))
    assert notes and _doubles(notes) == 0
    events, _ = to_events(notes)
    assert check(events, CFG).hard == ()


def test_masked_none_stays_impossible_under_bias(loaded, monkeypatch) -> None:
    real = S.allowed

    def no_right_none(state, t, hand, cfg, other_token=NONE_TOKEN):
        m = real(state, t, hand, cfg, other_token=other_token)
        if hand == 1 and other_token != NONE_TOKEN:
            m = m.copy()
            m[NONE_TOKEN] = False
        return m
    monkeypatch.setattr(S, "allowed", no_right_none)
    notes, _ = _run(loaded, _times(300), double_bias=DoubleBias(60.0, 0.0, 0.0, 0.0, 4))
    events, _ = to_events(notes)
    left_events = [e for e in events if e.left != NONE_TOKEN]
    assert left_events and all(e.right != NONE_TOKEN for e in left_events)


def test_bias_keeps_hand_balance(loaded) -> None:
    t = _times(800)

    def left_share(notes) -> float:
        return sum(n.color == 0 for n in notes) / len(notes)
    plain = left_share(_run(loaded, t, seed=1, double_bias=None)[0])
    biased = left_share(_run(loaded, t, seed=1)[0])
    assert abs(biased - plain) < 0.05


@pytest.mark.parametrize("bad", [
    dict(base=-0.1), dict(short=-1.0), dict(very_short=float("nan")), dict(per_run=-0.5),
    dict(run_cap=-1), dict(run_cap=2.5), dict(base=float("inf"))])
def test_double_bias_rejects_invalid_values(bad) -> None:
    with pytest.raises(ValueError):
        DoubleBias(**bad)


def test_double_bias_value_grows_with_run_and_short_gap() -> None:
    b = DoubleBias(base=0.5, short=0.25, very_short=1.0, per_run=0.5, run_cap=3)
    assert b.at(0.5, 0) == 0.5
    assert b.at(0.2, 0) == 0.75
    assert b.at(0.1, 0) == 1.75
    assert b.at(0.5, 2) == 1.5
    assert b.at(0.5, 9) == 2.0      # run capped at 3


def _dot_share(notes) -> float:
    return sum(n.direction == 8 for n in notes) / len(notes)


def test_dot_brake_off_reproduces_unbraked_output(loaded) -> None:
    t = _times(300)
    plain = _run(loaded, t, seed=5, dot_brake=None)[0]
    assert _run(loaded, t, seed=5, dot_brake=DotBrake(0.0, 4))[0] == plain


def test_dot_brake_reduces_dot_runs(loaded) -> None:
    t = _times(800)
    plain = _run(loaded, t, seed=2, dot_brake=None)[0]
    braked = _run(loaded, t, seed=2, dot_brake=DotBrake(50.0, 1))[0]
    assert _dot_share(braked) < _dot_share(plain)
    assert DEFAULT_DOT_BRAKE != DotBrake(0.0, 4)


def test_dots_still_allowed_when_mask_leaves_only_dots(loaded, monkeypatch) -> None:
    def only_dots(state, t, hand, cfg, other_token=NONE_TOKEN):
        m = np.zeros(VOCAB_SIZE, bool)
        m[list(DOT_TOKENS)] = True
        return m
    monkeypatch.setattr(S, "allowed", only_dots)
    notes, _ = _run(loaded, _times(60), dot_brake=DotBrake(80.0, 4))
    assert notes and all(n.direction == 8 for n in notes)


def test_dot_brake_never_samples_masked_token(loaded, monkeypatch) -> None:
    real = S.allowed

    def no_dots(state, t, hand, cfg, other_token=NONE_TOKEN):
        m = real(state, t, hand, cfg, other_token=other_token).copy()
        m[list(DOT_TOKENS)] = False
        return m
    monkeypatch.setattr(S, "allowed", no_dots)
    notes, _ = _run(loaded, _times(200), dot_brake=DotBrake(5.0, 4))
    assert notes and all(n.direction != 8 for n in notes)


@pytest.mark.parametrize("bad", [dict(per_dot=-1.0), dict(per_dot=float("nan")), dict(cap=-1),
                                 dict(cap=1.5), dict(free=-1), dict(per_dot=float("inf"))])
def test_dot_brake_rejects_invalid_values(bad) -> None:
    with pytest.raises(ValueError):
        DotBrake(**bad)


def test_dot_brake_penalty_grows_then_caps() -> None:
    b = DotBrake(per_dot=0.5, cap=3, free=1)
    assert [b.penalty(r) for r in (0, 1, 2, 4, 9)] == [0.0, 0.0, 0.5, 1.5, 1.5]


def _scripted_logits(loaded, script, dot_brake, monkeypatch):
    """Logits seen by `_pick` (left, right per event) while the tokens are forced to `script`."""
    seen, queue = [], [tok for pair in script for tok in pair]

    def fake_pick(logits, mask, temperature, top_p, gen, masses):
        seen.append(logits.clone())
        return queue.pop(0)
    monkeypatch.setattr(S, "_pick", fake_pick)
    times = 2.0 + np.arange(len(script)) * 1.0
    _run(loaded, times, double_bias=None, dot_brake=dot_brake)
    return seen


def test_dot_brake_run_state_per_hand(loaded, monkeypatch) -> None:
    dot, plain, none = DOT_TOKENS[0], 1, NONE_TOKEN
    script = [(dot, none), (none, dot), (dot, dot), (dot, plain), (none, dot), (dot, plain)]
    brake = DotBrake(2.0, cap=2, free=1)
    base = _scripted_logits(loaded, script, None, monkeypatch)
    braked = _scripted_logits(loaded, script, brake, monkeypatch)
    # runs of (left, right) before each event:
    # (0,0) (1,0) (1,1) (2,2) (3,0) (3,1); penalty = 2 * min(max(run - 1, 0), 2)
    expect = [(0, 0), (0, 0), (0, 0), (2, 2), (4, 0), (4, 0)]
    assert len(braked) == 12
    for i, (pl, pr) in enumerate(expect):
        for call, pen in ((2 * i, pl), (2 * i + 1, pr)):
            delta = base[call] - braked[call]
            want = torch.zeros(VOCAB_SIZE)
            want[list(DOT_TOKENS)] = pen
            torch.testing.assert_close(delta, want, atol=1e-5, rtol=0)


def test_sample_notes_defaults_use_default_dot_brake_and_double_bias() -> None:
    import inspect
    params = inspect.signature(sample_notes).parameters
    assert params["dot_brake"].default is DEFAULT_DOT_BRAKE
    assert params["double_bias"].default is DEFAULT_DOUBLE_BIAS


def test_dot_brake_rejects_bool_per_dot() -> None:
    with pytest.raises(ValueError):
        DotBrake(per_dot=True)
