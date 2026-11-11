from dataclasses import replace

from bsmaker.formats.events import Event
from bsmaker.formats.tokens import encode_token
from bsmaker.playability.checker import Report, Violation, check
from bsmaker.playability.config import PlayabilityConfig

CFG = PlayabilityConfig()
UP, DOWN, LEFT, RIGHT, UL, UR, DL, DR, DOT = range(9)


def tok(n):
    return 0 if n is None else encode_token(*n)


def ev(t, left=None, right=None):
    return Event(t, tok(left), tok(right))


def rules(report, kind="hard"):
    return [v.rule for v in getattr(report, kind)]


def test_alternating_up_down_stream_is_clean():
    events = [ev(i * 0.25, right=(2, 1, DOWN if i % 2 == 0 else UP)) for i in range(8)]
    r = check(events, CFG)
    assert r.hard == () and r.soft == ()


def test_double_directional_is_parity_violation():
    r = check([ev(0, right=(2, 1, DOWN)), ev(0.25, right=(2, 1, DOWN))], CFG)
    assert rules(r) == ["parity"]
    assert r.hard[0].hand == 1 and r.hard[0].time_s == 0.25


def test_diagonal_within_45_is_allowed():
    r = check([ev(0, right=(2, 1, DOWN)), ev(0.2, right=(2, 1, UL))], CFG)
    assert r.hard == () and r.soft == ()


def test_sideways_fast_is_violation_slow_is_ok():
    assert rules(check([ev(0, right=(2, 1, DOWN)), ev(0.2, right=(2, 1, LEFT))], CFG)) == ["parity"]
    r = check([ev(0, right=(2, 1, DOWN)), ev(0.5, right=(2, 1, LEFT))], CFG)
    assert r.hard == () and r.soft == ()


def test_reset_after_long_gap_is_soft():
    r = check([ev(0, right=(2, 1, DOWN)), ev(1.0, right=(2, 1, DOWN))], CFG)
    assert r.hard == ()
    assert rules(r, "soft") == ["reset"]


def test_dot_inherits_expected_angle():
    ok = [ev(0, right=(2, 1, DOWN)), ev(0.25, right=(2, 1, DOT)), ev(0.5, right=(2, 1, DOWN))]
    r = check(ok, CFG)
    assert r.hard == () and r.soft == ()
    bad = [ev(0, right=(2, 1, DOWN)), ev(0.25, right=(2, 1, DOT)), ev(0.5, right=(2, 1, UP))]
    assert rules(check(bad, CFG)) == ["parity"]


def test_hand_speed():
    assert rules(check([ev(0, right=(1, 1, DOWN)), ev(0.05, right=(1, 1, UP))], CFG)) == ["hand_speed"]
    r = check([ev(0, right=(0, 1, DOWN)), ev(0.10, right=(3, 1, UP))], CFG)
    assert "hand_speed" in rules(r)
    r = check([ev(0, right=(0, 1, DOWN)), ev(0.16, right=(3, 1, UP))], CFG)
    assert "hand_speed" not in rules(r)


def test_same_cell_double():
    r = check([ev(0, left=(1, 1, DOWN), right=(1, 1, DOWN))], CFG)
    assert rules(r) == ["same_cell"]
    assert r.hard[0].hand == 1


def test_crossover_double():
    events = [ev(0, left=(1, 1, DOWN), right=(2, 1, DOWN))]
    assert check(events, CFG).hard == ()
    cross = [ev(0, left=(2, 1, DOWN), right=(1, 1, DOWN))]
    assert rules(check(cross, CFG)) == ["crossover"]
    assert check(cross, replace(CFG, forbid_crossover=False)).hard == ()


def test_handclap():
    r = check([ev(0, left=(1, 1, RIGHT), right=(2, 1, LEFT))], CFG)
    assert rules(r) == ["handclap"]
    assert check([ev(0, left=(1, 1, DOWN), right=(2, 1, DOWN))], CFG).hard == ()


def test_vision_block():
    r = check([ev(0, left=(1, 1, DOWN)), ev(0.08, right=(1, 0, DOWN))], CFG)
    assert rules(r) == ["vision_block"]
    r = check([ev(0, left=(1, 1, DOWN)), ev(0.20, right=(1, 0, DOWN))], CFG)
    assert r.hard == ()


def test_rare_token_blocked():
    t = encode_token(1, 0, UP)
    cfg = replace(CFG, rare_tokens=(frozenset(), frozenset({t})))
    assert rules(check([ev(0, right=(1, 0, UP))], cfg)) == ["rare_token"]
    assert check([ev(0, left=(1, 0, UP))], cfg).hard == ()


def test_big_jump_is_soft():
    r = check([ev(0, right=(0, 1, DOWN)), ev(0.17, right=(3, 1, UP))], CFG)
    assert rules(r, "soft") == ["big_jump"]


def test_hard_rate():
    assert Report(0, (), ()).hard_rate == 0
    assert Report(4, (Violation(0.0, 1, "parity"),), ()).hard_rate == 0.25


# --- slider tails ---------------------------------------------------------------------------

from bsmaker.playability.checker import allowed, drop_slider_tails  # noqa: E402
from bsmaker.playability.state import initial_state  # noqa: E402

SLIDER = replace(CFG, slider_gap_s=0.12)


def test_slider_tail_dropped():
    events = [ev(0, right=(2, 1, DOWN)), ev(0.05, right=(2, 2, DOT)), ev(0.30, right=(2, 1, UP))]
    r = check(events, SLIDER)
    assert r.hard == () and r.n_notes == 2
    assert "hand_speed" in rules(check(events, replace(CFG, slider_gap_s=0.0)))


def test_slider_tail_needs_compatible_direction():
    events = [ev(0, right=(2, 1, DOWN)), ev(0.05, right=(2, 2, UP))]
    r = check(events, SLIDER)
    assert "hand_speed" in rules(r) and r.n_notes == 2


def test_slider_chain_measured_from_kept_note():
    events = [ev(0, right=(2, 1, DOWN)), ev(0.05, right=(2, 1, DOT)), ev(0.10, right=(2, 1, DOT))]
    assert len(drop_slider_tails(events, 0.12)) == 1
    events.append(ev(0.13, right=(2, 1, DOT)))
    kept = drop_slider_tails(events, 0.12)
    assert [e.time_s for e in kept] == [0, 0.13]


def test_slider_drop_removes_empty_event():
    events = [ev(0, left=(1, 1, DOWN), right=(2, 1, DOWN)),
              ev(0.05, left=(1, 1, DOT), right=(2, 1, DOT)),
              ev(0.30, left=(1, 1, UP))]
    kept = drop_slider_tails(events, 0.12)
    assert [e.time_s for e in kept] == [0, 0.30]
    assert drop_slider_tails(events, 0.0) == events


def test_allowed_refuses_a_profile_with_slider_tails():
    import pytest

    from bsmaker.playability.checker import allowed
    from bsmaker.playability.config import PlayabilityConfig
    from bsmaker.playability.state import initial_state
    with pytest.raises(ValueError):
        allowed(initial_state(), 0.0, 0, PlayabilityConfig(slider_gap_s=0.12))
