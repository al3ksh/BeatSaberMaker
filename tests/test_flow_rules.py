"""Tests for the rule-based flow layer (dot stacks and snake runs)."""
from __future__ import annotations

import random

import numpy as np
import pytest

from bsmaker.audio.beats import ENV_HOP, onset_envelope
from bsmaker.audio.decode import SAMPLE_RATE
from bsmaker.flow_rules import (
    ROTATION,
    SLITHER,
    STACK_CHANCE,
    FlowReport,
    add_dot_stacks,
    apply_flow,
    apply_snakes,
    auto_flow_level,
    final_flow_ok,
    find_runs,
    repair_loose_dots,
    rewrite_run,
    stack_spacing_s,
)
from bsmaker.formats.model import Note
from bsmaker.playability.config import PlayabilityConfig

HOP_S = ENV_HOP / SAMPLE_RATE


def _note(t: float, x: int, y: int, color: int, direction: int, bpm: float = 150.0) -> Note:
    return Note(time_s=t, beat=t * bpm / 60.0, x=x, y=y, color=color, direction=direction)


def test_onset_envelope_shape_and_hop() -> None:
    sr = SAMPLE_RATE
    y = np.zeros(sr * 2, dtype=np.float32)
    y[sr : sr + 512] = 0.5
    env, hop_s = onset_envelope(y)
    assert hop_s == pytest.approx(HOP_S)
    assert env.ndim == 1 and len(env) > 0


def test_stack_spacing() -> None:
    # smallest of {1/16, 1/12, 1/8, 1/6, 1/4} beat that is >= 0.045 s
    assert stack_spacing_s(150.0) == pytest.approx(0.05)  # 1/8
    assert stack_spacing_s(200.0) == pytest.approx(0.05)  # 1/6
    assert stack_spacing_s(220.0) == pytest.approx(60.0 / 220.0 / 6.0)
    assert stack_spacing_s(60.0) == pytest.approx(60.0 / 60.0 / 16.0)
    assert stack_spacing_s(30.0) is None  # 1/16 beat = 0.125 s > 0.09 s
    assert stack_spacing_s(0.0) is None


def test_spacing_half_time_bpm() -> None:
    assert stack_spacing_s(110.0) == pytest.approx(60.0 / 110.0 / 12.0)


def test_stack_chance_by_level() -> None:
    assert STACK_CHANCE == (0.0, 0.25, 0.5, 0.8)


def test_dots_on_cut_ray() -> None:
    bpm = 150.0
    notes = [_note(1.0, 1, 0, 0, 0, bpm), _note(2.0, 1, 0, 0, 1, bpm)]
    env = np.ones(400, dtype=np.float32)
    out, stacks, dots, reverted = add_dot_stacks(
        notes, bpm=bpm, onset_env=env, onset_hop_s=HOP_S,
        chance=1.0, rng=random.Random(0), mask_cfg=PlayabilityConfig(),
    )
    assert stacks == 1 and reverted == 0 and dots in (1, 2)
    added = [n for n in out if n.direction == 8]
    assert len(added) == dots
    for idx, d in enumerate(added, start=1):
        assert d.color == 0
        assert d.direction == 8
        assert d.x == 1
        assert d.y in (1, 2)
        assert d.y == idx
        assert d.time_s == pytest.approx(1.0 + 0.05 * idx)
        assert d.beat == pytest.approx(d.time_s * bpm / 60.0, abs=1e-6)


def test_no_stack_when_ray_leaves_grid() -> None:
    bpm = 150.0
    notes = [_note(1.0, 1, 2, 0, 0, bpm), _note(2.0, 1, 0, 0, 1, bpm)]
    env = np.ones(400, dtype=np.float32)
    out, stacks, dots, reverted = add_dot_stacks(
        notes, bpm=bpm, onset_env=env, onset_hop_s=HOP_S,
        chance=1.0, rng=random.Random(0), mask_cfg=PlayabilityConfig(),
    )
    assert stacks == 0 and dots == 0 and reverted == 0
    assert out == notes


def test_no_stack_when_next_note_close() -> None:
    bpm = 150.0
    notes = [_note(1.0, 1, 0, 0, 0, bpm), _note(1.19, 1, 0, 0, 1, bpm)]
    env = np.ones(400, dtype=np.float32)
    out, stacks, dots, _ = add_dot_stacks(
        notes, bpm=bpm, onset_env=env, onset_hop_s=HOP_S,
        chance=1.0, rng=random.Random(0), mask_cfg=PlayabilityConfig(),
    )
    assert stacks == 0 and dots == 0
    assert out == notes


def test_no_stack_when_other_hand_busy() -> None:
    bpm = 150.0
    notes = [
        _note(1.0, 1, 0, 0, 0, bpm),
        _note(1.06, 2, 0, 1, 1, bpm),  # down at y=0: ineligible itself, blocks L stack
        _note(2.0, 1, 0, 0, 1, bpm),
    ]
    env = np.ones(400, dtype=np.float32)
    out, stacks, dots, _ = add_dot_stacks(
        notes, bpm=bpm, onset_env=env, onset_hop_s=HOP_S,
        chance=1.0, rng=random.Random(0), mask_cfg=PlayabilityConfig(),
    )
    assert stacks == 0 and dots == 0
    assert out == notes


def test_onset_required() -> None:
    bpm = 150.0
    notes = [_note(1.0, 1, 0, 0, 0, bpm), _note(2.0, 1, 0, 0, 1, bpm)]
    env = np.ones(400, dtype=np.float32)
    k0 = round(1.03 / HOP_S)
    k1 = round(1.13 / HOP_S)
    env[k0 : k1 + 1] = 0.0
    out, stacks, dots, _ = add_dot_stacks(
        notes, bpm=bpm, onset_env=env, onset_hop_s=HOP_S,
        chance=1.0, rng=random.Random(0), mask_cfg=PlayabilityConfig(),
    )
    assert stacks == 0 and dots == 0
    assert out == notes


def test_no_dot_past_audio_end() -> None:
    bpm = 150.0
    notes = [_note(1.0, 1, 0, 0, 0, bpm)]
    # Env ends right at t=1.02 s, before the first dot at 1.05 s
    n_frames = int(1.02 / HOP_S)
    env = np.ones(n_frames, dtype=np.float32)
    out, stacks, dots, _ = add_dot_stacks(
        notes, bpm=bpm, onset_env=env, onset_hop_s=HOP_S,
        chance=1.0, rng=random.Random(0), mask_cfg=PlayabilityConfig(),
    )
    assert stacks == 0 and dots == 0
    assert out == notes


def test_chance_zero_identity() -> None:
    bpm = 150.0
    notes = [_note(1.0, 1, 0, 0, 0, bpm), _note(2.0, 1, 0, 0, 1, bpm)]
    env = np.ones(400, dtype=np.float32)
    out, stacks, dots, reverted = add_dot_stacks(
        notes, bpm=bpm, onset_env=env, onset_hop_s=HOP_S,
        chance=0.0, rng=random.Random(0), mask_cfg=PlayabilityConfig(),
    )
    assert out == notes
    assert (stacks, dots, reverted) == (0, 0, 0)


def test_stacks_pass_validation() -> None:
    bpm = 150.0
    notes: list[Note] = []
    t = 1.0
    for i in range(200):
        hand = i % 2
        x = 1 if hand == 0 else 2
        d = (i // 2) % 2  # alternating 0 (up) and 1 (down) per hand
        y = 0 if d == 0 else 2
        notes.append(_note(t, x, y, hand, d, bpm))
        t += 0.25
    env = np.ones(int((t + 2.0) / HOP_S), dtype=np.float32)
    cfg = PlayabilityConfig()
    out, stacks, dots, _ = add_dot_stacks(
        notes, bpm=bpm, onset_env=env, onset_hop_s=HOP_S,
        chance=1.0, rng=random.Random(42), mask_cfg=cfg,
    )
    assert stacks > 0 and dots > 0
    assert final_flow_ok(notes, out, cfg)
    assert FlowReport(level=1).level == 1


def test_failing_stack_reverted(monkeypatch) -> None:
    bpm = 150.0
    notes = [_note(1.0, 1, 0, 0, 0, bpm), _note(2.0, 1, 0, 0, 1, bpm)]
    env = np.ones(400, dtype=np.float32)
    monkeypatch.setattr("bsmaker.flow_rules.window_ok", lambda *a, **k: False)
    out, stacks, dots, reverted = add_dot_stacks(
        notes, bpm=bpm, onset_env=env, onset_hop_s=HOP_S,
        chance=1.0, rng=random.Random(0), mask_cfg=PlayabilityConfig(),
    )
    assert out == notes
    assert (stacks, dots, reverted) == (0, 0, 1)


def _alternating_run(n: int, *, start_t: float = 1.0, gap: float = 0.2, start_hand: int = 0) -> list[Note]:
    notes: list[Note] = []
    for k in range(n):
        hand = (start_hand + k) % 2
        x = 1 if hand == 0 else 2
        d = (k // 2) % 2
        notes.append(_note(start_t + k * gap, x, 0, hand, d))
    return notes


def test_find_runs_basic() -> None:
    notes = _alternating_run(10, gap=0.2)
    assert find_runs(notes) == [(0, 10)]


def test_run_breaks_on_double() -> None:
    notes = _alternating_run(12, gap=0.2)
    # Replace index 5..6 with a double at the time of index 5
    t5 = notes[5].time_s
    notes[5] = _note(t5, 1, 0, 0, 0)
    notes[6] = _note(t5, 2, 0, 1, 0)
    runs = find_runs(notes)
    assert all(not (i <= 5 < j) and not (i <= 6 < j) for i, j in runs)
    assert runs == []


def test_run_gap_limits() -> None:
    # A 0.35 s gap splits 16 notes into two 8-note runs
    first = _alternating_run(8, start_t=1.0, gap=0.2, start_hand=0)
    second = _alternating_run(8, start_t=first[-1].time_s + 0.35, gap=0.2, start_hand=0)
    assert find_runs(first + second) == [(0, 8), (8, 16)]

    # A 0.08 s gap also splits
    second_fast = _alternating_run(8, start_t=first[-1].time_s + 0.08, gap=0.2, start_hand=0)
    assert find_runs(first + second_fast) == [(0, 8), (8, 16)]


_UPS = (0, 4, 5)
_SWEEP_CELLS = {
    0: {(1, 0, 1), (1, 0, 7), (0, 1, 4), (1, 1, 0), (1, 2, 0)},
    1: {(2, 0, 1), (2, 0, 6), (3, 1, 5), (2, 1, 0), (2, 2, 0)},
}


@pytest.mark.parametrize("seed", range(6))
@pytest.mark.parametrize("gap", [0.10, 0.15])
def test_slither_is_reference_sweep(seed: int, gap: float) -> None:
    from bsmaker.playability.geometry import angle_diff, direction_angle

    notes = _alternating_run(16, gap=gap, start_hand=0)
    out = rewrite_run(notes, 0, len(notes), SLITHER, random.Random(seed))
    assert len(out) == len(notes)
    assert [n.time_s for n in out] == [n.time_s for n in notes]
    last: dict[int, Note] = {}
    for n in out:
        # Each hand stays on its anchor pair; up cuts never sit on the bottom row.
        assert (n.x, n.y, n.direction) in _SWEEP_CELLS[n.color]
        if n.direction in _UPS:
            assert n.y >= 1
        p = last.get(n.color)
        if p is not None:
            # alternating parity, every turn within 45 deg of a clean flip
            assert (p.direction in _UPS) != (n.direction in _UPS)
            expected = (direction_angle(p.direction) + 180.0) % 360.0
            assert angle_diff(direction_angle(n.direction), expected) <= 45.0
            assert max(abs(n.x - p.x), abs(n.y - p.y)) <= 2
        last[n.color] = n


def test_slither_mixes_sweeps_and_verticals() -> None:
    notes = _alternating_run(16, gap=0.15, start_hand=0)
    out = rewrite_run(notes, 0, len(notes), SLITHER, random.Random(0))
    dirs = {n.direction for n in out}
    assert {4, 5} <= dirs  # outward-up diagonals (UL left, UR right)
    assert 0 in dirs  # straight ups


def test_slither_full_height_only_with_long_gap() -> None:
    # Same-hand gap 0.30 s (>= 0.25): vertical ups go to the top row.
    slow = rewrite_run(_alternating_run(16, gap=0.15), 0, 16, SLITHER, random.Random(1))
    assert {n.y for n in slow if n.direction == 0} == {2}
    # Same-hand gap 0.20 s (< 0.25): vertical ups only one row up.
    fast = rewrite_run(_alternating_run(16, gap=0.10), 0, 16, SLITHER, random.Random(1))
    assert {n.y for n in fast if n.direction == 0} == {1}


def test_slither_parity_with_previous() -> None:
    # Previous left note before run cuts up (0) -> first left note in run must cut down (1)
    prev = [
        _note(0.2, 1, 0, 0, 0),
        _note(0.4, 2, 0, 1, 1),
    ]
    run = _alternating_run(8, start_t=1.0, gap=0.2, start_hand=0)
    notes = prev + run
    out = rewrite_run(notes, 2, len(notes), SLITHER, random.Random(0))
    assert out[:2] == prev
    assert out[2].color == 0 and out[2].direction == 1
    assert out[3].color == 1 and out[3].direction == 0


def test_rotation_cycle() -> None:
    notes = _alternating_run(8, gap=0.2, start_hand=0)
    out = rewrite_run(notes, 0, 8, ROTATION, random.Random(0))
    expected = [
        (0, 1, 4), (1, 0, 6), (2, 0, 7), (3, 1, 5),
        (0, 1, 4), (1, 0, 6), (2, 0, 7), (3, 1, 5),
    ]
    assert [(n.x, n.y, n.direction) for n in out] == expected


def test_rotation_falls_back() -> None:
    from bsmaker.formats.tokens import encode_token

    notes = _alternating_run(10, gap=0.2, start_hand=0)
    # Block R(1, 0, 6) (rotation only; the sweep never uses it) as a rare token so ROTATION
    # fails window_ok and falls back to SLITHER
    bad_rot = encode_token(1, 0, 6)
    cfg = PlayabilityConfig(rare_tokens=(frozenset(), frozenset({bad_rot})))
    # Pick a seed that selects ROTATION first (rng.random() < 0.3)
    seed = next(s for s in range(100) if random.Random(s).random() < 0.3)
    # Use a two-step rng check: share=1.0 draws once (< 1.0), then template draws (< 0.3)
    def picks_rotation(s: int) -> bool:
        r = random.Random(s)
        r.random()
        return r.random() < 0.3

    seed = next(s for s in range(100) if picks_rotation(s))
    out, found, rewritten, n_rewritten, reverted = apply_snakes(
        notes, share=1.0, rng=random.Random(seed), mask_cfg=cfg,
    )
    assert found == 1 and rewritten == 1 and n_rewritten == 10 and reverted == 0
    assert all((n.x, n.y, n.direction) in _SWEEP_CELLS[n.color] for n in out)
    assert final_flow_ok(notes, out, cfg)


def test_share_zero_identity() -> None:
    notes = _alternating_run(10, gap=0.2)
    out, found, rewritten, n_rewritten, reverted = apply_snakes(
        notes, share=0.0, rng=random.Random(0), mask_cfg=PlayabilityConfig(),
    )
    assert out == notes
    assert found == 1 and (rewritten, n_rewritten, reverted) == (0, 0, 0)


def test_seed_deterministic() -> None:
    notes = _alternating_run(16, gap=0.2)
    cfg = PlayabilityConfig()
    out1 = apply_snakes(notes, share=0.8, rng=random.Random(99), mask_cfg=cfg)
    out2 = apply_snakes(notes, share=0.8, rng=random.Random(99), mask_cfg=cfg)
    assert out1 == out2


def test_apply_flow_level0_identity() -> None:
    bpm = 150.0
    notes = _alternating_run(12, gap=0.2)
    env = np.ones(400, dtype=np.float32)
    out, rep = apply_flow(
        notes, bpm=bpm, onset_env=env, onset_hop_s=HOP_S,
        level=0, seed=0, mask_cfg=PlayabilityConfig(),
    )
    assert out == notes
    assert rep == FlowReport(level=0)


def test_auto_flow_level() -> None:
    assert auto_flow_level(["tech"]) == 2
    assert auto_flow_level(["dance"]) == 1
    assert auto_flow_level([]) == 1
    assert auto_flow_level(["dance", "tech"]) == 2





def _stacks(notes, env=None, bpm=150.0, seed=0, cfg=None):
    env = np.ones(400, dtype=np.float32) if env is None else env
    return add_dot_stacks(
        notes, bpm=bpm, onset_env=env, onset_hop_s=HOP_S,
        chance=1.0, rng=random.Random(seed), mask_cfg=cfg or PlayabilityConfig(),
    )


def test_stack_when_next_note_at_0_2s() -> None:
    # 1 dot: next same-hand note >= max(0.20, 0.05 + 0.13) = 0.20 s after the arrow
    notes = [_note(1.0, 1, 0, 0, 0), _note(1.2, 1, 0, 0, 1)]
    out, stacks, dots, reverted = _stacks(notes)
    assert (stacks, dots, reverted) == (1, 1, 0)
    assert [(n.x, n.y) for n in out if n.direction == 8] == [(1, 1)]


def test_second_dot_needs_span_plus_0_13() -> None:
    # 2 dots (span 0.10 s) need the next note >= 0.23 s later; at 0.22 s only 1 dot fits
    for seed in range(20):
        _, _, dots, _ = _stacks([_note(1.0, 1, 0, 0, 0), _note(1.22, 2, 0, 0, 1)], seed=seed)
        assert dots == 1
    assert any(
        _stacks([_note(1.0, 1, 0, 0, 0), _note(1.24, 2, 0, 0, 1)], seed=seed)[2] == 2
        for seed in range(20)
    )


def test_max_stack_span_0_11() -> None:
    # 135 BPM: spacing 1/8 beat = 0.0556 s, two dots would span 0.111 s > 0.11 s
    bpm = 135.0
    assert stack_spacing_s(bpm) == pytest.approx(60.0 / bpm / 8.0)
    for seed in range(20):
        _, stacks, dots, _ = _stacks(
            [_note(1.0, 1, 0, 0, 0, bpm), _note(2.0, 1, 0, 0, 1, bpm)], bpm=bpm, seed=seed)
        assert stacks == 1 and dots == 1


def test_stack_allowed_with_double_at_arrow_time() -> None:
    notes = [
        _note(1.0, 1, 0, 0, 0),
        _note(1.0, 2, 0, 1, 1),  # other hand at the arrow's own time (a double)
        _note(2.0, 1, 0, 0, 1),
        _note(2.0, 2, 2, 1, 0),
    ]
    out, stacks, _, _ = _stacks(notes)
    assert stacks == 1
    assert any(n.color == 0 and n.direction == 8 and 1.0 < n.time_s < 1.2 for n in out)


def test_stack_blocked_by_other_hand_after_double_window() -> None:
    notes = [_note(1.0, 1, 0, 0, 0), _note(1.02, 2, 0, 1, 1), _note(2.0, 1, 0, 0, 1)]
    _, stacks, dots, _ = _stacks(notes)
    assert stacks == 0 and dots == 0


def test_onset_p50_is_enough() -> None:
    # 65% of frames at 0.5 and 35% at 1.0: p50 = 0.5 (passes), p70 = 1.0 (would fail)
    env = np.full(400, 0.5, dtype=np.float32)
    env[200:340] = 1.0
    notes = [_note(1.0, 1, 0, 0, 0), _note(2.0, 1, 0, 0, 1)]
    _, stacks, _, _ = _stacks(notes, env=env)
    assert stacks == 1


def test_stack_dot_must_not_block_other_hand_vision() -> None:
    # A dot at (1, 1) is in the vision zone; the right hand's next note in column 1 comes
    # 0.06 s after it (outside the other-hand window, inside vision_gap_s 0.12).
    blocked = [_note(1.0, 1, 0, 0, 0), _note(1.11, 1, 0, 1, 1), _note(2.0, 1, 0, 0, 1)]
    _, stacks, _, _ = _stacks(blocked)
    assert stacks == 0
    free = [_note(1.0, 1, 0, 0, 0), _note(1.11, 3, 0, 1, 1), _note(2.0, 1, 0, 0, 1)]
    _, stacks, _, _ = _stacks(free)
    assert stacks == 1


def _repair(notes, cfg=None):
    return repair_loose_dots(notes, mask_cfg=cfg or PlayabilityConfig())


def test_repair_loose_dot_gets_parity_direction() -> None:
    notes = [_note(1.0, 1, 0, 0, 0), _note(1.5, 2, 2, 0, 8)]  # up, then a loose dot
    out, repaired, reverted = _repair(notes)
    assert (repaired, reverted) == (1, 0)
    assert out[1].direction == 1 and (out[1].x, out[1].y, out[1].time_s) == (2, 2, 1.5)


def test_repair_prefers_up_down_within_45() -> None:
    # previous swing up-left (135) -> expected down-right (315); down (270) is within 45
    notes = [_note(1.0, 2, 0, 0, 4), _note(1.5, 1, 2, 0, 8)]
    out, repaired, _ = _repair(notes)
    assert repaired == 1 and out[1].direction == 1


def test_repair_keeps_extension_dots() -> None:
    notes = [_note(1.0, 1, 0, 0, 0), _note(1.05, 1, 1, 0, 8), _note(1.1, 1, 2, 0, 8)]
    out, repaired, reverted = _repair(notes)
    assert out == notes and (repaired, reverted) == (0, 0)


def test_repair_keeps_dot_without_previous_swing() -> None:
    notes = [_note(1.0, 1, 0, 0, 8), _note(1.5, 1, 2, 0, 1)]
    out, repaired, reverted = _repair(notes)
    assert out == notes and (repaired, reverted) == (0, 0)


def test_repair_reverted_when_window_fails(monkeypatch) -> None:
    monkeypatch.setattr("bsmaker.flow_rules.window_ok", lambda *a, **k: False)
    notes = [_note(1.0, 1, 0, 0, 0), _note(1.5, 2, 2, 0, 8)]
    out, repaired, reverted = _repair(notes)
    assert out == notes and repaired == 0 and reverted == 1


def test_repair_window_follows_dot_parity_chain() -> None:
    # The dot at 1.0 is loose. "Down" fits its own parity, but parity carries through the
    # dot at 1.75 to the arrow at 2.05 (outside a plain +-1 s window), which would then
    # break the strict 45 deg limit. The exact parity direction (down-right) must win.
    notes = [
        _note(0.6, 1, 0, 0, 4),   # up-left
        _note(1.0, 1, 2, 0, 8),   # loose dot
        _note(1.75, 1, 1, 0, 8),  # dot (loose too)
        _note(2.05, 1, 0, 0, 3),  # right: 45 deg from the base expectation (315)
    ]
    cfg = PlayabilityConfig()
    out, _, _ = _repair(notes, cfg)
    assert out[1].direction == 7
    assert final_flow_ok(notes, out, cfg)


def test_final_flow_ok_checks_mask_on_non_added_notes() -> None:
    cfg = PlayabilityConfig()
    pre = [_note(1.0, 1, 0, 0, 0), _note(1.12, 1, 1, 0, 8)]
    assert final_flow_ok(pre, pre, cfg)
    # Moving the dot two columns breaks hand_speed under the mask profile, although the
    # 0.12 s slider-tail profile drops it as a tail and would not see it.
    moved = [pre[0], _note(1.12, 3, 1, 0, 8)]
    assert not final_flow_ok(pre, moved, cfg)


def test_final_flow_ok_no_new_violations_with_added_dots() -> None:
    cfg = PlayabilityConfig()
    pre = [_note(1.0, 0, 0, 0, 0), _note(2.0, 0, 2, 0, 1)]
    ok = pre[:1] + [_note(1.05, 0, 1, 0, 8)] + pre[1:]
    assert final_flow_ok(pre, ok, cfg)
    # An added dot 0.13 s after the arrow is no tail; 3 cells away it breaks hand_speed.
    bad = pre[:1] + [_note(1.13, 3, 0, 0, 8)] + pre[1:]
    assert not final_flow_ok(pre, bad, cfg)


def test_apply_flow_repairs_before_stacks() -> None:
    bpm = 150.0
    notes = [_note(1.0, 1, 2, 0, 1, bpm), _note(1.5, 1, 0, 0, 8, bpm), _note(2.5, 1, 2, 0, 1, bpm)]
    env = np.ones(400, dtype=np.float32)
    cfg = PlayabilityConfig()
    out, rep = apply_flow(
        notes, bpm=bpm, onset_env=env, onset_hop_s=HOP_S, level=3, seed=0, mask_cfg=cfg,
    )
    assert rep.dots_repaired == 1
    assert [n.direction for n in out if n.time_s == 1.5] == [0]
    assert final_flow_ok(notes, out, cfg)


def test_no_stack_inside_weave_run() -> None:
    # 10 alternating notes 0.2 s apart, each hand flipping up/down: a dot inside would break
    # the weave, so only the run's last note may get a stack.
    notes = []
    for k in range(10):
        hand = k % 2
        d = (k // 2 + hand) % 2  # L: 0,1,0,...  R: 1,0,1,...
        notes.append(_note(1.0 + 0.2 * k, 1 + hand, 1, hand, d))
    out, stacks, dots, _ = _stacks(notes)
    assert stacks == 1
    assert [n.time_s > 2.8 for n in out if n.direction == 8] == [True] * dots


def test_stack_allowed_on_short_alternation() -> None:
    notes = [_note(1.0, 1, 1, 0, 0), _note(1.2, 2, 1, 1, 1), _note(1.4, 1, 1, 0, 1)]
    _, stacks, _, _ = _stacks(notes)
    assert stacks >= 2
