"""Tests for the rule-based flow layer (dot stacks and snake runs)."""
from __future__ import annotations

import math
import random

import numpy as np
import pytest

from bsmaker.audio.beats import ENV_HOP, onset_envelope
from bsmaker.audio.decode import SAMPLE_RATE
from bsmaker.flow_rules import (
    ROTATION,
    SLITHER,
    FlowReport,
    add_dot_stacks,
    apply_flow,
    apply_snakes,
    auto_flow_level,
    find_runs,
    rewrite_run,
    stack_spacing_s,
    validation_cfg,
    window_ok,
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
    assert stack_spacing_s(150.0) == pytest.approx(0.05)
    assert stack_spacing_s(220.0) == pytest.approx(60.0 / 220.0 / 4.0)
    assert stack_spacing_s(60.0) is None


def test_spacing_half_time_bpm() -> None:
    assert stack_spacing_s(110.0) == pytest.approx(60.0 / 110.0 / 8.0)


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
    notes = [_note(1.0, 1, 0, 0, 0, bpm), _note(1.3, 1, 0, 0, 1, bpm)]
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
    assert window_ok(out, 0.0, math.inf, validation_cfg(cfg))
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


def test_slither_shape() -> None:
    notes = _alternating_run(16, gap=0.2, start_hand=0)
    out = rewrite_run(notes, 0, len(notes), SLITHER, random.Random(3))
    assert len(out) == len(notes)
    last_dir: dict[int, int] = {}
    last_x: dict[int, int] = {}
    prev_left_x: int | None = None
    for n in out:
        assert n.y == 0
        assert n.direction in (0, 1)
        if n.color in last_dir:
            assert n.direction != last_dir[n.color]
        last_dir[n.color] = n.direction
        if n.color == 0:
            assert 0 <= n.x <= 2
            prev_left_x = n.x
        else:
            assert 1 <= n.x <= 3
            if prev_left_x is not None:
                assert n.x >= prev_left_x
        if n.color in last_x:
            assert abs(n.x - last_x[n.color]) <= 1
        last_x[n.color] = n.x


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
    # Block L(0, 1, 4) as a rare token so ROTATION fails window_ok and falls back to SLITHER
    bad_rot = encode_token(0, 1, 4)
    cfg = PlayabilityConfig(rare_tokens=(frozenset({bad_rot}), frozenset()))
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
    assert all(n.y == 0 and n.direction in (0, 1) for n in out)
    assert window_ok(out, 0.0, math.inf, validation_cfg(cfg))


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



