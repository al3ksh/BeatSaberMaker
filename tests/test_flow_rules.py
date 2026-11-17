"""Tests for the rule-based flow layer (dot stacks and snake runs)."""
from __future__ import annotations

import math
import random

import numpy as np
import pytest

from bsmaker.audio.beats import ENV_HOP, onset_envelope
from bsmaker.audio.decode import SAMPLE_RATE
from bsmaker.flow_rules import (
    FlowReport,
    add_dot_stacks,
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

