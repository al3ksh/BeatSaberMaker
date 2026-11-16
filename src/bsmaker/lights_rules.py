"""Rule-based lightshow: fallback when no light model is available, and baseline for evaluation.

Light types and values use the classic v2 numbering (0 back lasers, 1 ring lights, 2 left lasers,
3 right lasers, 4 centre lights, 8 ring spin, 9 ring zoom, 12/13 laser speeds; values for types 0-4:
0 off, blue on/flash/fade = 1/2/3, red on/flash/fade = 5/6/7).
"""
from __future__ import annotations

import math
import random
from collections.abc import Sequence

import numpy as np

from bsmaker.data.align import onset_envelope
from bsmaker.formats.events import FRAMES_PER_SECOND
from bsmaker.formats.model import LightEvent, Note

BACK, RING, LEFT, RIGHT, CENTRE, SPIN, ZOOM, LEFT_SPEED, RIGHT_SPEED = 0, 1, 2, 3, 4, 8, 9, 12, 13
_ON, _FLASH, _FADE = 0, 1, 2
_VALUES = {0: (5, 6, 7), 1: (1, 2, 3)}  # colour A = red, colour B = blue
_ONSET_PERCENTILE = {1: 90.0, 2: 75.0, 3: 50.0}  # strictly above this percentile of the peaks
_RING_STEP_BEATS = {1: 4.0, 2: 1.0, 3: 0.5}
_FOLLOW_STEP_BEATS = {2: 1.0, 3: 0.5}
_SWAP_BEATS = 8
_BAR_BEATS = 4
_ZOOM_BEATS = 8 * _BAR_BEATS
_SPEED_BEATS = 4 * _BAR_BEATS


def rule_lights(mel: np.ndarray, notes: Sequence[Note], bpm: float, offset_s: float,
                duration_s: float, level: int, seed: int) -> list[LightEvent]:
    """Rule-based light events, sorted by (time, type), unique per (time, type)."""
    if level <= 0 or bpm <= 0 or duration_s <= 0:
        return []
    level = min(level, 3)
    beat_s = 60.0 / bpm
    rng = random.Random(seed)
    first_colour = rng.randrange(2)

    def colour(t: float) -> int:
        return (first_colour + math.floor((t - offset_s) / beat_s / _SWAP_BEATS)) % 2

    def beat_times(step_beats: float) -> list[float]:
        k0 = math.ceil((0.0 - offset_s) / beat_s / step_beats)
        k1 = math.floor((duration_s - offset_s) / beat_s / step_beats)
        return [offset_s + k * step_beats * beat_s for k in range(k0, k1 + 1)]

    events: dict[tuple[float, int], int] = {}  # (time, type) -> value

    def add(t: float, typ: int, value: int) -> None:
        if 0.0 <= t <= duration_s:
            events.setdefault((t, typ), value)

    def lit(t: float, kind: int) -> int:
        return _VALUES[colour(t)][kind]

    bars = beat_times(_BAR_BEATS)
    for t in bars:
        add(t, BACK, lit(t, _FLASH))
        add(t, SPIN, 0)
    for t in beat_times(_ZOOM_BEATS):
        add(t, ZOOM, 0)
    for t in beat_times(_RING_STEP_BEATS[level]):
        add(t, RING, lit(t, _FADE))

    if level > 1:  # level 1 is bars only: no onset-driven centre lights
        for t in _strong_onsets(mel, level, duration_s):
            add(t, CENTRE, lit(t, _ON))

    ordered = sorted(notes, key=lambda n: n.time_s)
    if level == 1:  # lasers light on a bar start when their hand plays in that bar
        times_by_hand = {c: [n.time_s for n in ordered if n.color == c] for c in (0, 1)}
        for t in bars:
            for c, typ in ((0, LEFT), (1, RIGHT)):
                ts = times_by_hand[c]
                if np.searchsorted(ts, t, "left") < np.searchsorted(ts, t + _BAR_BEATS * beat_s, "left"):
                    add(t, typ, lit(t, _ON))
    else:
        last_follow = {0: -math.inf, 1: -math.inf}
        step = _FOLLOW_STEP_BEATS[level] * beat_s
        for n in ordered:
            if n.color in last_follow and n.time_s - last_follow[n.color] >= step - 1e-9:
                add(n.time_s, LEFT if n.color == 0 else RIGHT, lit(n.time_s, _ON))
                last_follow[n.color] = n.time_s

    hand_times = {c: sorted(n.time_s for n in notes if n.color == c) for c in (0, 1)}
    window_s = _SPEED_BEATS * beat_s
    for t in beat_times(_SPEED_BEATS):
        for colour_id, typ in ((0, LEFT_SPEED), (1, RIGHT_SPEED)):
            ts = hand_times[colour_id]
            count = np.searchsorted(ts, t + window_s, "left") - np.searchsorted(ts, t, "left")
            add(t, typ, int(min(7, max(1, round(2.0 * count / window_s)))))

    # "off" one beat after a group's last event unless something new arrives within that beat
    by_type: dict[int, list[float]] = {}
    for (t, typ) in events:
        if typ in (BACK, RING, LEFT, RIGHT, CENTRE):
            by_type.setdefault(typ, []).append(t)
    for typ, ts in by_type.items():
        ts.sort()
        for i, t in enumerate(ts):
            nxt = ts[i + 1] if i + 1 < len(ts) else math.inf
            if nxt - t > beat_s + 1e-9:
                add(t + beat_s, typ, 0)

    return [LightEvent(t, t * bpm / 60.0, typ, value, 1.0)
            for (t, typ), value in sorted(events.items())]


def _strong_onsets(mel: np.ndarray, level: int, duration_s: float) -> list[float]:
    env = onset_envelope(mel) if np.size(mel) else np.zeros(0, dtype=np.float32)
    if env.size < 3:
        return []
    inner = np.arange(1, env.size - 1)
    peaks = inner[(env[1:-1] > env[:-2]) & (env[1:-1] >= env[2:]) & (env[1:-1] > 0)]
    if peaks.size == 0:
        return []
    threshold = np.percentile(env[peaks], _ONSET_PERCENTILE[level])
    return [float(i) / FRAMES_PER_SECOND for i in peaks[env[peaks] > threshold]
            if i / FRAMES_PER_SECOND <= duration_s]
