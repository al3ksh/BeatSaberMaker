import numpy as np

from bsmaker.formats.events import FRAMES_PER_SECOND
from bsmaker.formats.model import Note
from bsmaker.lights_rules import rule_lights

BPM = 120.0
TYPES = {0, 1, 2, 3, 4, 8, 9, 12, 13}
LIGHT_VALUES = {0, 1, 2, 3, 5, 6, 7}


def make_mel(duration_s, seed=0):
    n = int(duration_s * FRAMES_PER_SECOND)
    rng = np.random.default_rng(seed)
    mel = rng.normal(size=(n, 80)).astype(np.float16)
    mel[:: int(FRAMES_PER_SECOND // 2)] += 4  # onsets about every half second
    return mel


def make_notes(duration_s):
    out, t = [], 2.0
    while t < duration_s - 1:
        out.append(Note(t, t * 2, 1, 1, len(out) % 2, 8))
        t += 0.5
    return out


def run(level, duration=60.0, seed=1, notes=True, offset=0.3):
    mel = make_mel(duration)
    return rule_lights(mel, make_notes(duration) if notes else [], BPM, offset, duration, level, seed)


def test_level_zero_empty():
    assert run(0) == []


def test_counts_grow_with_level():
    n1, n2, n3 = (len(run(lv)) for lv in (1, 2, 3))
    assert 0 < n1 < n2 < n3


def test_only_supported_types_and_values():
    for lv in (1, 2, 3):
        for e in run(lv):
            assert e.type in TYPES and e.float_value == 1.0
            if e.type in (0, 1, 2, 3, 4):
                assert e.value in LIGHT_VALUES
            elif e.type in (8, 9):
                assert e.value == 0
            else:
                assert 1 <= e.value <= 7
            assert 0.0 <= e.time_s <= 60.0 and abs(e.beat - e.time_s * 2) < 1e-9


def test_sorted_and_unique():
    for lv in (1, 2, 3):
        ev = run(lv)
        keys = [(e.time_s, e.type) for e in ev]
        assert keys == sorted(keys) and len(keys) == len(set(keys))


def test_structure():
    ev = run(2, offset=0.0)
    back = [e for e in ev if e.type == 0 and e.value != 0]
    assert all(abs(e.time_s / 2.0 - round(e.time_s / 2.0)) < 1e-9 for e in back)  # bar = 2 s
    assert {e.time_s for e in ev if e.type == 9} == {0.0, 16.0, 32.0, 48.0}  # zoom every 8 bars
    assert any(e.type == 4 for e in ev)
    # colour swaps every 8 beats (4 s): ring fades at beat 0 and beat 8 use different colours
    ring = {e.time_s: e.value for e in ev if e.type == 1}
    assert ring[0.0] != ring[4.0] and ring[0.0] == ring[2.0]
    # an off follows a lone event one beat later
    assert any(e.type == 2 and e.value == 0 for e in run(1))


def test_works_without_notes_and_short_song():
    ev = run(3, notes=False)
    assert ev and all(e.type not in (2, 3) or e.value == 0 for e in ev)
    short = run(3, duration=31.0)
    assert short and max(e.time_s for e in short) <= 31.0
    assert rule_lights(np.zeros((0, 80), np.float16), [], BPM, 0.0, 31.0, 2, 1) != []
    assert rule_lights(make_mel(31), [], BPM, 0.0, 0.0, 2, 1) == []


def test_deterministic():
    assert run(3, seed=5) == run(3, seed=5)


def test_level_one_is_bars_only():
    ev = run(1, offset=0.3)
    bar_s = 2.0
    for e in ev:
        if e.value != 0 and e.type in (0, 1, 2, 3, 8, 12, 13):
            assert abs((e.time_s - 0.3) / bar_s - round((e.time_s - 0.3) / bar_s)) < 1e-9
    assert not any(e.type == 4 for e in ev)
