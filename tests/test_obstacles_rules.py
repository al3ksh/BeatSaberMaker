import random

from bsmaker.formats.model import Note
from bsmaker.obstacles_rules import gameplay_obstacles
from bsmaker.playability.layers import wall_ok

BPM = 120.0  # beat = 0.5 s


def note(t, x, y=1, color=0):
    return Note(t, t * BPM / 60, x, y, color, 8)


def side_notes(duration, x_choices=(2, 3), step=0.5):
    out, t = [], 1.0
    while t < duration:
        out.append(note(t, x_choices[len(out) % len(x_choices)]))
        t += step
    return out


def test_level_gating():
    notes = side_notes(120)
    for level in (0, 1):
        assert gameplay_obstacles(notes, BPM, 120, level, 1) == []
    assert gameplay_obstacles(notes, BPM, 120, 2, 1) != []


def test_spacing_and_margins():
    notes = side_notes(300)
    for level, spacing in ((2, 45.0), (3, 30.0)):
        obs = gameplay_obstacles(notes, BPM, 300, level, 3)
        assert 1 <= len(obs) <= int(300 // spacing)
        for o in obs:
            assert o.time_s >= 10.0 and o.time_s + o.duration_s <= 295.0
        for a, b in zip(obs, obs[1:]):
            assert b.time_s - a.time_s >= spacing


def test_side_step_only_with_notes_on_other_side():
    notes = side_notes(120, x_choices=(2, 3))  # everything on the right
    obs = gameplay_obstacles(notes, BPM, 120, 2, 5)
    assert obs
    for o in obs:
        assert (o.x, o.w, o.y, o.h) == (0, 2, 0, 5)
        assert 1 <= o.duration_beats <= 2 and abs(o.duration_s - o.duration_beats * 0.5) < 1e-9
        assert abs(o.beat - o.time_s * 2) < 1e-9


def test_side_step_needs_two_notes_and_clear_window():
    # one single note on the right: no side-step; a note on the left inside the window blocks it
    assert all(o.h == 3 for o in gameplay_obstacles([note(30.0, 3)], BPM, 120, 2, 1))
    mixed = side_notes(120, x_choices=(0, 3))
    assert all(o.h == 3 for o in gameplay_obstacles(mixed, BPM, 120, 2, 1))


def test_duck_only_in_gaps_and_centred():
    # dense notes everywhere on both sides except a gap of 3 s between 40 and 43
    notes = [n for n in side_notes(120, x_choices=(0, 3), step=0.5) if not 39.9 < n.time_s < 43.1]
    obs = gameplay_obstacles(notes, BPM, 120, 2, 1)
    assert len(obs) == 1
    o = obs[0]
    assert (o.x, o.w, o.y, o.h) == (0, 4, 2, 3) and o.duration_beats == 1.0
    before = max(n.time_s for n in notes if n.time_s < 41)
    after = min(n.time_s for n in notes if n.time_s > 41)
    assert abs((o.time_s + o.duration_s / 2) - (before + after) / 2) < 1e-9


def test_sides_alternate():
    # notes alternate every 60 s between pure-right and pure-left stretches
    notes = []
    for block in range(7):
        xs = (2, 3) if block % 2 == 0 else (0, 1)
        t = block * 60 + 1.0
        while t < block * 60 + 59:
            notes.append(note(t, xs[len(notes) % 2]))
            t += 0.5
    obs = [o for o in gameplay_obstacles(notes, BPM, 420, 2, 7) if o.h == 5]
    assert len(obs) >= 3
    for a, b in zip(obs, obs[1:]):
        assert a.x != b.x


def test_no_candidates_and_short_songs():
    assert all(o.h == 3 for o in gameplay_obstacles([], BPM, 120, 3, 1))  # empty song: ducks only
    full = [note(1.0 + 0.5 * i, x) for i in range(240) for x in (0, 1, 2, 3)]  # no side, no gap
    assert gameplay_obstacles(full, BPM, 120, 3, 1) == []
    short = gameplay_obstacles(side_notes(31), BPM, 31, 3, 1)
    assert len(short) <= 1 and all(10.0 <= o.time_s and o.time_s + o.duration_s <= 26.0 for o in short)
    assert gameplay_obstacles(side_notes(31), BPM, 31, 2, 1) == []
    assert gameplay_obstacles([], 0, 120, 2, 1) == []


def test_deterministic():
    notes = side_notes(300)
    a = gameplay_obstacles(notes, BPM, 300, 3, 11)
    assert a == gameplay_obstacles(notes, BPM, 300, 3, 11)


def test_all_results_pass_wall_ok_property():
    for seed in range(40):
        rng = random.Random(seed)
        n = rng.choice((0, 5, 50, 400))
        notes = sorted((note(rng.uniform(0, 200), rng.randrange(4), rng.randrange(3), rng.randrange(2))
                        for _ in range(n)), key=lambda x: x.time_s)
        bpm = rng.choice((90.0, 128.0, 180.0))
        for level in (2, 3):
            for o in gameplay_obstacles(notes, bpm, 200, level, seed):
                assert wall_ok(o, notes)
