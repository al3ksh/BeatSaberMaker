from dataclasses import replace
from pathlib import Path

from bsmaker.data.filters import nps, reject_reasons
from bsmaker.formats.model import BeatMap, Difficulty, DifficultyFlags, Note


def _notes(n=100, span=100.0):
    return tuple(Note(i * span / (n - 1), i, 0, 0, 1, 8) for i in range(n))


def _flags(**kw):
    base = dict(has_bpm_changes=False, has_offgrid_notes=False, has_custom_note_data=False,
                has_time_offset=False)
    base.update(kw)
    return DifficultyFlags(**base)


def _diff(notes=None, characteristic="Standard", flags=None):
    return Difficulty(characteristic, "Expert", "v3", 16.0, 0.0,
                      _notes() if notes is None else notes, 0, 0, 0, 0, flags or _flags())


def _bm(diff, requirements=(), suggestions=()):
    return BeatMap(Path("x"), "k", "t", "a", "m", 120.0, None, tuple(requirements),
                   tuple(suggestions), (diff,), ())


def _r(bm=None, diff=None, duration_s=120.0, hard_rate=0.0, automapper=None):
    diff = diff or _diff()
    return reject_reasons(bm or _bm(diff), diff, duration_s=duration_s, hard_rate=hard_rate,
                          automapper=automapper)


def test_accepts_clean_standard():
    assert _r() == []


def test_rejects_onesaber():
    assert _r(diff=_diff(characteristic="OneSaber")) == ["not_standard"]


def test_rejects_noodle_requirement():
    d = _diff()
    assert _r(bm=_bm(d, requirements=["Noodle Extensions"]), diff=d) == ["requirements"]


def test_chroma_suggestion_is_fine():
    d = _diff()
    assert _r(bm=_bm(d, suggestions=["Chroma"]), diff=d) == []


def test_rejects_offgrid():
    assert _r(diff=_diff(flags=_flags(has_offgrid_notes=True))) == ["offgrid_or_custom_notes"]
    assert _r(diff=_diff(flags=_flags(has_custom_note_data=True))) == ["offgrid_or_custom_notes"]


def _tempo(kind, lo=120.0, hi=120.0, **kw):
    return _diff(flags=_flags(has_bpm_changes=True, tempo_kind=kind, bpm_min=lo, bpm_max=hi),
                 **kw)


def test_tempo_reasons_per_kind():
    assert _r(diff=_tempo("external")) == ["tempo_external"]
    assert _r(diff=_tempo("editor")) == ["tempo_unverified"]
    assert _r(diff=_tempo("events")) == ["tempo_unverified"]
    assert _r(diff=_tempo("external", 5, 5)) == ["tempo_external"]  # external wins over trick
    assert _r(diff=_tempo("events", 5, 120)) == ["bpm_trick"]  # trick wins over unverified


def test_bpm_trick_limits():
    nan = float("nan")
    for lo, hi in [(0, 120), (nan, nan), (19.9, 120), (120, 1000.1), (120, nan)]:
        assert _r(diff=_tempo("editor", lo, hi)) == ["bpm_trick"], (lo, hi)
    assert _r(diff=_tempo("editor", 20, 1000)) == ["tempo_unverified"]


def test_tempo_verified_removes_only_tempo_unverified():
    kw = dict(duration_s=120.0, hard_rate=0.0, automapper=None, tempo_verified=True)
    d = _tempo("events")
    assert reject_reasons(_bm(d), d, **kw) == []
    d = _tempo("events", 5, 120)
    assert reject_reasons(_bm(d), d, **kw) == ["bpm_trick"]
    d = _tempo("external")
    assert reject_reasons(_bm(d), d, **kw) == ["tempo_external"]


def test_tempo_reason_keeps_position():
    d = _diff(_notes(79), flags=_flags(has_bpm_changes=True, tempo_kind="editor",
                                       bpm_min=120.0, bpm_max=120.0, has_time_offset=True))
    assert _r(diff=d) == ["tempo_unverified", "time_offset", "too_few_notes"]


def test_rejects_time_offset():
    assert _r(diff=_diff(flags=_flags(has_time_offset=True))) == ["time_offset"]


def test_rejects_79_notes_accepts_80():
    assert _r(diff=_diff(_notes(79))) == ["too_few_notes"]
    assert _r(diff=_diff(_notes(80))) == []


def test_rejects_duration_59_and_601():
    assert _r(duration_s=59.0) == ["duration"]
    assert _r(duration_s=601.0) == ["duration"]
    assert _r(duration_s=60.0) == []
    assert _r(duration_s=600.0) == []


def test_rejects_nps_out_of_range():
    assert _r(diff=_diff(_notes(100, 1000.0))) == ["nps_range"]  # 0.1 nps
    assert _r(diff=_diff(_notes(100, 5.0))) == ["nps_range"]  # 20 nps


def test_rejects_automapper_true_but_not_none():
    assert _r(automapper=True) == ["automapper"]
    assert _r(automapper=None) == []
    assert _r(automapper=False) == []


def test_rejects_hard_rate_above_002():
    assert _r(hard_rate=0.021) == ["hard_rate"]
    assert _r(hard_rate=0.02) == []


def test_no_audio_when_duration_unknown():
    assert _r(duration_s=None) == ["no_audio"]


def test_multiple_reasons_all_listed():
    d = _diff(_notes(10, 100.0), characteristic="Lawless")
    out = _r(diff=d, hard_rate=0.5, automapper=True)
    assert {"not_standard", "too_few_notes", "nps_range", "automapper", "hard_rate"} <= set(out)


def test_nps_definition():
    ns = tuple(Note(3.0 * i / 9, i, 0, 0, 1, 8) for i in range(10))
    assert nps(ns) == 10 / 3.0
    assert nps(ns[:1]) == 0.0
    assert nps(()) == 0.0
    assert nps((ns[0], replace(ns[1], time_s=0.0))) == 0.0
