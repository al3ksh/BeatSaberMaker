import json
import math
import shutil
from pathlib import Path

from bsmaker.formats.reader import read_map
from bsmaker.formats.tempo import TempoMap, read_tempo

FIX = Path(__file__).resolve().parents[1] / "fixtures" / "maps"


def test_constant_map():
    assert TempoMap(120.0, ()).to_seconds(8) == 8 * 60 / 120


def test_two_points():
    t = TempoMap(120.0, ((0.0, 120.0), (8.0, 60.0)))
    assert t.to_seconds(8) == 4.0
    assert t.to_seconds(10) == 6.0


def test_first_point_later_uses_info_bpm_before():
    t = TempoMap(120.0, ((4.0, 60.0),))
    assert t.to_seconds(2) == 1.0
    assert t.to_seconds(4) == 2.0
    assert t.to_seconds(6) == 4.0


def test_unsorted_and_duplicate_points():
    t = TempoMap(120.0, ((8.0, 60.0), (0.0, 120.0), (8.0, 30.0)))
    assert t.to_seconds(8) == 4.0
    assert t.to_seconds(9) == 6.0  # the last duplicate (30 bpm) wins


def test_bpm_range():
    assert TempoMap(120.0, ((0.0, 90.0), (4.0, 200.0))).bpm_range() == (90.0, 200.0)


def test_read_tempo_none_and_v3_single_equal(tmp_path):
    assert read_tempo({"colorNotes": []}, "v3", 120.0, tmp_path)[0] == "none"
    d = {"colorNotes": [], "bpmEvents": [{"b": 0, "m": 120.005}]}
    assert read_tempo(d, "v3", 120.0, tmp_path)[0] == "none"


def test_read_tempo_v3_events(tmp_path):
    d = {"colorNotes": [], "bpmEvents": [{"b": 0, "m": 120}, {"b": 8, "m": 60}]}
    kind, t = read_tempo(d, "v3", 120.0, tmp_path)
    assert kind == "events" and t.to_seconds(10) == 6.0


def test_read_tempo_v2_editor(tmp_path):
    d = {"_notes": [], "_customData": {"_BPMChanges": [{"_time": 4, "_BPM": 90}, {"_time": 8, "_bpm": 150}]}}
    kind, t = read_tempo(d, "v2", 120.0, tmp_path)
    assert kind == "editor" and t.bpm_range() == (90.0, 150.0)


def test_read_tempo_v2_events_beat_editor(tmp_path):
    d = {"_notes": [], "_events": [{"_time": 8, "_type": 100, "_floatValue": 60}],
         "_customData": {"_BPMChanges": [{"_time": 4, "_BPM": 90}]}}
    kind, t = read_tempo(d, "v2", 120.0, tmp_path)
    assert kind == "events" and t.points == ((8.0, 60.0),)


def test_read_tempo_external(tmp_path):
    (tmp_path / "BPMInfo.dat").write_text(json.dumps({"_regions": [{}, {}]}))
    assert read_tempo({"colorNotes": []}, "v3", 120.0, tmp_path)[0] == "external"
    d = {"colorNotes": [], "bpmEvents": [{"b": 0, "m": 120}, {"b": 8, "m": 60}]}
    assert read_tempo(d, "v3", 120.0, tmp_path)[0] == "events"


def test_invalid_values_become_nan(tmp_path):
    for bad in (0, "abc", -5):
        d = {"_notes": [], "_events": [{"_time": 4, "_type": 100, "_floatValue": bad}]}
        kind, t = read_tempo(d, "v2", 120.0, tmp_path)
        lo, hi = t.bpm_range()
        assert kind == "events" and math.isnan(lo) and math.isnan(hi)


def _v3_folder(tmp_path, diff_extra, name="tempo_map"):
    dst = tmp_path / name
    shutil.copytree(FIX / "v2_basic", dst)
    info = json.loads((dst / "Info.dat").read_text(encoding="utf-8-sig"))
    notes = [{"b": b, "x": 1, "y": 0, "c": 0, "d": 1} for b in (4, 12)]
    (dst / "ExpertStandard.dat").write_text(json.dumps({"version": "3.0.0", "colorNotes": notes, **diff_extra}))
    (dst / "Info.dat").write_text(json.dumps(info))
    return dst


def test_read_map_v3_follows_tempo_map(tmp_path):
    m = read_map(_v3_folder(tmp_path, {"bpmEvents": [{"b": 0, "m": 120}, {"b": 8, "m": 60}]}))
    d = m.difficulties[-1]
    assert d.flags.tempo_kind == "events" and d.flags.has_bpm_changes
    assert (d.flags.bpm_min, d.flags.bpm_max) == (60.0, 120.0)
    assert [n.time_s for n in d.notes] == [2.0, 8.0]
    assert [n.beat for n in d.notes] == [4, 12]


def test_read_map_v3_nan_falls_back_to_constant(tmp_path):
    m = read_map(_v3_folder(tmp_path, {"bpmEvents": [{"b": 0, "m": 120}, {"b": 8, "m": 0}]}))
    d = m.difficulties[-1]
    assert [n.time_s for n in d.notes] == [2.0, 6.0]
    assert math.isnan(d.flags.bpm_min)


def test_read_map_v2_editor_keeps_constant_times(tmp_path):
    dst = tmp_path / "ed"
    shutil.copytree(FIX / "v2_basic", dst)
    p = dst / "ExpertStandard.dat"
    data = json.loads(p.read_text(encoding="utf-8-sig"))
    data["_customData"] = {"_BPMChanges": [{"_time": 4, "_BPM": 60}]}
    p.write_text(json.dumps(data))
    d = read_map(dst).difficulties[-1]
    assert d.flags.tempo_kind == "editor" and d.flags.has_bpm_changes
    assert d.notes[0].time_s == d.notes[0].beat * 60 / 120
