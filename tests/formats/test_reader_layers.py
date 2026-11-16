import json
from pathlib import Path

from bsmaker.formats.model import Arc, Bomb, Chain, LightEvent, Obstacle
from bsmaker.formats.reader import read_map


def _v2_info(env="NiceEnvironment", bpm=120):
    return {"_version": "2.0.0", "_songName": "S", "_songAuthorName": "A", "_levelAuthorName": "M",
            "_beatsPerMinute": bpm, "_songFilename": "song.egg", "_environmentName": env,
            "_difficultyBeatmapSets": [{"_beatmapCharacteristicName": "Standard",
                                        "_difficultyBeatmaps": [{"_difficulty": "Expert",
                                                                 "_beatmapFilename": "E.dat",
                                                                 "_noteJumpMovementSpeed": 16,
                                                                 "_noteJumpStartBeatOffset": 0}]}]}


def _v3_info(bpm=120):
    return _v2_info(bpm=bpm)


def _write(tmp_path: Path, info: dict, diff: dict) -> Path:
    (tmp_path / "Info.dat").write_text(json.dumps(info), encoding="utf-8")
    (tmp_path / "E.dat").write_text(json.dumps(diff), encoding="utf-8")
    return tmp_path


V2 = {"_version": "2.2.0",
      "_notes": [{"_time": 1, "_lineIndex": 1, "_lineLayer": 0, "_type": 0, "_cutDirection": 1},
                 {"_time": 3, "_lineIndex": 2, "_lineLayer": 1, "_type": 3, "_cutDirection": 0}],
      "_obstacles": [{"_time": 4, "_lineIndex": 0, "_type": 0, "_duration": 2, "_width": 1},
                     {"_time": 2, "_lineIndex": -1, "_type": 1, "_duration": 1.5, "_width": 2},
                     {"_time": 5, "_lineIndex": 1, "_type": 2, "_duration": 1, "_width": 1}],
      "_events": [{"_time": 0, "_type": 1, "_value": 3},
                  {"_time": 8, "_type": 100, "_value": 0, "_floatValue": 150},
                  {"_time": 1, "_type": 5, "_value": 7, "_floatValue": 0.5}],
      "_sliders": [{"_time": 1}]}


def test_v2_layers(tmp_path):
    m = read_map(_write(tmp_path, _v2_info(), V2), layers=True)
    assert m.environment == "NiceEnvironment"
    lay = m.difficulties[0].layers
    assert lay.bombs == (Bomb(1.5, 3, 2, 1),)
    assert lay.obstacles == (Obstacle(1.0, 2, 0.75, 1.5, -1, 2, 2, 3),
                             Obstacle(2.0, 4, 1.0, 2, 0, 0, 1, 5))
    assert lay.dropped == 1
    assert lay.arcs == () and lay.chains == () and not lay.group_lighting
    assert lay.lights == (LightEvent(0.0, 0, 1, 3, 1.0), LightEvent(0.5, 1, 5, 7, 0.5))
    d = m.difficulties[0]
    assert len(d.notes) == 1 and d.n_bombs == 1 and d.n_obstacles == 3


V3 = {"version": "3.2.0",
      "colorNotes": [{"b": 1, "x": 0, "y": 0, "c": 0, "d": 1}],
      "bombNotes": [{"b": 2, "x": 3, "y": 2}],
      "obstacles": [{"b": 4, "x": 4, "y": 0, "d": 2, "w": 1, "h": 5}],
      "sliders": [{"b": 1, "c": 1, "x": 1, "y": 1, "d": 5, "mu": 1, "tb": 3, "tx": 2, "ty": 2,
                   "tc": 4, "tmu": 1, "m": 0}],
      "burstSliders": [{"b": 5, "c": 0, "x": 2, "y": 0, "d": 1, "tb": 6, "tx": 3, "ty": 2,
                        "sc": 4, "s": 0.5}],
      "basicBeatmapEvents": [{"b": 1, "et": 4, "i": 2, "f": 0.75}],
      "lightColorEventBoxGroups": [{"b": 1}]}


def test_v3_layers(tmp_path):
    m = read_map(_write(tmp_path, _v3_info(), V3), layers=True)
    lay = m.difficulties[0].layers
    assert lay.bombs == (Bomb(1.0, 2, 3, 2),)
    assert lay.obstacles == (Obstacle(2.0, 4, 1.0, 2, 4, 0, 1, 5),)
    assert lay.arcs == (Arc(0.5, 1, 1, 1, 1, 5, 1.5, 3, 2, 2, 4),)
    assert lay.chains == (Chain(2.5, 5, 0, 2, 0, 1, 3.0, 6, 3, 2, 4, 0.5),)
    assert lay.lights == (LightEvent(0.5, 1, 4, 2, 0.75),)
    assert lay.group_lighting and lay.dropped == 0


def test_tempo_map_times(tmp_path):
    diff = dict(V3, bpmEvents=[{"b": 0, "m": 120}, {"b": 4, "m": 60}],
                obstacles=[{"b": 3, "x": 0, "y": 0, "d": 3, "w": 1, "h": 3}],
                sliders=[{"b": 3, "c": 0, "x": 0, "y": 0, "d": 1, "tb": 6, "tx": 0, "ty": 0, "tc": 1}])
    lay = read_map(_write(tmp_path, _v3_info(), diff), layers=True).difficulties[0].layers
    w = lay.obstacles[0]
    assert (w.time_s, w.duration_s) == (1.5, 0.5 + 2.0)
    assert lay.arcs[0].time_s == 1.5 and lay.arcs[0].tail_time_s == 2.0 + 2.0


def test_layers_off_by_default(tmp_path):
    m = read_map(_write(tmp_path, _v3_info(), V3))
    assert m.difficulties[0].layers is None and len(m.difficulties[0].notes) == 1
    assert m.difficulties[0].n_bombs == 1


def test_malformed_dropped(tmp_path):
    diff = dict(V3, obstacles=[{"b": "x", "x": 0, "y": 0, "d": 1, "w": 1, "h": 3},
                               {"b": 1, "x": None, "y": 0, "d": 1, "w": 1, "h": 3},
                               {"b": 1, "x": 0, "y": 0, "d": 1e308, "w": 1, "h": 3},
                               {"b": 1, "x": 0, "y": 0, "d": 1, "w": 1, "h": 3}])
    lay = read_map(_write(tmp_path, _v3_info(), diff), layers=True).difficulties[0].layers
    assert len(lay.obstacles) == 1 and lay.dropped == 3


def test_v4_has_no_layers(tmp_path):
    info = {"version": "4.0.0", "song": {"title": "S", "author": "A"}, "audio": {"bpm": 120, "songFilename": "s.ogg"},
            "environmentNames": ["BillieEnvironment"],
            "difficultyBeatmaps": [{"characteristic": "Standard", "difficulty": "Expert",
                                    "beatmapDataFilename": "E.dat", "noteJumpMovementSpeed": 16,
                                    "noteJumpStartBeatOffset": 0}]}
    diff = {"version": "4.0.0", "colorNotes": [], "colorNotesData": [], "obstacles": []}
    m = read_map(_write(tmp_path, info, diff), layers=True)
    assert m.difficulties[0].schema == "v4" and m.difficulties[0].layers is None
    assert m.environment == "BillieEnvironment"


def test_v3_missing_fields_default_to_zero(tmp_path):
    diff = {"version": "3.3.0", "colorNotes": [], "obstacles": [{"b": 4, "d": 1, "w": 1, "h": 5}],
            "basicBeatmapEvents": [{"b": 2, "et": 1}], "bombNotes": [{"b": 1}]}
    lay = read_map(_write(tmp_path, _v3_info(), diff), layers=True).difficulties[0].layers
    assert (lay.obstacles[0].x, lay.obstacles[0].y) == (0, 0)
    assert lay.lights[0].value == 0 and lay.lights[0].float_value == 1.0
    assert lay.bombs == (Bomb(0.5, 1, 0, 0),) and lay.dropped == 0


def test_v3_present_bad_value_still_dropped(tmp_path):
    diff = {"version": "3.3.0", "colorNotes": [], "bombNotes": [{"b": 1, "x": "a"}, {"b": 1, "y": None}]}
    lay = read_map(_write(tmp_path, _v3_info(), diff), layers=True).difficulties[0].layers
    assert lay.bombs == () and lay.dropped == 2


import pytest

from bsmaker.formats.tempo import TempoMap

BAD = ["str", None, float("nan"), float("inf"), True, 1e7]


def _lay(tmp_path, diff, v2=False):
    info = _v2_info()
    return read_map(_write(tmp_path, info, diff), layers=True).difficulties[0].layers


V3_OK = {
    "obstacles": {"b": 1, "x": 0, "y": 0, "d": 1, "w": 1, "h": 3},
    "bombNotes": {"b": 1, "x": 0, "y": 0},
    "sliders": {"b": 1, "c": 0, "x": 0, "y": 0, "d": 1, "tb": 2, "tx": 0, "ty": 0, "tc": 1},
    "burstSliders": {"b": 1, "c": 0, "x": 0, "y": 0, "d": 1, "tb": 2, "tx": 0, "ty": 0, "sc": 3, "s": 1},
    "basicBeatmapEvents": {"b": 1, "et": 0, "i": 1, "f": 1},
}
V2_OK = {
    "_obstacles": {"_time": 1, "_lineIndex": 0, "_type": 0, "_duration": 1, "_width": 1},
    "_notes": {"_time": 1, "_lineIndex": 0, "_lineLayer": 0, "_type": 3, "_cutDirection": 0},
    "_events": {"_time": 1, "_type": 0, "_value": 1},
}


def _bad_cases(ok):
    for key, obj in ok.items():
        for field in obj:
            if field in ("_type", "_cutDirection") and key in ("_notes", "_obstacles"):
                continue
            for bad in BAD:
                yield key, dict(obj, **{field: bad})


def _count(lay):
    return len(lay.obstacles) + len(lay.bombs) + len(lay.arcs) + len(lay.chains) + len(lay.lights)


def test_v3_bad_values_dropped(tmp_path):
    for key, bad in _bad_cases(V3_OK):
        diff = {"version": "3.3.0", "colorNotes": [], key: [bad]}
        lay = _lay(tmp_path, diff)
        assert (_count(lay), lay.dropped) == (0, 1), (key, bad)


def test_v2_bad_values_dropped(tmp_path):
    for key, bad in _bad_cases(V2_OK):
        diff = {"_version": "2.2.0", "_notes": [], key: [bad]} if key != "_notes" else \
            {"_version": "2.2.0", "_notes": [bad]}
        lay = _lay(tmp_path, diff)
        # a bad _time/_lineIndex etc. may turn a bomb into a non-bomb only for _type, skipped above
        assert (_count(lay), lay.dropped) == (0, 1), (key, bad)


def test_non_dict_entries_and_non_list_containers(tmp_path):
    diff = {"version": "3.3.0", "colorNotes": [], **{k: [5, "x", None] for k in V3_OK}}
    lay = _lay(tmp_path, diff)
    assert (_count(lay), lay.dropped) == (0, 15)
    diff = {"version": "3.3.0", "colorNotes": [], **{k: "x" for k in V3_OK}}
    lay = _lay(tmp_path, diff)
    assert (_count(lay), lay.dropped) == (0, 0)
    lay = _lay(tmp_path, {"_notes": [], "_obstacles": [3], "_events": [None], "_sliders": "x"})
    assert lay.dropped == 2


def test_tempo_times_all_kinds_v3(tmp_path):
    diff = {"version": "3.3.0", "colorNotes": [], "bpmEvents": [{"b": 0, "m": 120}, {"b": 4, "m": 60}],
            "bombNotes": [{"b": 6, "x": 1, "y": 1}],
            "burstSliders": [{"b": 5, "c": 0, "x": 0, "y": 0, "d": 1, "tb": 7, "tx": 0, "ty": 0, "sc": 3, "s": 1}],
            "basicBeatmapEvents": [{"b": 8, "et": 0, "i": 1}]}
    lay = _lay(tmp_path, diff)
    assert lay.bombs[0].time_s == 4.0
    assert (lay.chains[0].time_s, lay.chains[0].tail_time_s) == (3.0, 5.0)
    assert lay.lights[0].time_s == 6.0


def test_tempo_times_v2_events(tmp_path):
    diff = {"_notes": [], "_events": [{"_time": 4, "_type": 100, "_value": 0, "_floatValue": 60},
                                      {"_time": 6, "_type": 2, "_value": 1}],
            "_obstacles": [{"_time": 5, "_lineIndex": 0, "_type": 0, "_duration": 2, "_width": 1}]}
    lay = _lay(tmp_path, diff)
    assert lay.lights[0].time_s == 4.0 and len(lay.lights) == 1
    assert (lay.obstacles[0].time_s, lay.obstacles[0].duration_s) == (3.0, 2.0)


def test_rounding_and_default_float(tmp_path):
    diff = {"version": "3.3.0", "colorNotes": [], "bombNotes": [{"b": 1, "x": 0.4, "y": 2.6}],
            "basicBeatmapEvents": [{"b": 1, "et": 0, "i": 1}]}
    lay = _lay(tmp_path, diff)
    assert (lay.bombs[0].x, lay.bombs[0].y) == (0, 3) and lay.lights[0].float_value == 1.0
    lay = _lay(tmp_path, {"_notes": [], "_events": [{"_time": 1, "_type": 0, "_value": 1}]})
    assert lay.lights[0].float_value == 1.0


@pytest.mark.parametrize("key", ["lightColorEventBoxGroups", "lightRotationEventBoxGroups",
                                 "lightTranslationEventBoxGroups"])
def test_group_lighting_each_list(tmp_path, key):
    assert _lay(tmp_path, {"version": "3.3.0", "colorNotes": [], key: [{"b": 1}]}).group_lighting
    assert not _lay(tmp_path, {"version": "3.3.0", "colorNotes": [], key: []}).group_lighting


def test_bisect_matches_reference():
    import random
    rng = random.Random(1)
    pts = tuple((rng.uniform(1, 200), rng.uniform(60, 240)) for _ in range(50))
    tm = TempoMap(120.0, pts)

    def ref(beat):
        seconds, prev, bpm = 0.0, 0.0, 120.0
        for pb, pbpm in tm.points:
            if beat <= pb:
                break
            seconds += (pb - prev) * 60.0 / bpm
            prev, bpm = pb, pbpm
        return seconds + (beat - prev) * 60.0 / bpm

    beats = [rng.uniform(-5, 250) for _ in range(200)] + [b for b, _ in tm.points] + [0.0, -1.0]
    for b in beats:
        assert tm.to_seconds(b) == ref(b)
