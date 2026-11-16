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
