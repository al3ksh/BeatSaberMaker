import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from bsmaker.cli import app
from bsmaker.formats.model import Arc, Bomb, Chain, Layers, Note, Obstacle
from bsmaker.playability.layers import (
    NoteIndex, arc_ok, bomb_forbidden_cells, bomb_mask_ok, chain_ok, check_layers, wall_cells, wall_ok)

RULES = ("wall_on_note", "bomb_in_swing", "chain_conflict", "arc_overrun")


def note(t, x=0, y=0, color=0, d=8):
    return Note(t, t * 2, x, y, color, d)


def wall(t, dur, x=0, y=0, w=1, h=1):
    return Obstacle(t, t * 2, dur, dur * 2, x, y, w, h)


def chain(t, tail_t, color=0, x=0, y=0, tx=1, ty=0):
    return Chain(t, t * 2, color, x, y, 8, tail_t, tail_t * 2, tx, ty, 3, 1.0)


def arc(t, tail_t, color=0):
    return Arc(t, t * 2, color, 0, 0, 8, tail_t, tail_t * 2, 1, 0, 8)


def test_wall_cells_clipped():
    assert wall_cells(wall(0, 1, x=-2, y=-1, w=3, h=2)) == {(0, 0)}
    assert wall_cells(wall(0, 1, x=4, w=2)) == set()
    assert wall_cells(wall(0, 1, x=2, w=5, y=0, h=5)) == {(x, y) for x in (2, 3) for y in range(3)}
    assert wall_cells(wall(0, 1, w=0)) == set()


def test_wall_margin_boundaries():
    w = wall(1.0, 1.0)  # 1.0 s to 2.0 s, margin 0.1 s on both sides
    assert not wall_ok(w, [note(0.95)])
    assert wall_ok(w, [note(0.85)])
    assert not wall_ok(w, [note(2.05)])
    assert wall_ok(w, [note(2.15)])
    assert not wall_ok(w, [note(1.5)])
    assert wall_ok(w, [note(1.5, x=1)])  # another cell
    assert wall_ok(wall(1.0, -1.0, x=5), [note(1.0)])  # outside the grid covers nothing


@pytest.mark.parametrize("d,expected", [
    (0, {(1, 1), (1, 2), (1, 0)}), (1, {(1, 1), (1, 0), (1, 2)}),
    (2, {(1, 1), (0, 1), (2, 1)}), (3, {(1, 1), (2, 1), (0, 1)}),
    (4, {(1, 1), (0, 2), (2, 0)}), (5, {(1, 1), (2, 2), (0, 0)}),
    (6, {(1, 1), (0, 0), (2, 2)}), (7, {(1, 1), (2, 0), (0, 2)}),
    (8, {(1, 1)}),
])
def test_bomb_neighbours_all_directions(d, expected):
    assert bomb_forbidden_cells([note(5.0, 1, 1, d=d)], 5.0) == expected


def test_bomb_clipped_and_window():
    assert bomb_forbidden_cells([note(5.0, 0, 0, d=2)], 5.0) == {(0, 0), (1, 0)}
    assert bomb_forbidden_cells([note(5.39, 1, 1)], 5.0) == {(1, 1)}
    assert bomb_forbidden_cells([note(5.41, 1, 1)], 5.0) == set()
    assert bomb_forbidden_cells([note(4.61, 1, 1)], 5.0) == {(1, 1)}
    assert bomb_forbidden_cells([note(4.59, 1, 1)], 5.0) == set()
    assert bomb_forbidden_cells([], 5.0) == set()


def test_bomb_mask_ok():
    notes = [note(5.0, 1, 1, d=3)]
    assert not bomb_mask_ok(1 << (1 * 4 + 2), notes, 5.0)
    assert not bomb_mask_ok(1 << (1 * 4 + 0), notes, 5.0)
    assert bomb_mask_ok(1 << (2 * 4 + 3) | 1 << 3, notes, 5.0)
    assert bomb_mask_ok(0, notes, 5.0)
    assert bomb_mask_ok(1 << 5, notes, 6.0)


def test_chain_gap_boundary():
    assert chain_ok(chain(1.0, 1.5), [note(1.0), note(1.25, x=3, y=2)])
    assert not chain_ok(chain(1.0, 1.5), [note(1.0), note(1.2, x=3, y=2)])
    assert chain_ok(chain(1.0, 1.5), [note(1.0)])  # the hand has no later note


def test_chain_cells_free():
    head = note(1.0)
    assert not chain_ok(chain(1.0, 1.5), [head, note(1.3, x=1, y=0, color=1), note(2.0, x=3, y=2)])
    assert chain_ok(chain(1.0, 1.5), [head, note(2.0, x=3, y=2), note(2.1, x=1, y=0, color=1)])
    assert chain_ok(chain(1.0, 1.5), [head, note(2.0, x=1, y=0)])  # next note time is exclusive
    c = chain(1.0, 1.5, x=0, y=0, tx=2, ty=2)
    assert not chain_ok(c, [head, note(1.3, x=1, y=1, color=1)])
    assert chain_ok(c, [head, note(1.3, x=1, y=0, color=1)])


def test_arc_overrun_boundary():
    assert arc_ok(arc(1.0, 2.0), [note(2.0)])
    assert not arc_ok(arc(1.0, 2.1), [note(2.0)])
    assert arc_ok(arc(1.0, 9.0), [note(2.0, color=1)])
    assert arc_ok(arc(1.0, 9.0), [])


def test_check_layers_each_rule():
    notes = [note(1.0, 1, 1), note(4.0), note(4.1, x=2), note(7.5, x=2, y=2), note(10.0, x=3, y=1, color=1)]
    layers = Layers(bombs=(Bomb(10.0, 20.0, 3, 1),), obstacles=(wall(1.0, 0.5, x=1, y=1),),
                    arcs=(arc(7.0, 8.5),), chains=(chain(4.0, 4.5),), lights=(), group_lighting=False,
                    dropped=0)
    assert sorted(v.rule for v in check_layers(notes, layers)) == sorted(RULES)
    assert check_layers(notes, Layers((), (), (), (), (), False, 0)) == []


def test_note_index_accepts_prebuilt():
    idx = NoteIndex([note(2.0), note(1.0)])
    assert idx.times == [1.0, 2.0]
    assert wall_ok(wall(5, 1), idx)


def _cli_map(folder: Path) -> Path:
    folder.mkdir()
    info = {"_version": "2.0.0", "_songName": "S", "_songAuthorName": "A", "_levelAuthorName": "M",
            "_beatsPerMinute": 120, "_songFilename": "song.egg", "_environmentName": "NiceEnvironment",
            "_difficultyBeatmapSets": [{"_beatmapCharacteristicName": "Standard", "_difficultyBeatmaps": [
                {"_difficulty": "Expert", "_beatmapFilename": "E.dat", "_noteJumpMovementSpeed": 16,
                 "_noteJumpStartBeatOffset": 0}]}]}

    def n(b, x, y, c):
        return {"b": b, "x": x, "y": y, "c": c, "d": 8, "a": 0}

    diff = {"version": "3.2.0", "bpmEvents": [], "rotationEvents": [],
            "colorNotes": [n(2, 1, 1, 0), n(8, 0, 0, 0), n(8.2, 2, 0, 0), n(15, 2, 2, 0), n(20, 3, 1, 1)],
            "bombNotes": [{"b": 20, "x": 3, "y": 1}],
            "obstacles": [{"b": 2, "x": 1, "y": 1, "d": 1, "w": 1, "h": 1}],
            "burstSliders": [{"b": 8, "c": 0, "x": 0, "y": 0, "d": 8, "tb": 9, "tx": 1, "ty": 0,
                              "sc": 3, "s": 1}],
            "sliders": [{"b": 14, "c": 0, "x": 0, "y": 0, "d": 8, "mu": 1, "tb": 17, "tx": 1, "ty": 0,
                         "tc": 8, "tmu": 1, "m": 0}],
            "basicBeatmapEvents": []}
    (folder / "Info.dat").write_text(json.dumps(info), encoding="utf-8")
    (folder / "E.dat").write_text(json.dumps(diff), encoding="utf-8")
    return folder


def test_cli_check_layer_violations(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    folder = _cli_map(tmp_path / "m")
    result = CliRunner().invoke(app, ["check", str(folder)])
    assert result.exit_code == 0, result.output
    assert "layers: walls 1, bombs 1, arcs 1, chains 1, light events 0" in result.output
    for rule in RULES:
        assert f"{rule}=1" in result.output
    detail = [ln for ln in result.output.splitlines()
              if ln.strip().split(" ")[-1] in RULES and "=" not in ln]
    assert len(detail) == 4
