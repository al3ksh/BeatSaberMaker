import json
from pathlib import Path

import pytest

from bsmaker.formats.model import Arc, Bomb, Chain, LightEvent, Note, Obstacle
from bsmaker.formats.reader import read_map
from bsmaker.formats.writer import LevelData, write_map, write_map_multi

AUDIO = Path(__file__).parent.parent / "fixtures" / "maps" / "v2_basic" / "song.egg"
GOLDEN = Path(__file__).parent.parent / "fixtures" / "writer_golden"  # written by the pre-Plan-4 writer


def _notes():
    return [Note(b * 0.5, float(b), b % 4, b % 3, b % 2, b % 9) for b in range(1, 9)]


def _write(folder, **kw):
    args = dict(title="T", artist="A", bpm=120.0, rank="Expert", njs=16.0, njs_offset=0.0,
                notes=_notes(), audio_src=AUDIO)
    args.update(kw)
    return write_map(folder, **args)


def test_roundtrip_notes(tmp_path):
    folder = _write(tmp_path / "BSM - t")
    m = read_map(folder)
    d = m.difficulties[0]
    assert [(n.beat, n.x, n.y, n.color, n.direction) for n in d.notes] == sorted(
        [(n.beat, n.x, n.y, n.color, n.direction) for n in _notes()],
        key=lambda t: (t[0], t[3], t[1], t[2]))
    assert d.schema == "v3"
    assert m.mapper == "BeatSaberMaker"
    assert m.problems == ()


def test_writes_expected_files(tmp_path):
    folder = _write(tmp_path / "BSM - t")
    assert sorted(p.name for p in folder.iterdir()) == [
        "ExpertStandard.dat", "Info.dat", "cover.png", "song.egg"]
    assert (folder / "cover.png").read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    assert (folder / "song.egg").read_bytes()[:4] == b"OggS"


def test_refuses_existing_folder(tmp_path):
    folder = _write(tmp_path / "BSM - t")
    with pytest.raises(FileExistsError):
        _write(folder)


def test_overwrite_requires_bsm_prefix(tmp_path):
    other = tmp_path / "Other"
    other.mkdir()
    with pytest.raises(ValueError):
        _write(other, overwrite=True)
    folder = _write(tmp_path / "BSM - t")
    _write(folder, overwrite=True, rank="Easy")
    assert (folder / "EasyStandard.dat").exists()
    assert not (folder / "ExpertStandard.dat").exists()


def test_info_has_no_requirements(tmp_path):
    folder = _write(tmp_path / "BSM - t")
    info = json.loads((folder / "Info.dat").read_text(encoding="utf-8"))
    assert info["_version"] == "2.1.0"
    bm = info["_difficultyBeatmapSets"][0]["_difficultyBeatmaps"][0]
    assert bm["_difficultyRank"] == 7
    assert "_customData" not in bm
    assert read_map(folder).requirements == ()


def test_ffmpeg_failure_leaves_no_folder(tmp_path):
    bad = tmp_path / "bad.wav"
    bad.write_bytes(b"not audio")
    with pytest.raises(RuntimeError, match="ffmpeg"):
        _write(tmp_path / "BSM - x", audio_src=bad)
    assert not (tmp_path / "BSM - x").exists()


def test_ffmpeg_timeout_is_a_clean_error(tmp_path, monkeypatch):
    import subprocess

    from bsmaker.formats import writer as W

    seen = {}

    def fake_run(cmd, **kw):
        seen["cmd"], seen["timeout"] = cmd, kw.get("timeout")
        raise subprocess.TimeoutExpired(cmd, kw.get("timeout"))

    monkeypatch.setattr(W.subprocess, "run", fake_run)
    with pytest.raises(RuntimeError, match="timed out"):
        W._encode_audio(tmp_path / "a.wav", tmp_path / "b.egg")
    assert "-nostdin" in seen["cmd"] and seen["timeout"] == 300


def test_notes_only_output_is_byte_identical_to_the_old_writer(tmp_path):
    folder = _write(tmp_path / "BSM - t")
    for name in ("Info.dat", "ExpertStandard.dat"):
        assert (folder / name).read_bytes() == (GOLDEN / name).read_bytes()


def _expert():
    notes = [Note(1.0, 2.0, 1, 0, 0, 1), Note(2.0, 4.0, 2, 1, 1, 1), Note(3.0, 6.0, 1, 0, 0, 1)]
    return LevelData(
        "Expert", 18.0, 0.25, notes,
        bombs=[Bomb(1.5, 0, 0, 1), Bomb(2.5, 0, 3, 2)],
        obstacles=[Obstacle(2.0, 0, 1.0, 0, -1, 0, 1, 5), Obstacle(3.5, 0, 0.5, 0, 4, 2, 1, 3)],
        arcs=[Arc(1.0, 0, 0, 1, 0, 1, 3.0, 0, 2, 2, 4)],
        chains=[Chain(2.0, 0, 1, 2, 1, 1, 2.5, 0, 3, 2, 5, 0.5)])


def _easy():
    return LevelData("Easy", 10.0, 0.0, [Note(1.0, 2.0, 1, 1, 0, 8)])


LIGHTS = [LightEvent(0.0, 0, 1, 5, 1.0), LightEvent(1.0, 0, 4, 2, 0.5), LightEvent(2.25, 0, 12, 3, 1.0)]


def test_multi_level_roundtrip_with_layers(tmp_path):
    folder = write_map_multi(tmp_path / "BSM - m", title="T", artist="A", bpm=120.0,
                             levels=[_expert(), _easy()], lights=LIGHTS, environment="NiceEnvironment",
                             audio_src=AUDIO)
    assert sorted(p.name for p in folder.iterdir()) == [
        "EasyStandard.dat", "ExpertStandard.dat", "Info.dat", "cover.png", "song.egg"]
    info = json.loads((folder / "Info.dat").read_text(encoding="utf-8"))
    assert info["_environmentName"] == "NiceEnvironment"
    names = [d["_difficulty"] for d in info["_difficultyBeatmapSets"][0]["_difficultyBeatmaps"]]
    assert names == ["Easy", "Expert"]
    m = read_map(folder, layers=True)
    assert m.environment == "NiceEnvironment" and m.problems == ()
    by = {d.rank: d for d in m.difficulties}
    e, lay = by["Expert"], by["Expert"].layers
    assert (e.njs, e.njs_offset) == (18.0, 0.25) and len(e.notes) == 3
    assert [(b.time_s, b.x, b.y) for b in lay.bombs] == [(1.5, 0, 1), (2.5, 3, 2)]
    exp_ob = [(2.0, 1.0, -1, 0, 1, 5), (3.5, 0.5, 4, 2, 1, 3)]
    assert [(o.time_s, o.duration_s, o.x, o.y, o.w, o.h) for o in lay.obstacles] == exp_ob
    assert [o.duration_beats for o in lay.obstacles] == [2.0, 1.0]
    (a,) = lay.arcs
    assert (a.time_s, a.beat, a.color, a.x, a.y, a.d) == (1.0, 2.0, 0, 1, 0, 1)
    assert (a.tail_time_s, a.tail_beat, a.tx, a.ty, a.td) == (3.0, 6.0, 2, 2, 4)
    (c,) = lay.chains
    assert (c.time_s, c.color, c.x, c.y, c.d) == (2.0, 1, 2, 1, 1)
    assert (c.tail_time_s, c.tx, c.ty, c.segments, c.squish) == (2.5, 3, 2, 5, 0.5)
    assert [(l.time_s, l.type, l.value, l.float_value) for l in lay.lights] == [
        (0.0, 1, 5, 1.0), (1.0, 4, 2, 0.5), (2.25, 12, 3, 1.0)]
    assert not lay.group_lighting and lay.dropped == 0
    assert by["Easy"].layers.lights == lay.lights and len(by["Easy"].notes) == 1
    assert by["Easy"].layers.bombs == () and by["Easy"].njs == 10.0


def test_object_beats_are_ignored(tmp_path):
    lv = LevelData("Expert", 16.0, 0.0, [], bombs=[Bomb(1.0, 99.0, 1, 1)])
    folder = write_map_multi(tmp_path / "BSM - b", title="T", artist="A", bpm=60.0, levels=[lv],
                             audio_src=AUDIO)
    raw = json.loads((folder / "ExpertStandard.dat").read_text(encoding="utf-8"))
    assert raw["bombNotes"] == [{"b": 1.0, "x": 1, "y": 1}]


def _multi(tmp_path, levels, **kw):
    return write_map_multi(tmp_path / "BSM - x", title="T", artist="A", bpm=120.0, levels=levels,
                           audio_src=AUDIO, **kw)


def test_duplicate_and_unknown_ranks_rejected(tmp_path):
    with pytest.raises(ValueError, match="duplicate"):
        _multi(tmp_path, [_easy(), _easy()])
    with pytest.raises(ValueError, match="unknown rank"):
        _multi(tmp_path, [LevelData("Insane", 16.0, 0.0, [])])
    assert not (tmp_path / "BSM - x").exists()


def test_environment_must_be_nonempty_string(tmp_path):
    for bad in ("", None, ["Nice"]):
        with pytest.raises(ValueError, match="environment"):
            _multi(tmp_path, [_easy()], environment=bad)


@pytest.mark.parametrize("kwargs", [
    {"obstacles": [Obstacle(1.0, 0, -1.0, -2.0, 0, 0, 1, 5)]},    # negative duration
    {"obstacles": [Obstacle(1.0, 0, 0.0, 0.0, 0, 0, 1, 5)]},      # zero duration
    {"obstacles": [Obstacle(1.0, 0, 1.0, 2.0, 0, 0, 0, 5)]},      # zero width
    {"obstacles": [Obstacle(1.0, 0, 1.0, 2.0, 5, 0, 1, 5)]},      # x out of range
    {"obstacles": [Obstacle(1.0, 0, 1.0, 2.0, 0, 3, 1, 5)]},      # y out of range
    {"obstacles": [Obstacle(1.0, 0, 1.0, 2.0, 0, 0, 1, 6)]},      # height out of range
    {"bombs": [Bomb(1.0, 0, 4, 0)]},                               # bomb outside the grid
    {"bombs": [Bomb(1.0, 0, 0, -1)]},
    {"arcs": [Arc(1.0, 0, 0, 1, 0, 1, 2.0, 0, 4, 0, 1)]},          # arc tail outside the grid
    {"arcs": [Arc(1.0, 0, 0, 1, 3, 1, 2.0, 0, 0, 0, 1)]},          # arc head outside the grid
    {"chains": [Chain(1.0, 0, 0, 1, 0, 1, 2.0, 0, 1, 3, 3, 1.0)]},  # chain tail outside the grid
])
def test_invalid_layer_objects_rejected(tmp_path, kwargs):
    with pytest.raises(ValueError):
        _multi(tmp_path, [LevelData("Expert", 16.0, 0.0, [], **kwargs)])
    assert not (tmp_path / "BSM - x").exists()


_A = dict(time_s=1.0, beat=0, color=0, x=1, y=0, d=1, tail_time_s=2.0, tail_beat=0, tx=2, ty=1, td=1)
_C = dict(time_s=1.0, beat=0, color=0, x=1, y=0, d=1, tail_time_s=2.0, tail_beat=0, tx=2, ty=1,
          segments=3, squish=0.5)
_NAN, _INF = float("nan"), float("inf")


@pytest.mark.parametrize("kwargs,what", [
    ({"arcs": [Arc(**{**_A, "d": 9})]}, "arc 0"),
    ({"arcs": [Arc(**{**_A, "td": -1})]}, "arc 0"),
    ({"arcs": [Arc(**{**_A, "tail_time_s": _NAN})]}, "arc 0"),
    ({"arcs": [Arc(**_A), Arc(**{**_A, "time_s": -1.0})]}, "arc 1"),
    ({"chains": [Chain(**{**_C, "d": 9})]}, "chain 0"),
    ({"chains": [Chain(**{**_C, "segments": 0})]}, "chain 0"),
    ({"chains": [Chain(**{**_C, "squish": 0.0})]}, "chain 0"),
    ({"chains": [Chain(**{**_C, "squish": 1.5})]}, "chain 0"),
    ({"chains": [Chain(**{**_C, "squish": _NAN})]}, "chain 0"),
    ({"chains": [Chain(**{**_C, "tail_time_s": _INF})]}, "chain 0"),
    ({"bombs": [Bomb(_NAN, 0, 1, 1)]}, "bomb 0"),
    ({"bombs": [Bomb(1.0, 0, 1, 1), Bomb(-0.5, 0, 1, 1)]}, "bomb 1"),
    ({"obstacles": [Obstacle(_INF, 0, 1.0, 1.0, 0, 0, 1, 5)]}, "wall 0"),
    ({"obstacles": [Obstacle(1.0, 0, _NAN, 1.0, 0, 0, 1, 5)]}, "wall 0"),
    ({"obstacles": [Obstacle(1.0, 0, _INF, 1.0, 0, 0, 1, 5)]}, "wall 0"),
    ({"obstacles": [Obstacle(-1.0, 0, 1.0, 1.0, 0, 0, 1, 5)]}, "wall 0"),
])
def test_review_validation_names_object_and_index(tmp_path, kwargs, what):
    with pytest.raises(ValueError, match=what):
        _multi(tmp_path, [LevelData("Expert", 16.0, 0.0, [], **kwargs)])
    assert not (tmp_path / "BSM - x").exists()


@pytest.mark.parametrize("ev", [LightEvent(_NAN, 0, 1, 1, 1.0), LightEvent(-1.0, 0, 1, 1, 1.0),
                                LightEvent(1.0, 0, 1, 1, _NAN)])
def test_invalid_light_events_rejected(tmp_path, ev):
    with pytest.raises(ValueError, match="light event 1"):
        _multi(tmp_path, [_easy()], lights=[LightEvent(0.0, 0, 1, 1, 1.0), ev])


def test_writer_forwards_explicit_ffmpeg_path(tmp_path, monkeypatch):
    from bsmaker.formats import writer as W

    fake_bin = tmp_path / "ffmpeg.exe"
    fake_bin.write_bytes(b"")
    seen = {}

    def fake_encode(src, dst, *, ffmpeg_path=None):
        seen["ffmpeg_path"] = ffmpeg_path
        dst.write_bytes(b"OggS")

    monkeypatch.setattr(W, "_encode_audio", fake_encode)
    write_map_multi(
        tmp_path / "BSM - f", title="T", artist="A", bpm=120.0,
        levels=[_easy()], audio_src=AUDIO, ffmpeg_path=fake_bin,
    )
    assert seen["ffmpeg_path"] == fake_bin
