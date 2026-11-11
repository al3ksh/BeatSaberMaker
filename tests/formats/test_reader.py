import json
import shutil
from pathlib import Path

import pytest

from bsmaker.formats.reader import map_key, read_map

FIX = Path(__file__).resolve().parents[1] / "fixtures" / "maps"
EXPERT = "ExpertStandard.dat"


def _copy(name: str, tmp_path: Path) -> Path:
    dst = tmp_path / name
    shutil.copytree(FIX / name, dst)
    return dst


def _edit(path: Path, fn) -> None:
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    fn(data)
    path.write_text(json.dumps(data), encoding="utf-8")


def _expert(m):
    return m.difficulties[-1]


def test_reads_v2_basic():
    m = read_map(FIX / "v2_basic")
    d = m.difficulties[0]
    assert (m.title, m.artist, m.mapper, m.bpm) == ("Fixture Song", "Fixture Artist", "Fixture Mapper", 120)
    assert m.audio_path is not None and m.audio_path.name == "song.egg"
    assert m.requirements == ("Mapping Extensions",) and m.suggestions == ("Chroma",)
    assert d.schema == "v2" and d.characteristic == "Standard" and d.rank == "Expert"
    assert len(d.notes) == 6 and d.n_bombs == 1 and d.n_obstacles == 1
    assert d.njs == 16 and d.njs_offset == 0
    n = d.notes[0]
    assert (n.time_s, n.beat, n.x, n.y, n.color, n.direction) == (1.0, 2, 1, 0, 0, 1)
    assert not any(d.flags.__dict__.values())
    assert m.problems == ()


def test_reads_v3_basic():
    d = read_map(FIX / "v3_basic").difficulties[0]
    assert d.schema == "v3" and len(d.notes) == 6 and d.n_bombs == 1
    assert d.n_arcs == 1 and d.n_chains == 0 and d.n_obstacles == 1
    n = d.notes[0]
    assert (n.time_s, n.beat, n.x, n.y, n.color, n.direction) == (1.0, 2, 1, 0, 0, 1)


def test_reads_v4_basic():
    m = read_map(FIX / "v4_basic")
    d = m.difficulties[0]
    assert (m.title, m.artist, m.mapper, m.bpm) == ("Fixture Song", "Fixture Artist", "Fixture Mapper", 120)
    assert m.audio_path is not None
    assert d.schema == "v4" and d.characteristic == "Standard" and d.rank == "Expert"
    assert len(d.notes) == 6 and d.n_bombs == 1
    n = d.notes[0]
    assert (n.time_s, n.beat, n.x, n.y, n.color, n.direction) == (1.0, 2, 1, 0, 0, 1)
    assert d.notes[-1].direction == 8


def test_schema_detected_by_keys_not_version(tmp_path):
    folder = _copy("v2_basic", tmp_path)
    _edit(folder / EXPERT, lambda d: d.update({"version": "3.0.0", "_version": "3.0.0"}))
    assert read_map(folder).difficulties[0].schema == "v2"
    folder = _copy("v3_basic", tmp_path)
    _edit(folder / EXPERT, lambda d: d.update({"version": "2.0.0"}))
    assert read_map(folder).difficulties[0].schema == "v3"


def _bpm_customdata(f):
    _edit(f / EXPERT, lambda d: d.update(_customData={"_bpmChanges": [{"_time": 1, "_bpm": 90}]}))


def _bpm_event100(f):
    _edit(f / EXPERT, lambda d: d["_events"].append({"_time": 1, "_type": 100, "_value": 140}))


def _bpm_bpminfo(f):
    (f / "BPMInfo.dat").write_text(json.dumps({"_regions": [{}, {}]}), encoding="utf-8")


def _bpm_v3_two(f):
    _edit(f / EXPERT, lambda d: d.update(bpmEvents=[{"b": 0, "m": 120}, {"b": 8, "m": 130}]))


def _bpm_v3_differs(f):
    _edit(f / EXPERT, lambda d: d.update(bpmEvents=[{"b": 0, "m": 130}]))


def _bpm_audiodata(f):
    (f / "AudioData.dat").write_text(json.dumps({"bpmData": [{}, {}]}), encoding="utf-8")


@pytest.mark.parametrize("base,mutate,expected", [
    ("v2_basic", None, False),
    ("v2_basic", _bpm_customdata, True),
    ("v2_basic", _bpm_event100, True),
    ("v2_basic", _bpm_bpminfo, True),
    ("v3_basic", None, False),
    ("v3_basic", _bpm_v3_two, True),
    ("v3_basic", _bpm_v3_differs, True),
    ("v4_basic", None, False),
    ("v4_basic", _bpm_audiodata, True),
])
def test_bpm_change_flag(tmp_path, base, mutate, expected):
    folder = _copy(base, tmp_path)
    if mutate:
        mutate(folder)
    name = "ExpertStandard.beatmap.dat" if base == "v4_basic" else EXPERT
    assert (folder / name).is_file()
    assert read_map(folder).difficulties[0].flags.has_bpm_changes is expected


def test_bpm_change_fixture():
    assert read_map(FIX / "v2_bpmchanges").difficulties[0].flags.has_bpm_changes


def test_offgrid_flag_and_exclusion():
    d = read_map(FIX / "v2_offgrid").difficulties[0]
    assert d.flags.has_offgrid_notes
    assert len(d.notes) == 6
    assert all(0 <= n.x <= 3 and 0 <= n.direction <= 8 for n in d.notes)


def test_custom_note_data_flag(tmp_path):
    assert not read_map(FIX / "v2_basic").difficulties[0].flags.has_custom_note_data
    folder = _copy("v2_basic", tmp_path)
    _edit(folder / EXPERT, lambda d: d["_notes"][0].update(_customData={"_color": [1, 0, 0]}))
    assert read_map(folder).difficulties[0].flags.has_custom_note_data
    folder = _copy("v3_basic", tmp_path)
    _edit(folder / EXPERT, lambda d: d["colorNotes"][0].update(customData={"color": [1, 0, 0]}))
    assert read_map(folder).difficulties[0].flags.has_custom_note_data


def test_time_offset_flag(tmp_path):
    folder = _copy("v2_basic", tmp_path)
    _edit(folder / "Info.dat", lambda d: d.update(_songTimeOffset=0.5))
    assert read_map(folder).difficulties[0].flags.has_time_offset


def test_notes_sorted():
    d = read_map(FIX / "v2_reversed").difficulties[0]
    keys = [(n.time_s, n.color, n.x, n.y) for n in d.notes]
    assert keys == sorted(keys) and len(keys) == 6


def test_bom_and_lowercase_info():
    m = read_map(FIX / "v2_bom_lowercase")
    assert m.title == "Fixture Song" and len(m.difficulties) == 1 and m.problems == ()


def test_missing_difficulty_recorded():
    m = read_map(FIX / "v2_missing_diff")
    assert len(m.difficulties) == 1
    assert m.problems and "ExpertStandard.dat" in m.problems[0]


def test_broken_json_recorded(tmp_path):
    folder = _copy("v2_basic", tmp_path)
    (folder / EXPERT).write_text("{not json", encoding="utf-8")
    m = read_map(folder)
    assert m.difficulties == () and len(m.problems) == 1


def test_map_key():
    assert map_key("1021b (Art Of The Devil - nitronik.exe)") == "1021b"
    k = map_key("My Local Map")
    assert k.startswith("local-") and len(k) == len("local-") + 10
    assert map_key("deadbeef") == "deadbeef"  # bare hex name is a key (scraped folders)


def test_omitted_zero_fields_default_to_zero(tmp_path):
    folder = _copy("v3_basic", tmp_path)
    _edit(folder / EXPERT, lambda d: d.update(colorNotes=[{"b": 1}]))
    n = read_map(folder).difficulties[0].notes[0]
    assert (n.x, n.y, n.color, n.direction) == (0, 0, 0, 0)
    folder = _copy("v4_basic", tmp_path)
    _edit(folder / "ExpertStandard.beatmap.dat",
          lambda d: d.update(colorNotes=[{"b": 1}], colorNotesData=[{"c": 1}]))
    n = read_map(folder).difficulties[0].notes[0]
    assert (n.x, n.y, n.color, n.direction) == (0, 0, 1, 0)


def test_audio_fallback_to_audio_ogg(tmp_path):
    folder = _copy("v2_basic", tmp_path)
    (folder / "song.egg").rename(folder / "audio.ogg")
    m = read_map(folder)
    assert m.audio_path is not None and m.audio_path.name == "audio.ogg"


def test_map_key_bare_and_prefixed():
    assert map_key("2f3a") == "2f3a"
    assert map_key("2f3a (Song - Mapper)") == "2f3a"
    assert map_key("Song Name").startswith("local-")
    assert map_key("xyz").startswith("local-")


@pytest.mark.parametrize("bad", ["../song.egg", "sub/song.egg", "sub\\song.egg", "C:song.egg",
                                 "C:\\song.egg", "/abs.egg", "..", "."])
def test_unsafe_audio_name_treated_as_missing(tmp_path, bad):
    folder = _copy("v2_basic", tmp_path)
    (tmp_path / "song.egg").write_bytes(b"x")
    (folder / "sub").mkdir()
    (folder / "sub" / "song.egg").write_bytes(b"x")
    _edit(folder / "Info.dat", lambda d: d.update(_songFilename=bad))
    assert read_map(folder).audio_path is None


@pytest.mark.parametrize("bad", ["../ExpertStandard.dat", "sub/ExpertStandard.dat",
                                 "sub\\ExpertStandard.dat", "C:ExpertStandard.dat"])
def test_unsafe_difficulty_name_treated_as_missing(tmp_path, bad):
    folder = _copy("v2_basic", tmp_path)
    (tmp_path / "ExpertStandard.dat").write_bytes((folder / EXPERT).read_bytes())
    (folder / "sub").mkdir()
    (folder / "sub" / "ExpertStandard.dat").write_bytes((folder / EXPERT).read_bytes())

    def edit(d):
        d["_difficultyBeatmapSets"][0]["_difficultyBeatmaps"][0]["_beatmapFilename"] = bad
    _edit(folder / "Info.dat", edit)
    m = read_map(folder)
    assert m.difficulties == () and any("missing difficulty file" in p for p in m.problems)
