import json
from pathlib import Path

import pytest

from bsmaker.formats.model import Note
from bsmaker.formats.reader import read_map
from bsmaker.formats.writer import write_map

AUDIO = Path(__file__).parent.parent / "fixtures" / "maps" / "v2_basic" / "song.egg"


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
