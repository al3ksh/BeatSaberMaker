import zipfile
from pathlib import Path

import pytest

from bsmaker.data import extract
from bsmaker.data.extract import UnsafeArchiveError, safe_extract


def _zip(path: Path, entries: list[tuple[str, bytes]]) -> Path:
    with zipfile.ZipFile(path, "w") as z:
        for name, data in entries:
            z.writestr(name, data)
    return path


def _names(d: Path) -> set[str]:
    return {p.name for p in d.iterdir()}


def _empty_or_absent(d: Path) -> bool:
    return not d.exists() or not list(d.iterdir())


def test_extracts_dat_and_audio_only(tmp_path):
    z = _zip(tmp_path / "a.zip", [("Info.dat", b"{}"), ("Easy.DAT", b"{}"), ("song.egg", b"x"),
                                  ("cover.jpg", b"j"), ("evil.exe", b"e"), ("script.py", b"p"),
                                  ("a.ogg", b"o"), ("b.wav", b"w")])
    out = safe_extract(z, tmp_path / "dest")
    assert _names(tmp_path / "dest") == {"Info.dat", "Easy.DAT", "song.egg", "a.ogg", "b.wav"}
    assert {p.name for p in out} == _names(tmp_path / "dest")


def test_path_traversal_is_flattened(tmp_path):
    z = _zip(tmp_path / "a.zip", [("../../outside.dat", b"1"), ("C:/abs.dat", b"2"),
                                  ("..\\..\\back.dat", b"3")])
    safe_extract(z, tmp_path / "dest")
    assert (tmp_path / "dest" / "outside.dat").read_bytes() == b"1"
    assert (tmp_path / "dest" / "back.dat").read_bytes() == b"3"
    assert (tmp_path / "dest" / "abs.dat").read_bytes() == b"2"
    assert not (tmp_path.parent / "outside.dat").exists() and not (tmp_path / "outside.dat").exists()
    assert not (tmp_path / "abs.dat").exists() and not (tmp_path / "back.dat").exists()


def test_oversize_entry_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(extract, "MAX_ENTRY_BYTES", 1000)
    z = _zip(tmp_path / "a.zip", [("ok.dat", b"x"), ("big.dat", b"x" * 5000)])
    with pytest.raises(UnsafeArchiveError):
        safe_extract(z, tmp_path / "dest")
    assert _empty_or_absent(tmp_path / "dest")


def test_lying_header_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(extract, "MAX_ENTRY_BYTES", 1000)
    z = _zip(tmp_path / "a.zip", [("big.dat", b"x" * 5000)])
    real = zipfile.ZipFile.infolist

    def lying(self):
        infos = real(self)
        for i in infos:
            i.file_size = 10
        return infos
    monkeypatch.setattr(zipfile.ZipFile, "infolist", lying)
    with pytest.raises(UnsafeArchiveError):
        safe_extract(z, tmp_path / "dest")
    assert _empty_or_absent(tmp_path / "dest")


def test_total_size_limit(tmp_path, monkeypatch):
    monkeypatch.setattr(extract, "MAX_ENTRY_BYTES", 1000)
    monkeypatch.setattr(extract, "MAX_TOTAL_BYTES", 2500)
    z = _zip(tmp_path / "a.zip", [(f"{i}.dat", b"x" * 900) for i in range(4)])
    with pytest.raises(UnsafeArchiveError):
        safe_extract(z, tmp_path / "dest")
    assert _empty_or_absent(tmp_path / "dest")


def test_nested_dirs_flattened(tmp_path):
    z = _zip(tmp_path / "a.zip", [("a/b/Info.dat", b"{}"), ("a/", b""), ("a/b/song.egg", b"x")])
    safe_extract(z, tmp_path / "dest")
    assert _names(tmp_path / "dest") == {"Info.dat", "song.egg"}


def test_duplicate_basename_first_wins(tmp_path):
    z = _zip(tmp_path / "a.zip", [("x/Info.dat", b"first"), ("y/Info.dat", b"second")])
    safe_extract(z, tmp_path / "dest")
    assert (tmp_path / "dest" / "Info.dat").read_bytes() == b"first"


@pytest.mark.parametrize("name", ["CON.dat", "nul.dat", "Com1.egg", "lpt9.dat", "aux.ogg", "PRN.wav",
                                  "a:b.dat", "x/b:file.dat", ".hidden.dat", "trail.dat.",
                                  "trail.dat ", ".dat"])
def test_reserved_and_odd_names_skipped(tmp_path, name):
    z = _zip(tmp_path / "a.zip", [(name, b"x"), ("Info.dat", b"{}")])
    safe_extract(z, tmp_path / "dest")
    assert _names(tmp_path / "dest") == {"Info.dat"}


def test_corrupt_zip_rejected(tmp_path):
    bad = tmp_path / "bad.zip"
    bad.write_bytes(b"this is not a zip file")
    with pytest.raises(UnsafeArchiveError):
        safe_extract(bad, tmp_path / "dest")


def test_dest_must_be_new_or_empty(tmp_path):
    z = _zip(tmp_path / "a.zip", [("Info.dat", b"{}")])
    dest = tmp_path / "dest"
    dest.mkdir()
    (dest / "junk").write_bytes(b"x")
    with pytest.raises(UnsafeArchiveError):
        safe_extract(z, dest)
    assert _names(dest) == {"junk"}


def test_empty_existing_dest_ok(tmp_path):
    z = _zip(tmp_path / "a.zip", [("Info.dat", b"{}")])
    (tmp_path / "dest").mkdir()
    assert len(safe_extract(z, tmp_path / "dest")) == 1
