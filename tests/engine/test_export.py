import zipfile
from pathlib import Path
import pytest

from bsmaker.engine.export import export_map_zip


def test_export_map_zip_places_files_at_root_and_rejects_non_map(tmp_path: Path) -> None:
    folder = tmp_path / "BSM - Test"
    folder.mkdir()
    (folder / "Info.dat").write_text("{}", encoding="utf-8")
    (folder / "ExpertStandard.dat").write_text("{}", encoding="utf-8")
    (folder / "song.egg").write_bytes(b"OggS")
    (folder / "ignored.tmp").write_bytes(b"x")

    out = export_map_zip(folder, tmp_path / "exports" / "Test.zip")
    assert out.is_file()
    with zipfile.ZipFile(out) as zf:
        assert sorted(zf.namelist()) == ["ExpertStandard.dat", "Info.dat", "song.egg"]

    not_map = tmp_path / "empty"
    not_map.mkdir()
    with pytest.raises(ValueError, match="Info.dat"):
        export_map_zip(not_map, tmp_path / "bad.zip")
