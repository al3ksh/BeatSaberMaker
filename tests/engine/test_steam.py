from pathlib import Path

from bsmaker.engine.steam import detect_custom_levels, parse_library_folders_vdf

VDF_SAMPLE = r'''
"libraryfolders"
{
	"0"
	{
		"path"		"C:\\Program Files (x86)\\Steam"
		"apps"
		{
			"228980"		"0"
		}
	}
	"1"
	{
		"path"		"D:\\SteamLibrary"
		"apps"
		{
			"620980"		"123456"
		}
	}
}
'''


def test_parse_library_folders_vdf_and_detect_secondary_drive(tmp_path: Path) -> None:
    paths = parse_library_folders_vdf(VDF_SAMPLE)
    assert paths == [Path(r"C:\Program Files (x86)\Steam"), Path(r"D:\SteamLibrary")]

    steam_root = tmp_path / "Steam"
    lib2 = tmp_path / "D_SteamLibrary"
    (steam_root / "steamapps").mkdir(parents=True)
    bs_data = lib2 / "steamapps" / "common" / "Beat Saber" / "Beat Saber_Data"
    bs_data.mkdir(parents=True)
    vdf = f'"libraryfolders"\n{{\n "0" {{ "path" "{str(steam_root).replace("\\", "\\\\")}" }}\n "1" {{ "path" "{str(lib2).replace("\\", "\\\\")}" }}\n}}\n'
    (steam_root / "steamapps" / "libraryfolders.vdf").write_text(vdf, encoding="utf-8")

    found = detect_custom_levels(reg_paths=[steam_root])
    assert found == bs_data / "CustomLevels"
    assert found.is_dir()

    assert detect_custom_levels(reg_paths=[tmp_path / "missing_steam"]) is None
