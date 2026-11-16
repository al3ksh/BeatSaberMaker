"""Steam Beat Saber CustomLevels auto-detection on Windows."""
from __future__ import annotations

import re
from collections.abc import Sequence
from pathlib import Path

_PATH_RE = re.compile(r'"path"\s+"([^"]+)"', re.IGNORECASE)


def parse_library_folders_vdf(text: str) -> list[Path]:
    """Extract unique Steam library folder paths from `libraryfolders.vdf` text."""
    seen: set[str] = set()
    paths: list[Path] = []
    for raw in _PATH_RE.findall(text):
        unescaped = raw.replace("\\\\", "\\").strip()
        if not unescaped:
            continue
        key = unescaped.lower()
        if key not in seen:
            seen.add(key)
            paths.append(Path(unescaped))
    return paths


def _read_registry_steam_paths() -> tuple[list[Path], list[Path]]:
    """Return `(steam_roots, direct_game_dirs)` from the Windows Registry."""
    try:
        import winreg
    except ImportError:
        return [], []

    steam_roots: list[Path] = []
    game_dirs: list[Path] = []
    steam_keys = (
        (winreg.HKEY_CURRENT_USER, r"Software\Valve\Steam", "SteamPath"),
        (winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\WOW6432Node\Valve\Steam", "InstallPath"),
        (winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\Valve\Steam", "InstallPath"),
    )
    for hive, subkey, val_name in steam_keys:
        try:
            with winreg.OpenKey(hive, subkey) as key:
                val, _ = winreg.QueryValueEx(key, val_name)
                if isinstance(val, str) and val.strip():
                    steam_roots.append(Path(val.strip()))
        except OSError:
            continue

    try:
        with winreg.OpenKey(
            winreg.HKEY_LOCAL_MACHINE,
            r"SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall\Steam App 620980",
        ) as key:
            val, _ = winreg.QueryValueEx(key, "InstallLocation")
            if isinstance(val, str) and val.strip():
                game_dirs.append(Path(val.strip()))
    except OSError:
        pass

    return steam_roots, game_dirs


def _custom_levels_from_game_dir(game_dir: Path) -> Path | None:
    bs_data = game_dir / "Beat Saber_Data"
    if not bs_data.is_dir():
        return None
    custom = bs_data / "CustomLevels"
    custom.mkdir(parents=True, exist_ok=True)
    return custom


def detect_custom_levels(
    *,
    reg_paths: Sequence[Path] | None = None,
    steam_path_override: Path | None = None,
) -> Path | None:
    """Locate Beat Saber's `CustomLevels` directory across Steam libraries."""
    steam_roots: list[Path] = []
    direct_game_dirs: list[Path] = []

    if steam_path_override is not None:
        steam_roots.append(Path(steam_path_override))

    if reg_paths is not None:
        steam_roots.extend(Path(p) for p in reg_paths)
    else:
        reg_steam, reg_games = _read_registry_steam_paths()
        steam_roots.extend(reg_steam)
        direct_game_dirs.extend(reg_games)
        steam_roots.append(Path(r"C:\Program Files (x86)\Steam"))

    libraries: list[Path] = []
    seen_libs: set[str] = set()

    def _add_lib(p: Path) -> None:
        key = str(p).lower()
        if key not in seen_libs:
            seen_libs.add(key)
            libraries.append(p)

    for root in steam_roots:
        _add_lib(root)
        vdf = root / "steamapps" / "libraryfolders.vdf"
        if vdf.is_file():
            try:
                text = vdf.read_text(encoding="utf-8", errors="replace")
                for lib in parse_library_folders_vdf(text):
                    _add_lib(lib)
            except OSError:
                continue

    for lib in libraries:
        found = _custom_levels_from_game_dir(lib / "steamapps" / "common" / "Beat Saber")
        if found is not None:
            return found

    for game_dir in direct_game_dirs:
        found = _custom_levels_from_game_dir(game_dir)
        if found is not None:
            return found

    return None
