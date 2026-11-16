"""Export Beat Saber map folders to upload-ready ZIP archives."""
from __future__ import annotations

import os
import zipfile
from pathlib import Path


def _is_temp_file(path: Path) -> bool:
    name = path.name.lower()
    return name.endswith(".tmp") or name.endswith(".partial") or ".tmp-" in name


def export_map_zip(map_folder: Path, out_zip: Path) -> Path:
    """Package all map files in `map_folder` flat at the root of `out_zip`."""
    map_folder = Path(map_folder)
    out_zip = Path(out_zip)
    if not map_folder.is_dir():
        raise ValueError(f"map folder not found: {map_folder}")

    has_info = any(
        p.is_file() and p.name.lower() == "info.dat"
        for p in map_folder.iterdir()
    )
    if not has_info:
        raise ValueError(f"Info.dat not found in {map_folder}")

    out_zip.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_zip.with_name(out_zip.name + ".tmp")
    try:
        with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for item in sorted(map_folder.iterdir(), key=lambda p: p.name):
                if item.is_symlink() or not item.is_file() or _is_temp_file(item):
                    continue
                zf.write(item, arcname=item.name)
        os.replace(tmp, out_zip)
    except BaseException:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass
        raise
    return out_zip
