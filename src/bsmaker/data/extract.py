"""Safe extraction of downloaded map archives (untrusted data)."""
from __future__ import annotations

import re
import lzma
import shutil
import zipfile
import zlib
from pathlib import Path

MAX_ENTRY_BYTES = 60 * 2**20
MAX_TOTAL_BYTES = 150 * 2**20
MAX_DAT_BYTES = 20 * 2**20
MAX_ENTRIES = 200
AUDIO_EXTS = (".egg", ".ogg", ".wav")
_RESERVED = re.compile(r"^(con|prn|aux|nul|com[1-9]|lpt[1-9])$", re.IGNORECASE)
_CHUNK = 1 << 16


class UnsafeArchiveError(Exception):
    """The archive is corrupt, too large or otherwise not safe to extract."""


def _wanted_name(entry_name: str) -> str | None:
    base = entry_name.replace("\\", "/").rsplit("/", 1)[-1]
    if not base or base.startswith(".") or ":" in base or base[-1] in ". ":
        return None
    low = base.lower()
    if not (low.endswith(".dat") or low.endswith(AUDIO_EXTS)):
        return None
    if _RESERVED.match(base.split(".", 1)[0]) or _RESERVED.match(Path(base).stem):
        return None
    return base


def _limit(base: str) -> int:
    return MAX_DAT_BYTES if base.lower().endswith(".dat") else MAX_ENTRY_BYTES


def safe_extract(zip_path: Path, dest: Path) -> list[Path]:
    """Extract .dat and audio entries flat into `dest`; raise UnsafeArchiveError on any problem."""
    dest = Path(dest)
    if dest.exists() and (not dest.is_dir() or any(dest.iterdir())):
        raise UnsafeArchiveError(f"destination {dest} is not empty")
    created_dest = not dest.exists()
    written: list[Path] = []
    try:
        with zipfile.ZipFile(zip_path) as zf:
            plan: list[tuple[zipfile.ZipInfo, str]] = []
            seen: set[str] = set()
            for info in zf.infolist():
                if info.is_dir():
                    continue
                base = _wanted_name(info.filename)
                if base is None or base.lower() in seen:
                    continue
                if info.file_size > _limit(base):
                    raise UnsafeArchiveError(f"entry {base!r} declares {info.file_size} bytes")
                seen.add(base.lower())
                if len(seen) > MAX_ENTRIES:
                    raise UnsafeArchiveError(f"more than {MAX_ENTRIES} wanted entries")
                plan.append((info, base))
            dest.mkdir(parents=True, exist_ok=True)
            total = 0
            for info, base in plan:
                target = dest / base
                size = 0
                with zf.open(info) as src, target.open("xb") as out:
                    written.append(target)
                    while chunk := src.read(_CHUNK):
                        size += len(chunk)
                        total += len(chunk)
                        if size > _limit(base):
                            raise UnsafeArchiveError(f"entry {base!r} exceeds {_limit(base)} bytes")
                        if total > MAX_TOTAL_BYTES:
                            raise UnsafeArchiveError(f"archive exceeds {MAX_TOTAL_BYTES} bytes")
                        out.write(chunk)
    except UnsafeArchiveError:
        _cleanup(dest, written, created_dest)
        raise
    except (zipfile.BadZipFile, zipfile.LargeZipFile, OSError, EOFError, RuntimeError,
            NotImplementedError, ValueError, zlib.error, lzma.LZMAError) as exc:
        _cleanup(dest, written, created_dest)
        raise UnsafeArchiveError(f"cannot extract {zip_path}: {type(exc).__name__}: {exc}") from exc
    return written


def _cleanup(dest: Path, written: list[Path], created_dest: bool) -> None:
    if created_dest:
        shutil.rmtree(dest, ignore_errors=True)
        return
    for p in written:
        p.unlink(missing_ok=True)
