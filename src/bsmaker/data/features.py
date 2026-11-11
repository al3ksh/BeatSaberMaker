"""Log-mel feature cache for every map that has an accepted difficulty."""
from __future__ import annotations

import json
import shutil
import sqlite3
from collections.abc import Callable
from pathlib import Path

from bsmaker.audio.decode import AudioDecodeError
from bsmaker.audio.mel import cached_log_mel
from bsmaker.formats.reader import read_map


def _mark_no_audio(conn, key: str) -> None:
    conn.execute("UPDATE maps SET audio_ok = 0 WHERE key = ?", (key,))
    rows = conn.execute("SELECT characteristic, rank, reject_reasons FROM difficulties "
                        "WHERE map_key = ?", (key,)).fetchall()
    for r in rows:
        reasons = json.loads(r["reject_reasons"] or "[]")
        if "no_audio" not in reasons:
            reasons.append("no_audio")
        conn.execute("UPDATE difficulties SET accepted = 0, reject_reasons = ? "
                     "WHERE map_key = ? AND characteristic = ? AND rank = ?",
                     (json.dumps(reasons), key, r["characteristic"], r["rank"]))
    conn.commit()


MIN_FREE_BYTES = 10 * 2**30


def _default_free(path: Path) -> int:
    return shutil.disk_usage(path).free


def build_features(conn, features_dir: Path, progress_every: int = 0,
                   free_bytes: Callable[[Path], int] = _default_free) -> tuple[int, int]:
    """Write features_dir/<key>.npy per map with an accepted difficulty; return (computed, failed).

    Only a failed decode of that map's audio marks it `no_audio`. Environment errors (missing
    ffmpeg, OSError such as a full disk, sqlite3.Error) propagate. The run stops with a message
    when free space in `features_dir` drops below 10 GB.
    """
    features_dir = Path(features_dir)
    features_dir.mkdir(parents=True, exist_ok=True)
    # fetchall: do not hold a read transaction open across slow decoding
    todo = conn.execute(
        "SELECT DISTINCT d.map_key AS key, m.folder AS folder FROM difficulties d "
        "JOIN maps m ON m.key = d.map_key WHERE d.accepted = 1 ORDER BY d.map_key").fetchall()
    computed = failed = 0
    for i, row in enumerate(todo, 1):
        if progress_every and i % progress_every == 0:
            print(f"  features {i}/{len(todo)}", flush=True)
        if free_bytes(features_dir) < MIN_FREE_BYTES:
            print(f"stopping: free space in {features_dir} is below "
                  f"{MIN_FREE_BYTES // 2**30} GB; {len(todo) - i + 1} maps left", flush=True)
            break
        key = row["key"]
        try:
            audio = read_map(Path(row["folder"])).audio_path
        except sqlite3.Error:
            raise
        except Exception:  # noqa: BLE001 - this one map is unreadable now: record it, go on
            audio = None
        if audio is None or not Path(audio).is_file():
            _mark_no_audio(conn, key)
            failed += 1
            continue
        try:
            cached_log_mel(audio, features_dir / f"{key}.npy")
        except AudioDecodeError:
            _mark_no_audio(conn, key)
            failed += 1
        else:
            computed += 1
    return computed, failed
