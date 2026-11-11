"""Log-mel feature cache for every map that has an accepted difficulty."""
from __future__ import annotations

import json
import os
import shutil
import sqlite3
from collections.abc import Callable
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path

from bsmaker.audio.decode import AudioDecodeError, require_ffmpeg
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


def _worker_init() -> None:
    # numpy/librosa are already imported at module load, so these are best-effort hints for
    # libraries that read them lazily; the pool itself provides the parallelism.
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[name] = "1"


def _compute_one(audio: str, cache: str, key: str) -> tuple[str, str, str]:
    """Worker: decode + mel + write one .npy. Never touches the manifest.

    A bad audio file is a result ("no_audio"); every other exception propagates to the parent.
    """
    try:
        cached_log_mel(Path(audio), Path(cache))
    except AudioDecodeError as e:
        return key, "no_audio", str(e)
    return key, "ok", ""


def build_features(conn, features_dir: Path, progress_every: int = 0,
                   free_bytes: Callable[[Path], int] = _default_free,
                   workers: int = 1) -> tuple[int, int]:
    """Write features_dir/<key>.npy per map with an accepted difficulty; return (computed, failed).

    Only a failed decode of that map's audio marks it `no_audio`. Environment errors (missing
    ffmpeg, OSError such as a full disk, sqlite3.Error) propagate. The run stops with a message
    when free space in `features_dir` drops below 10 GB. With `workers > 1` decoding runs in
    worker processes; only this process touches the manifest.
    """
    features_dir = Path(features_dir)
    features_dir.mkdir(parents=True, exist_ok=True)
    # fetchall: do not hold a read transaction open across slow decoding
    todo = conn.execute(
        "SELECT DISTINCT d.map_key AS key, m.folder AS folder FROM difficulties d "
        "JOIN maps m ON m.key = d.map_key WHERE d.accepted = 1 ORDER BY d.map_key").fetchall()
    if workers > 1:
        return _build_parallel(conn, features_dir, todo, progress_every, free_bytes, workers)
    computed = failed = 0
    for i, row in enumerate(todo, 1):
        if progress_every and i % progress_every == 0:
            print(f"  features {i}/{len(todo)}", flush=True)
        if free_bytes(features_dir) < MIN_FREE_BYTES:
            print(f"stopping: free space in {features_dir} is below "
                  f"{MIN_FREE_BYTES // 2**30} GB; {len(todo) - i + 1} maps left", flush=True)
            break
        key = row["key"]
        audio = _audio_of(row)
        if audio is None:
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


def _audio_of(row) -> Path | None:
    try:
        audio = read_map(Path(row["folder"])).audio_path
    except sqlite3.Error:
        raise
    except Exception:  # noqa: BLE001 - this one map is unreadable now: record it, go on
        return None
    if audio is None or not Path(audio).is_file():
        return None
    return Path(audio)


def _build_parallel(conn, features_dir: Path, todo, progress_every: int,
                    free_bytes: Callable[[Path], int], workers: int) -> tuple[int, int]:
    require_ffmpeg()
    computed = failed = done = 0

    def record(fut) -> None:
        nonlocal computed, failed, done
        key, status, _msg = fut.result()  # worker exceptions (environment errors) propagate
        if status == "ok":
            computed += 1
        else:
            _mark_no_audio(conn, key)
            failed += 1
        done += 1
        if progress_every and done % progress_every == 0:
            print(f"  features {done}/{len(todo)}", flush=True)

    pending: set = set()
    executor = ProcessPoolExecutor(max_workers=workers, initializer=_worker_init)
    try:
        for i, row in enumerate(todo, 1):
            if free_bytes(features_dir) < MIN_FREE_BYTES:
                print(f"stopping: free space in {features_dir} is below "
                      f"{MIN_FREE_BYTES // 2**30} GB; {len(todo) - i + 1} maps left", flush=True)
                break
            key = row["key"]
            audio = _audio_of(row)
            if audio is None:
                _mark_no_audio(conn, key)
                failed += 1
                done += 1
                continue
            pending.add(executor.submit(_compute_one, str(audio),
                                        str(features_dir / f"{key}.npy"), key))
            while len(pending) >= workers * 4:
                finished, pending = wait(pending, return_when=FIRST_COMPLETED)
                for fut in finished:
                    record(fut)
        while pending:  # in-flight work finishes and is recorded, also after a low-disk stop
            finished, pending = wait(pending, return_when=FIRST_COMPLETED)
            for fut in finished:
                record(fut)
    except BaseException:
        executor.shutdown(wait=True, cancel_futures=True)
        raise
    executor.shutdown()
    return computed, failed
