"""Log-mel feature cache for every map that has an accepted difficulty."""
from __future__ import annotations

import json
from pathlib import Path

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


def build_features(conn, features_dir: Path, progress_every: int = 0) -> tuple[int, int]:
    """Write features_dir/<key>.npy per map with an accepted difficulty; return (computed, failed)."""
    features_dir = Path(features_dir)
    keys = [r["map_key"] for r in conn.execute(
        "SELECT DISTINCT map_key FROM difficulties WHERE accepted = 1 ORDER BY map_key")]
    computed = failed = 0
    for i, key in enumerate(keys, 1):
        if progress_every and i % progress_every == 0:
            print(f"  features {i}/{len(keys)}", flush=True)
        folder = conn.execute("SELECT folder FROM maps WHERE key = ?", (key,)).fetchone()["folder"]
        try:
            audio = read_map(Path(folder)).audio_path
            if audio is None:
                raise FileNotFoundError("no audio file")
            cached_log_mel(audio, features_dir / f"{key}.npy")
            computed += 1
        except Exception:  # noqa: BLE001 - a decode failure must not stop the run
            _mark_no_audio(conn, key)
            failed += 1
    return computed, failed
