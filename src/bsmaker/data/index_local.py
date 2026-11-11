"""Index a folder of Beat Saber maps into the manifest."""
from __future__ import annotations

import json
import math
import sqlite3
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from bsmaker.audio.decode import FfmpegMissingError
from bsmaker.data.filters import nps, reject_reasons
from bsmaker.data.manifest import upsert_difficulty, upsert_map
from bsmaker.data.split import assign_split
from bsmaker.formats.events import to_events
from bsmaker.formats.reader import map_key, read_map
from bsmaker.playability.checker import check
from bsmaker.playability.config import PlayabilityConfig

_PROBE_TIMEOUT_S = 30


@dataclass
class IndexSummary:
    n_maps: int = 0
    n_difficulties: int = 0
    n_accepted: int = 0
    reasons: dict[str, int] = field(default_factory=dict)
    n_duplicate_keys: int = 0


def probe_duration(path: Path | None) -> float | None:
    """Audio duration in seconds via ffprobe; None when the file is unusable (input is untrusted).

    Raises FfmpegMissingError when ffprobe itself cannot be run.
    """
    if path is None:
        return None
    cmd = ["ffprobe", "-v", "error", "-protocol_whitelist", "file,pipe", "-show_entries",
           "format=duration", "-of", "json", "-i", str(path)]
    try:
        proc = subprocess.run(cmd, capture_output=True, timeout=_PROBE_TIMEOUT_S, shell=False,
                              stdin=subprocess.DEVNULL)
        if proc.returncode != 0:
            return None
        value = float(json.loads(proc.stdout.decode("utf-8", "replace"))["format"]["duration"])
    except FileNotFoundError as exc:
        raise FfmpegMissingError("ffprobe not found on PATH") from exc
    except (OSError, subprocess.SubprocessError, ValueError, KeyError, TypeError):
        return None
    return value if math.isfinite(value) and value > 0 else None


def _mark_unreadable(conn, key: str) -> None:
    for r in conn.execute("SELECT characteristic, rank, reject_reasons FROM difficulties "
                          "WHERE map_key = ?", (key,)).fetchall():
        reasons = json.loads(r["reject_reasons"] or "[]")
        if "unreadable" not in reasons:
            reasons.append("unreadable")
        conn.execute("UPDATE difficulties SET accepted = 0, reject_reasons = ? "
                     "WHERE map_key = ? AND characteristic = ? AND rank = ?",
                     (json.dumps(reasons), key, r["characteristic"], r["rank"]))
    conn.commit()


def _index_one(conn, folder: Path, key: str, cfg: PlayabilityConfig,
               source: str) -> tuple[int, int, dict[str, int]]:
    """Index one folder in a single transaction; returns (n_difficulties, n_accepted, reasons).

    The slow work (parsing, ffprobe, checking) happens before the first write, so the write
    lock is held only briefly. May raise; the caller rolls back.
    """
    bm = read_map(folder)
    duration = probe_duration(bm.audio_path)
    row = conn.execute("SELECT bs_automapper FROM maps WHERE key = ?", (key,)).fetchone()
    automapper = None if row is None or row["bs_automapper"] is None else bool(row["bs_automapper"])
    split = assign_split(key)
    diff_rows: list[dict] = []
    n_acc = 0
    counts: dict[str, int] = {}
    for d in bm.difficulties:
        events, merged = to_events(d.notes)
        hard_rate = check(events, cfg).hard_rate
        reasons = reject_reasons(bm, d, duration_s=duration, hard_rate=hard_rate,
                                 automapper=automapper)
        diff_rows.append({
            "map_key": key, "characteristic": d.characteristic, "rank": d.rank,
            "schema": d.schema, "n_notes": len(d.notes), "n_events": len(events),
            "merged_notes": merged, "nps": nps(d.notes), "njs": d.njs,
            "njs_offset": d.njs_offset, "hard_rate": hard_rate,
            "accepted": 0 if reasons else 1, "reject_reasons": json.dumps(reasons),
            "split": split,
        })
        if reasons:
            for r in reasons:
                counts[r] = counts.get(r, 0) + 1
        else:
            n_acc += 1
    current = {(d.characteristic, d.rank) for d in bm.difficulties}
    with conn:  # one transaction: commits on success, rolls back on any exception
        for r in conn.execute("SELECT characteristic, rank FROM difficulties WHERE map_key = ?",
                              (key,)).fetchall():
            if (r["characteristic"], r["rank"]) not in current:
                conn.execute("DELETE FROM difficulties WHERE map_key = ? AND characteristic = ? "
                             "AND rank = ?", (key, r["characteristic"], r["rank"]))
        upsert_map(conn, {
            "key": key, "source": source, "folder": str(folder), "title": bm.title,
            "artist": bm.artist, "mapper": bm.mapper, "bpm": bm.bpm, "duration_s": duration,
            "requirements": json.dumps(list(bm.requirements)),
            "suggestions": json.dumps(list(bm.suggestions)),
            "audio_ok": 0 if duration is None else 1,
        }, commit=False)
        for dr in diff_rows:
            upsert_difficulty(conn, dr, commit=False)
    return len(diff_rows), n_acc, counts


def index_folder(conn, root: Path, cfg: PlayabilityConfig, source: str = "local",
                 progress_every: int = 0, only_keys: set[str] | None = None) -> IndexSummary:
    """Index every map folder under `root` (or only those whose key is in `only_keys`).

    sqlite3 errors and a missing ffprobe propagate: they are not verdicts on a map.
    """
    summary = IndexSummary()
    seen: set[str] = set()
    folders = sorted(p for p in Path(root).iterdir() if p.is_dir())
    for i, folder in enumerate(folders, 1):
        if progress_every and i % progress_every == 0:
            print(f"  indexed {i}/{len(folders)} folders", flush=True)
        key = map_key(folder.name)
        if only_keys is not None and key not in only_keys:
            continue
        if key in seen:
            summary.n_duplicate_keys += 1
            continue
        seen.add(key)
        summary.n_maps += 1
        try:
            n_d, n_acc, reasons = _index_one(conn, folder, key, cfg, source)
        except (sqlite3.Error, FfmpegMissingError):
            raise  # environment or database trouble, not a bad map
        except Exception:  # noqa: BLE001 - one bad folder must never stop the run
            summary.reasons["unreadable"] = summary.reasons.get("unreadable", 0) + 1
            _mark_unreadable(conn, key)
            continue
        summary.n_difficulties += n_d
        summary.n_accepted += n_acc
        for r, n in reasons.items():
            summary.reasons[r] = summary.reasons.get(r, 0) + n
    return summary
