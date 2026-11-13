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
from bsmaker.data.manifest import get_meta, upsert_difficulty, upsert_map
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
    except OSError:
        if not Path(path).is_file():
            return None  # the input file itself is missing or unreadable
        raise  # e.g. PermissionError launching ffprobe: an environment problem
    except (subprocess.SubprocessError, ValueError, KeyError, TypeError):
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


def _stored_threshold(conn) -> float | None:
    raw = get_meta(conn, "align_threshold")
    if raw is None:
        return None
    try:
        return float(raw)
    except ValueError:
        raise ValueError(f"manifest meta 'align_threshold' is not a number: {raw!r}") from None


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
    stored = list(conn.execute(
        "SELECT characteristic, rank, split, align_score FROM difficulties WHERE map_key = ?",
        (key,)))
    existing = {(r["characteristic"], r["rank"]): r["split"] for r in stored}
    align = {(r["characteristic"], r["rank"]): r["align_score"] for r in stored}
    threshold = _stored_threshold(conn)
    # keep the split of a re-indexed map (set by `data resplit`); new maps get a per-key
    # placeholder until the next resplit
    map_split = next((s for s in existing.values() if s), None) or assign_split(key)
    diff_rows: list[dict] = []
    n_acc = 0
    counts: dict[str, int] = {}
    for d in bm.difficulties:
        events, merged = to_events(d.notes)
        hard_rate = check(events, cfg).hard_rate
        score = align.get((d.characteristic, d.rank))
        verified = score is not None and threshold is not None and score >= threshold
        reasons = reject_reasons(bm, d, duration_s=duration, hard_rate=hard_rate,
                                 automapper=automapper, tempo_verified=verified)
        diff_rows.append({
            "map_key": key, "characteristic": d.characteristic, "rank": d.rank,
            "schema": d.schema, "n_notes": len(d.notes), "n_events": len(events),
            "merged_notes": merged, "nps": nps(d.notes), "njs": d.njs,
            "njs_offset": d.njs_offset, "tempo_kind": d.flags.tempo_kind, "hard_rate": hard_rate,
            "accepted": 0 if reasons else 1, "reject_reasons": json.dumps(reasons),
            "split": existing.get((d.characteristic, d.rank)) or map_split,
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
    _stored_threshold(conn)  # fail early on a corrupt threshold, outside the per-map handler
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


_TEMPO_REASONS = ("bpm_changes", "tempo_external", "bpm_trick", "tempo_unverified")


def reindex_tempo(conn, cfg: PlayabilityConfig) -> dict[str, int]:
    """Re-index only maps with a tempo reject reason (old `bpm_changes` or a new one).

    Goes through `_index_one`, so splits and map columns are kept. Folders that are missing
    or unreadable are counted under "skipped" and their rows are left as they are.
    Returns counts: "maps", "skipped" and one entry per tempo reason of the new rows.
    """
    _stored_threshold(conn)  # fail early on a corrupt threshold, outside the per-map handler
    where = " OR ".join("d.reject_reasons LIKE ?" for _ in _TEMPO_REASONS)
    maps = conn.execute(
        f"SELECT DISTINCT m.key, m.folder, m.source FROM maps m JOIN difficulties d "
        f"ON d.map_key = m.key WHERE {where} ORDER BY m.key",
        [f'%"{r}"%' for r in _TEMPO_REASONS]).fetchall()
    out = {"maps": 0, "skipped": 0, **{r: 0 for r in _TEMPO_REASONS[1:]}}
    for m in maps:
        folder = Path(m["folder"]) if m["folder"] else None
        if folder is None or not folder.is_dir():
            out["skipped"] += 1
            continue
        try:
            _, _, reasons = _index_one(conn, folder, m["key"], cfg, m["source"] or "local")
        except (sqlite3.Error, FfmpegMissingError):
            raise
        except Exception:  # noqa: BLE001 - one bad folder must never stop the run
            out["skipped"] += 1
            continue
        out["maps"] += 1
        for r in _TEMPO_REASONS[1:]:
            out[r] += reasons.get(r, 0)
    return out
