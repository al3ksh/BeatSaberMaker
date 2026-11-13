"""SQLite manifest of training maps and their difficulties."""
from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from pathlib import Path

SCHEMA_VERSION = 3

_SCHEMA = """
CREATE TABLE IF NOT EXISTS maps (
    key TEXT PRIMARY KEY, source TEXT, folder TEXT, title TEXT, artist TEXT, mapper TEXT,
    bpm REAL, duration_s REAL, requirements TEXT, suggestions TEXT, audio_ok INTEGER,
    bs_hash TEXT, bs_tags TEXT, bs_score REAL, bs_upvotes INTEGER, bs_ranked INTEGER,
    bs_bl_ranked INTEGER, bs_curated INTEGER, bs_automapper INTEGER,
    bs_fetched INTEGER DEFAULT 0, download_state TEXT
);
CREATE TABLE IF NOT EXISTS difficulties (
    map_key TEXT, characteristic TEXT, rank TEXT, schema TEXT, n_notes INTEGER,
    n_events INTEGER, merged_notes INTEGER, nps REAL, njs REAL, njs_offset REAL,
    hard_rate REAL, accepted INTEGER, reject_reasons TEXT, split TEXT,
    PRIMARY KEY (map_key, characteristic, rank)
);
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
"""


def open_manifest(path: Path) -> sqlite3.Connection:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, timeout=120)
    conn.row_factory = sqlite3.Row
    version = conn.execute("PRAGMA user_version").fetchone()[0]
    if version > SCHEMA_VERSION:
        conn.close()
        raise RuntimeError(f"manifest {path} has schema version {version}, "
                           f"this code supports {SCHEMA_VERSION}")
    # WAL: readers never block the writer and the writer never blocks readers.
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.executescript(_SCHEMA)
    if version < SCHEMA_VERSION:
        if "song_id" not in _columns(conn, "maps"):
            conn.execute("ALTER TABLE maps ADD COLUMN song_id TEXT")
        for col, typ in (("tempo_kind", "TEXT"), ("align_score", "REAL"),
                         ("align_score_const", "REAL")):
            if col not in _columns(conn, "difficulties"):
                conn.execute(f"ALTER TABLE difficulties ADD COLUMN {col} {typ}")
        conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
    conn.commit()
    return conn


def _columns(conn: sqlite3.Connection, table: str) -> set[str]:
    return {r[1] for r in conn.execute(f"PRAGMA table_info({table})")}


def _upsert(conn: sqlite3.Connection, table: str, pk: tuple[str, ...], row: dict,
            commit: bool = True) -> None:
    unknown = set(row) - _columns(conn, table)
    if unknown:
        raise ValueError(f"unknown columns for {table}: {sorted(unknown)}")
    cols = list(row)
    rest = [c for c in cols if c not in pk]
    if rest:
        action = "DO UPDATE SET " + ", ".join(f"{c}=excluded.{c}" for c in rest)
    else:
        action = "DO NOTHING"
    sql = (f"INSERT INTO {table} ({', '.join(cols)}) VALUES ({', '.join('?' * len(cols))}) "
           f"ON CONFLICT({', '.join(pk)}) {action}")
    conn.execute(sql, [row[c] for c in cols])
    if commit:
        conn.commit()


def upsert_map(conn: sqlite3.Connection, row: dict, commit: bool = True) -> None:
    """Insert, or update only the columns present in `row`.

    `commit=False` leaves the write in the caller's transaction (the caller commits).
    """
    _upsert(conn, "maps", ("key",), row, commit)


def upsert_difficulty(conn: sqlite3.Connection, row: dict, commit: bool = True) -> None:
    _upsert(conn, "difficulties", ("map_key", "characteristic", "rank"), row, commit)


def iter_accepted(conn: sqlite3.Connection, split: str | None = None) -> Iterator[sqlite3.Row]:
    sql = "SELECT * FROM difficulties WHERE accepted = 1"
    args: tuple = ()
    if split is not None:
        sql += " AND split = ?"
        args = (split,)
    yield from conn.execute(sql + " ORDER BY map_key, characteristic, rank", args).fetchall()


def get_meta(conn: sqlite3.Connection, key: str) -> str | None:
    row = conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
    return None if row is None else row[0]


def set_meta(conn: sqlite3.Connection, key: str, value: str) -> None:
    conn.execute("INSERT INTO meta (key, value) VALUES (?, ?) "
                 "ON CONFLICT(key) DO UPDATE SET value = excluded.value", (key, value))
    conn.commit()
