"""SQLite manifest of training maps and their difficulties."""
from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from pathlib import Path

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
"""


def open_manifest(path: Path) -> sqlite3.Connection:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.executescript(_SCHEMA)
    conn.commit()
    return conn


def _columns(conn: sqlite3.Connection, table: str) -> set[str]:
    return {r[1] for r in conn.execute(f"PRAGMA table_info({table})")}


def _upsert(conn: sqlite3.Connection, table: str, pk: tuple[str, ...], row: dict) -> None:
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
    conn.commit()


def upsert_map(conn: sqlite3.Connection, row: dict) -> None:
    """Insert, or update only the columns present in `row`."""
    _upsert(conn, "maps", ("key",), row)


def upsert_difficulty(conn: sqlite3.Connection, row: dict) -> None:
    _upsert(conn, "difficulties", ("map_key", "characteristic", "rank"), row)


def iter_accepted(conn: sqlite3.Connection, split: str | None = None) -> Iterator[sqlite3.Row]:
    sql = "SELECT * FROM difficulties WHERE accepted = 1"
    args: tuple = ()
    if split is not None:
        sql += " AND split = ?"
        args = (split,)
    yield from conn.execute(sql + " ORDER BY map_key, characteristic, rank", args).fetchall()
