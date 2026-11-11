"""Group maps of the same song and split the dataset by song, not by map."""
from __future__ import annotations

import re
import sqlite3
import unicodedata
from collections.abc import Sequence

from bsmaker.data.split import assign_split

_BRACKETS = re.compile(r"\([^)]*\)|\[[^\]]*\]")
_DURATION_TOLERANCE_S = 2.0


def normalise_title(text: str) -> str:
    """NFKC, casefold, drop (...) and [...] parts, keep only Unicode letters and digits."""
    text = _BRACKETS.sub("", unicodedata.normalize("NFKC", text or "").casefold())
    return "".join(c for c in text if c.isalnum())


def assign_song_ids(rows: Sequence[tuple[str, str, str, float | None]]) -> dict[str, str]:
    """Map every key to its song id (the smallest key of its group).

    Two maps are the same song when title and artist both normalise equal, or when the
    normalised titles are equal and the durations are within 2 s (chained, so transitive).
    A map with an empty normalised title is its own song; a NULL duration joins only the
    title+artist group.
    """
    parent: dict[str, str] = {key: key for key, *_ in rows}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    by_pair: dict[tuple[str, str], str] = {}
    by_title: dict[str, list[tuple[float, str]]] = {}
    for key, title, artist, duration in rows:
        nt = normalise_title(title)
        if not nt:
            continue
        first = by_pair.setdefault((nt, normalise_title(artist)), key)
        union(first, key)
        if duration is not None:
            by_title.setdefault(nt, []).append((duration, key))
    for bucket in by_title.values():
        bucket.sort()
        for (d0, k0), (d1, k1) in zip(bucket, bucket[1:]):
            if d1 - d0 <= _DURATION_TOLERANCE_S:
                union(k0, k1)
    return {key: find(key) for key in parent}


def _split_counts(conn: sqlite3.Connection) -> dict[str, int]:
    return {r[0]: r[1] for r in conn.execute(
        "SELECT split, COUNT(*) FROM difficulties GROUP BY split")}


def resplit(conn: sqlite3.Connection) -> dict[str, int]:
    """Set maps.song_id and every difficulties.split = assign_split(song_id); returns counts."""
    rows = [(r["key"], r["title"] or "", r["artist"] or "", r["duration_s"])
            for r in conn.execute("SELECT key, title, artist, duration_s FROM maps")]
    ids = assign_song_ids(rows)
    with conn:
        conn.executemany("UPDATE maps SET song_id = ? WHERE key = ?",
                         [(sid, key) for key, sid in ids.items()])
        conn.executemany("UPDATE difficulties SET split = ? WHERE map_key = ?",
                         [(assign_split(sid), key) for key, sid in ids.items()])
    counts = {"train": 0, "val": 0, "test": 0}
    counts.update(_split_counts(conn))
    return counts
