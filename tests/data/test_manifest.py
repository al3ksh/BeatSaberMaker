import sqlite3

from bsmaker.data.manifest import iter_accepted, open_manifest, upsert_difficulty, upsert_map


def _diff(key, rank="Expert", accepted=1, split="train"):
    return {"map_key": key, "characteristic": "Standard", "rank": rank, "accepted": accepted,
            "split": split, "reject_reasons": "[]"}


def test_creates_tables(tmp_path):
    conn = open_manifest(tmp_path / "sub" / "m.sqlite")
    names = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert {"maps", "difficulties"} <= names
    assert conn.row_factory is sqlite3.Row


def test_upsert_is_idempotent(tmp_path):
    conn = open_manifest(tmp_path / "m.sqlite")
    upsert_map(conn, {"key": "a1", "source": "local", "title": "One"})
    upsert_map(conn, {"key": "a1", "source": "local", "title": "Two"})
    rows = conn.execute("SELECT * FROM maps").fetchall()
    assert len(rows) == 1 and rows[0]["title"] == "Two"
    upsert_difficulty(conn, _diff("a1"))
    upsert_difficulty(conn, _diff("a1", accepted=0))
    rows = conn.execute("SELECT * FROM difficulties").fetchall()
    assert len(rows) == 1 and rows[0]["accepted"] == 0


def test_upsert_keeps_unlisted_columns(tmp_path):
    conn = open_manifest(tmp_path / "m.sqlite")
    upsert_map(conn, {"key": "a1", "source": "local", "bs_score": 0.9})
    upsert_map(conn, {"key": "a1", "title": "T"})
    r = conn.execute("SELECT * FROM maps").fetchone()
    assert r["bs_score"] == 0.9 and r["title"] == "T"


def test_iter_accepted_filters_by_split(tmp_path):
    conn = open_manifest(tmp_path / "m.sqlite")
    upsert_difficulty(conn, _diff("a", split="train"))
    upsert_difficulty(conn, _diff("b", split="val"))
    upsert_difficulty(conn, _diff("c", split="train", accepted=0))
    assert {r["map_key"] for r in iter_accepted(conn)} == {"a", "b"}
    assert [r["map_key"] for r in iter_accepted(conn, "val")] == ["b"]


def test_open_manifest_uses_wal_and_long_timeout(tmp_path):
    conn = open_manifest(tmp_path / "m.sqlite")
    assert conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    assert conn.execute("PRAGMA synchronous").fetchone()[0] == 1  # NORMAL


def test_open_reader_cursor_does_not_block_a_committing_writer(tmp_path):
    path = tmp_path / "m.sqlite"
    reader = open_manifest(path)
    for i in range(5):
        upsert_map(reader, {"key": f"a{i}", "source": "local"})
    cur = reader.execute("SELECT key FROM maps ORDER BY key")
    cur.fetchone()  # cursor stays open, as in a slow row-by-row loop
    writer = sqlite3.connect(path, timeout=0.5)
    writer.execute("INSERT INTO maps (key, source) VALUES ('zz', 'local')")
    writer.commit()  # must not raise "database is locked"
    writer.close()
    assert len(cur.fetchall()) == 4


def test_upsert_without_commit_can_be_rolled_back(tmp_path):
    conn = open_manifest(tmp_path / "m.sqlite")
    upsert_map(conn, {"key": "a1", "source": "local"}, commit=False)
    upsert_difficulty(conn, _diff("a1"), commit=False)
    conn.rollback()
    assert conn.execute("SELECT COUNT(*) FROM maps").fetchone()[0] == 0
    assert conn.execute("SELECT COUNT(*) FROM difficulties").fetchone()[0] == 0
