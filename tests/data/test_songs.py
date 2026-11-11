from bsmaker.data.manifest import open_manifest, upsert_difficulty, upsert_map
from bsmaker.data.songs import assign_song_ids, normalise_title, resplit
from bsmaker.data.split import assign_split


def test_normalise_title():
    assert normalise_title("Odo (TV Size) [Full]") == "odo"
    assert normalise_title("ＯＤＯ") == "odo"
    assert normalise_title("うっせぇわ") != ""
    assert normalise_title("!!!") == ""


def test_same_title_and_artist_grouped():
    ids = assign_song_ids([("b", "Odo", "Ado", 100.0), ("a", "ODO!", "ado", 300.0)])
    assert ids["a"] == ids["b"] == "a"


def test_same_title_close_duration_grouped():
    ids = assign_song_ids([("a", "Odo", "X", 180.0), ("b", "Odo", "Y", 181.5)])
    assert ids["a"] == ids["b"]
    ids = assign_song_ids([("a", "Odo", "X", 180.0), ("b", "Odo", "Y", 190.0)])
    assert ids["a"] != ids["b"]


def test_empty_title_is_own_song():
    ids = assign_song_ids([("a", "!!!", "X", 100.0), ("b", "???", "X", 100.0)])
    assert ids["a"] == "a" and ids["b"] == "b"


def test_grouping_is_transitive():
    # a~b by duration, b~c by title+artist (durations far apart), so a,b,c are one song
    ids = assign_song_ids([("a", "Odo", "P", 100.0), ("b", "Odo", "Q", 101.5),
                           ("c", "Odo", "Q", 200.0)])
    assert len({ids["a"], ids["b"], ids["c"]}) == 1


def test_null_duration_joins_only_title_artist_group():
    ids = assign_song_ids([("a", "Odo", "X", None), ("b", "Odo", "Y", 100.0),
                           ("c", "Odo", "X", 500.0)])
    assert ids["a"] == ids["c"] and ids["b"] == "b"


def test_song_id_is_smallest_key():
    ids = assign_song_ids([("c", "T", "A", 1.0), ("a", "T", "A", 1.0), ("b", "T", "A", 1.0)])
    assert set(ids.values()) == {"a"}


def _fill(conn):
    for key, title in [("k1", "Song"), ("k2", "Song"), ("k3", "Song"), ("z9", "Other")]:
        upsert_map(conn, {"key": key, "title": title, "artist": "A", "duration_s": 100.0})
        for rank in ("Easy", "Expert"):
            upsert_difficulty(conn, {"map_key": key, "characteristic": "Standard", "rank": rank,
                                     "accepted": 1, "split": "train", "reject_reasons": "[]"})


def test_resplit_puts_all_difficulties_of_a_song_in_one_split(tmp_path):
    conn = open_manifest(tmp_path / "m.sqlite")
    _fill(conn)
    counts = resplit(conn)
    rows = conn.execute("SELECT m.song_id, d.split FROM difficulties d JOIN maps m "
                        "ON m.key = d.map_key").fetchall()
    assert {r["song_id"] for r in rows if r["song_id"] == "k1"} == {"k1"}
    assert {r["split"] for r in rows if r["song_id"] == "k1"} == {assign_split("k1")}
    assert sum(1 for r in rows if r["song_id"] == "k1") == 6
    assert sum(counts.values()) == 8


def test_resplit_is_idempotent(tmp_path):
    conn = open_manifest(tmp_path / "m.sqlite")
    _fill(conn)
    a = resplit(conn)
    snap = [tuple(r) for r in conn.execute("SELECT map_key, rank, split FROM difficulties ORDER BY 1,2")]
    assert resplit(conn) == a
    assert snap == [tuple(r) for r in conn.execute(
        "SELECT map_key, rank, split FROM difficulties ORDER BY 1,2")]
