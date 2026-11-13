import json
import shutil
from pathlib import Path

from bsmaker.data.index_local import index_folder, reindex_tempo
from bsmaker.data.manifest import open_manifest, set_meta
from bsmaker.playability.config import PlayabilityConfig

FIX = Path(__file__).resolve().parents[1] / "fixtures" / "maps"
CFG = PlayabilityConfig()


def _v3_events(dst: Path) -> None:
    shutil.copytree(FIX / "v3_basic", dst)
    p = dst / "ExpertStandard.dat"
    d = json.loads(p.read_text("utf-8"))
    d["bpmEvents"] = [{"b": 0, "m": 120}, {"b": 8, "m": 90}]
    p.write_text(json.dumps(d), "utf-8")


def _setup(tmp_path):
    root = tmp_path / "levels"
    root.mkdir()
    shutil.copytree(FIX / "v2_basic", root / "1f00 (Const)")
    shutil.copytree(FIX / "v2_bpmchanges", root / "2f00 (Editor)")
    _v3_events(root / "3f00 (Events)")
    conn = open_manifest(tmp_path / "m.sqlite")
    index_folder(conn, root, CFG)
    return root, conn


def _reasons(conn, key):
    return json.loads(conn.execute("SELECT reject_reasons FROM difficulties WHERE map_key=?",
                                   (key,)).fetchone()[0])


def _row(conn, key):
    return tuple(conn.execute("SELECT * FROM difficulties WHERE map_key=?", (key,)).fetchone())


def _make_old(conn):
    conn.execute("UPDATE difficulties SET reject_reasons='[\"bpm_changes\"]', split='val' "
                 "WHERE map_key IN ('2f00','3f00')")
    conn.execute("UPDATE maps SET song_id='s2', bs_score=0.9 WHERE key='2f00'")
    conn.commit()


def test_reindex_tempo_converts_old_reason_and_keeps_split(tmp_path):
    root, conn = _setup(tmp_path)
    assert conn.execute("SELECT tempo_kind FROM difficulties WHERE map_key='3f00'"
                        ).fetchone()[0] == "events"
    _make_old(conn)
    const_before = _row(conn, "1f00")
    out = reindex_tempo(conn, CFG)
    assert out["maps"] == 2 and out["skipped"] == 0 and out["tempo_unverified"] == 2
    for key in ("2f00", "3f00"):
        r = _reasons(conn, key)
        assert "tempo_unverified" in r and "bpm_changes" not in r
        assert conn.execute("SELECT split FROM difficulties WHERE map_key=?",
                            (key,)).fetchone()[0] == "val"
    m = conn.execute("SELECT song_id, bs_score FROM maps WHERE key='2f00'").fetchone()
    assert tuple(m) == ("s2", 0.9)
    assert _row(conn, "1f00") == const_before
    # repeating is safe and picks the maps up again via the new reason
    assert reindex_tempo(conn, CFG)["maps"] == 2


def test_reindex_tempo_skips_missing_or_unreadable_folders(tmp_path):
    root, conn = _setup(tmp_path)
    _make_old(conn)
    shutil.rmtree(root / "2f00 (Editor)")
    (root / "3f00 (Events)" / "Info.dat").write_text("{not json", "utf-8")
    before = [_row(conn, "2f00"), _row(conn, "3f00")]
    out = reindex_tempo(conn, CFG)
    assert out["skipped"] == 2 and out["maps"] == 0
    assert [_row(conn, "2f00"), _row(conn, "3f00")] == before


def test_verified_by_stored_align_score_and_threshold(tmp_path):
    root, conn = _setup(tmp_path)

    def run(score, threshold):
        conn.execute("UPDATE difficulties SET align_score=? WHERE map_key='3f00'", (score,))
        conn.execute("DELETE FROM meta")
        if threshold is not None:
            set_meta(conn, "align_threshold", threshold)
        conn.commit()
        index_folder(conn, root, CFG, only_keys={"3f00"})
        return _reasons(conn, "3f00")

    assert "tempo_unverified" in run(0.9, None)  # no threshold: nothing is verified
    assert "tempo_unverified" in run(None, "0.5")  # no score
    assert "tempo_unverified" in run(0.4, "0.5")
    assert "tempo_unverified" not in run(0.5, "0.5")
    assert "tempo_unverified" not in run(0.9, "0.5")
