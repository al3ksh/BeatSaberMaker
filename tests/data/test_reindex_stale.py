import json
import shutil
from pathlib import Path

from bsmaker.data.index_local import index_folder
from bsmaker.data.manifest import open_manifest, upsert_difficulty
from bsmaker.playability.config import PlayabilityConfig

FIX = Path(__file__).resolve().parents[1] / "fixtures" / "maps"
CFG = PlayabilityConfig()


def _setup(tmp_path):
    root = tmp_path / "levels"
    root.mkdir()
    shutil.copytree(FIX / "v2_basic", root / "1f00 (X)")
    conn = open_manifest(tmp_path / "m.sqlite")
    index_folder(conn, root, CFG)
    return root, conn


def test_removed_difficulty_row_is_deleted(tmp_path):
    root, conn = _setup(tmp_path)
    assert conn.execute("SELECT COUNT(*) FROM difficulties").fetchone()[0] == 1
    (root / "1f00 (X)" / "ExpertStandard.dat").unlink()
    index_folder(conn, root, CFG)
    assert conn.execute("SELECT COUNT(*) FROM difficulties WHERE map_key='1f00'").fetchone()[0] == 0


def test_rows_of_maps_outside_root_are_kept(tmp_path):
    root, conn = _setup(tmp_path)
    upsert_difficulty(conn, {"map_key": "zzzz", "characteristic": "Standard", "rank": "Easy",
                             "accepted": 1, "reject_reasons": "[]", "split": "train"})
    index_folder(conn, root, CFG)
    assert conn.execute("SELECT accepted FROM difficulties WHERE map_key='zzzz'").fetchone()[0] == 1


def test_unreadable_map_rows_are_rejected(tmp_path):
    root, conn = _setup(tmp_path)
    conn.execute("UPDATE difficulties SET accepted=1, reject_reasons='[]'")
    conn.commit()
    (root / "1f00 (X)" / "Info.dat").write_text("{not json", "utf-8")
    index_folder(conn, root, CFG)
    rows = conn.execute("SELECT * FROM difficulties WHERE map_key='1f00'").fetchall()
    assert rows
    for r in rows:
        assert r["accepted"] == 0 and "unreadable" in json.loads(r["reject_reasons"])


def test_acceptance_flips_on_reindex(tmp_path):
    root, conn = _setup(tmp_path)
    conn.execute("UPDATE difficulties SET accepted=1, reject_reasons='[]'")
    conn.commit()
    index_folder(conn, root, CFG)
    r = conn.execute("SELECT * FROM difficulties").fetchone()
    assert r["accepted"] == 0 and "duration" in json.loads(r["reject_reasons"])
