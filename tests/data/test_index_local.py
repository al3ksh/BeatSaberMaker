import json
import shutil
from pathlib import Path

from bsmaker.data.features import build_features
from bsmaker.data.index_local import index_folder
from bsmaker.data.manifest import open_manifest, upsert_map
from bsmaker.playability.config import PlayabilityConfig

FIX = Path(__file__).resolve().parents[1] / "fixtures" / "maps"
CFG = PlayabilityConfig()


def _root(tmp_path):
    root = tmp_path / "levels"
    root.mkdir()
    for i, p in enumerate(sorted(FIX.iterdir())):
        shutil.copytree(p, root / f"{i + 1:x}a0 ({p.name})")
    return root


def _one(tmp_path, fixture="v2_basic", name="1f00 (X)"):
    root = tmp_path / "levels"
    root.mkdir()
    shutil.copytree(FIX / fixture, root / name)
    return root


def test_indexes_all_fixtures(tmp_path):
    root = _root(tmp_path)
    conn = open_manifest(tmp_path / "m.sqlite")
    s = index_folder(conn, root, CFG)
    assert s.n_maps == 8
    assert conn.execute("SELECT COUNT(*) FROM maps").fetchone()[0] == 8
    assert s.n_difficulties == conn.execute("SELECT COUNT(*) FROM difficulties").fetchone()[0]
    assert s.n_difficulties >= 8
    assert s.n_duplicate_keys == 0
    r = conn.execute("SELECT * FROM maps LIMIT 1").fetchone()
    assert r["audio_ok"] == 1 and abs(r["duration_s"] - 1.0) < 0.2
    assert r["source"] == "local"
    # one-second audio: every difficulty is rejected for duration
    assert s.n_accepted == 0 and s.reasons.get("duration", 0) > 0


def test_duplicate_key_counted_once(tmp_path):
    root = tmp_path / "levels"
    root.mkdir()
    shutil.copytree(FIX / "v2_basic", root / "abc1 (A)")
    shutil.copytree(FIX / "v2_basic", root / "abc1 (B)")
    conn = open_manifest(tmp_path / "m.sqlite")
    s = index_folder(conn, root, CFG)
    assert s.n_duplicate_keys == 1
    assert conn.execute("SELECT COUNT(*) FROM maps").fetchone()[0] == 1
    assert conn.execute("SELECT folder FROM maps").fetchone()[0].endswith("abc1 (A)")


def test_non_ascii_and_long_folder_names(tmp_path):
    root = _one(tmp_path, name="1f00 (ハルトマンの妖怪少女 - テスト " + "あ" * 80 + ")")
    conn = open_manifest(tmp_path / "m.sqlite")
    index_folder(conn, root, CFG)
    r = conn.execute("SELECT key, audio_ok FROM maps").fetchone()
    assert r["key"] == "1f00" and r["audio_ok"] == 1


def test_all_difficulties_of_a_map_share_split(tmp_path):
    root = _one(tmp_path)
    conn = open_manifest(tmp_path / "m.sqlite")
    index_folder(conn, root, CFG)
    splits = {r[0] for r in conn.execute("SELECT split FROM difficulties WHERE map_key='1f00'")}
    assert len(splits) == 1 and splits <= {"train", "val", "test"}


def test_reindex_does_not_duplicate_rows(tmp_path):
    root = _root(tmp_path)
    conn = open_manifest(tmp_path / "m.sqlite")
    index_folder(conn, root, CFG)
    a = [conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0] for t in ("maps", "difficulties")]
    index_folder(conn, root, CFG)
    b = [conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0] for t in ("maps", "difficulties")]
    assert a == b


def test_reindex_keeps_bs_columns(tmp_path):
    root = _one(tmp_path)
    conn = open_manifest(tmp_path / "m.sqlite")
    index_folder(conn, root, CFG)
    upsert_map(conn, {"key": "1f00", "bs_score": 0.77, "download_state": "done"})
    index_folder(conn, root, CFG)
    r = conn.execute("SELECT * FROM maps WHERE key='1f00'").fetchone()
    assert r["bs_score"] == 0.77 and r["download_state"] == "done"


def test_automapper_read_from_row(tmp_path):
    root = _one(tmp_path)
    conn = open_manifest(tmp_path / "m.sqlite")
    upsert_map(conn, {"key": "1f00", "bs_automapper": 1})
    index_folder(conn, root, CFG)
    reasons = json.loads(conn.execute("SELECT reject_reasons FROM difficulties").fetchone()[0])
    assert "automapper" in reasons


def test_unreadable_folder_is_counted_and_skipped(tmp_path):
    root = _one(tmp_path)
    bad = root / "2f00 (Bad)"
    bad.mkdir()
    (bad / "Info.dat").write_text("{not json", "utf-8")
    conn = open_manifest(tmp_path / "m.sqlite")
    s = index_folder(conn, root, CFG)
    assert s.n_maps == 2
    assert s.reasons.get("unreadable") == 1
    assert conn.execute("SELECT COUNT(*) FROM maps").fetchone()[0] == 1


def test_build_features_and_decode_failure(tmp_path):
    root = tmp_path / "levels"
    root.mkdir()
    shutil.copytree(FIX / "v2_basic", root / "1f00 (Good)")
    shutil.copytree(FIX / "v3_basic", root / "2f00 (Bad)")
    (root / "2f00 (Bad)" / "song.egg").write_bytes(b"this is not audio")
    conn = open_manifest(tmp_path / "m.sqlite")
    index_folder(conn, root, CFG)
    # force acceptance so the 1-second fixtures reach the feature stage
    conn.execute("UPDATE maps SET audio_ok=1")
    conn.execute("UPDATE difficulties SET accepted=1, reject_reasons='[]'")
    conn.commit()
    feats = tmp_path / "features"
    assert build_features(conn, feats) == (1, 1)
    assert (feats / "1f00.npy").is_file()
    bad = conn.execute("SELECT * FROM maps WHERE key='2f00'").fetchone()
    assert bad["audio_ok"] == 0
    rows = conn.execute(
        "SELECT accepted, reject_reasons FROM difficulties WHERE map_key='2f00'").fetchall()
    assert rows and all(r["accepted"] == 0 and "no_audio" in json.loads(r["reject_reasons"])
                        for r in rows)
    good = conn.execute("SELECT accepted FROM difficulties WHERE map_key='1f00'").fetchall()
    assert good and all(r["accepted"] == 1 for r in good)


def test_one_bad_folder_does_not_stop_the_run(tmp_path, monkeypatch):
    from bsmaker.data import index_local
    root = tmp_path / "levels"
    root.mkdir()
    shutil.copytree(FIX / "v2_basic", root / "1f00 (A)")
    shutil.copytree(FIX / "v2_basic", root / "2f00 (B)")
    conn = open_manifest(tmp_path / "m.sqlite")
    index_folder(conn, root, CFG)
    real = index_local.probe_duration

    def boom(path):
        if path is not None and "1f00" in str(path):
            raise RuntimeError("boom")
        return real(path)
    monkeypatch.setattr(index_local, "probe_duration", boom)
    s = index_folder(conn, root, CFG)
    assert s.reasons.get("unreadable", 0) == 1
    rows = conn.execute("SELECT accepted, reject_reasons FROM difficulties WHERE map_key='1f00'").fetchall()
    assert rows and all(r["accepted"] == 0 and "unreadable" in r["reject_reasons"] for r in rows)
    assert conn.execute("SELECT COUNT(*) FROM difficulties WHERE map_key='2f00'").fetchone()[0] > 0
