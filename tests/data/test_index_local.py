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


def test_sqlite_error_propagates_and_is_not_an_unreadable_verdict(tmp_path, monkeypatch):
    import sqlite3

    import pytest

    from bsmaker.data import index_local
    root = _one(tmp_path)
    conn = open_manifest(tmp_path / "m.sqlite")
    index_folder(conn, root, CFG)

    def locked(*a, **k):
        raise sqlite3.OperationalError("database is locked")
    monkeypatch.setattr(index_local, "upsert_difficulty", locked)
    with pytest.raises(sqlite3.OperationalError):
        index_folder(conn, root, CFG)
    rows = conn.execute("SELECT accepted, reject_reasons FROM difficulties").fetchall()
    assert rows and all("unreadable" not in r["reject_reasons"] for r in rows)


def test_one_folder_is_written_in_one_transaction(tmp_path, monkeypatch):
    from bsmaker.data import index_local
    root = _one(tmp_path)
    conn = open_manifest(tmp_path / "m.sqlite")

    def boom(*a, **k):
        raise RuntimeError("boom")
    monkeypatch.setattr(index_local, "upsert_difficulty", boom)
    s = index_folder(conn, root, CFG)
    assert s.reasons.get("unreadable") == 1
    # the map row written before the failing difficulty row was rolled back with it
    assert conn.execute("SELECT COUNT(*) FROM maps").fetchone()[0] == 0


def test_only_keys_limits_indexing(tmp_path):
    root = _root(tmp_path)
    conn = open_manifest(tmp_path / "m.sqlite")
    s = index_folder(conn, root, CFG, only_keys={"1a0"})
    assert s.n_maps == 1
    assert [r["key"] for r in conn.execute("SELECT key FROM maps")] == ["1a0"]


def test_probe_duration_without_ffprobe_raises(monkeypatch):
    import subprocess

    import pytest

    from bsmaker.audio.decode import AudioDecodeError, FfmpegMissingError
    from bsmaker.data import index_local

    def missing(*a, **k):
        raise FileNotFoundError("ffprobe")
    monkeypatch.setattr(subprocess, "run", missing)
    assert not issubclass(FfmpegMissingError, AudioDecodeError)
    with pytest.raises(FfmpegMissingError):
        index_local.probe_duration(Path("x.ogg"))


def test_index_folder_aborts_when_ffprobe_is_missing(tmp_path, monkeypatch):
    import pytest

    from bsmaker.audio.decode import FfmpegMissingError
    from bsmaker.data import index_local

    def missing(path):
        raise FfmpegMissingError("ffprobe not found")
    monkeypatch.setattr(index_local, "probe_duration", missing)
    conn = open_manifest(tmp_path / "m.sqlite")
    with pytest.raises(FfmpegMissingError):
        index_folder(conn, _one(tmp_path), CFG)
    assert conn.execute("SELECT COUNT(*) FROM maps").fetchone()[0] == 0


def _accepted_one(tmp_path):
    root = _one(tmp_path)
    conn = open_manifest(tmp_path / "m.sqlite")
    index_folder(conn, root, CFG)
    conn.execute("UPDATE maps SET audio_ok=1")
    conn.execute("UPDATE difficulties SET accepted=1, reject_reasons='[]'")
    conn.commit()
    return conn


def test_build_features_environment_errors_propagate_without_verdicts(tmp_path, monkeypatch):
    import pytest

    from bsmaker.audio.decode import FfmpegMissingError
    from bsmaker.data import features
    conn = _accepted_one(tmp_path)
    for exc in (FfmpegMissingError("no ffmpeg"), OSError("disk full")):
        def fail(*a, _e=exc, **k):
            raise _e
        monkeypatch.setattr(features, "cached_log_mel", fail)
        with pytest.raises(type(exc)):
            build_features(conn, tmp_path / "features")
    assert conn.execute("SELECT COUNT(*) FROM difficulties WHERE accepted=0").fetchone()[0] == 0
    assert conn.execute("SELECT audio_ok FROM maps").fetchone()[0] == 1


def test_build_features_stops_when_disk_is_low(tmp_path, capsys):
    conn = _accepted_one(tmp_path)
    feats = tmp_path / "features"
    assert build_features(conn, feats, free_bytes=lambda p: 0) == (0, 0)
    assert "free space" in capsys.readouterr().out
    assert not list(feats.glob("*.npy"))
    assert conn.execute("SELECT COUNT(*) FROM difficulties WHERE accepted=0").fetchone()[0] == 0
