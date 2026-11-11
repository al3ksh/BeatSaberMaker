import json
import shutil
import threading
import zipfile
from pathlib import Path

import pytest

from bsmaker.config import Config
from bsmaker.data import scrape as scrape_mod
from bsmaker.data.beatsaver import BeatSaverError, MapInfo
from bsmaker.data.manifest import open_manifest, upsert_map
from bsmaker.data.scrape import MIN_FREE_BYTES, download_one, scrape, select_candidates

FIX = Path(__file__).resolve().parents[1] / "fixtures" / "maps" / "v2_basic"


def info(key, *, score=0.5, upvotes=0, ranked=False, bl=False, curated=False, auto=False):
    return MapInfo(key=key, hash="h" + key, download_url=f"https://r2cdn.beatsaver.com/{key}.zip",
                   title="T", mapper="M", tags=("tech",), score=score, upvotes=upvotes,
                   ranked=ranked, bl_ranked=bl, curated=curated, automapper=auto)


class FakeClient:
    """Serves search pages by filter kind; downloads write a real zip of the v2_basic fixture."""

    def __init__(self, ranked=(), bl=(), curated=(), rated=(), page_size=2, fail=(), bad_zip=()):
        self.cats = {"ranked": list(ranked), "bl": list(bl), "curated": list(curated),
                     "rated": list(rated)}
        self.page_size = page_size
        self.fail, self.bad_zip = set(fail), set(bad_zip)
        self.pages: list[tuple[str, int]] = []
        self.downloads: list[str] = []
        self.lock = threading.Lock()

    @staticmethod
    def _cat(f):
        if f.get("leaderboard") == "Ranked":
            return "ranked"
        if f.get("leaderboard") == "BeatLeader":
            return "bl"
        if f.get("curated"):
            return "curated"
        return "rated"

    def search(self, page, **filters):
        cat = self._cat(filters)
        self.pages.append((cat, page))
        items = self.cats[cat]
        return items[page * self.page_size:(page + 1) * self.page_size]

    def download(self, url, dest, max_bytes=0):
        key = url.rsplit("/", 1)[1][:-4]
        with self.lock:
            self.downloads.append(key)
        if key in self.fail:
            raise BeatSaverError("boom")
        if key in self.bad_zip:
            Path(dest).write_bytes(b"not a zip")
            return
        with zipfile.ZipFile(dest, "w") as z:
            for p in FIX.iterdir():
                z.write(p, p.name)
            z.writestr("cover.jpg", b"jpg")


@pytest.fixture(autouse=True)
def fake_compact(monkeypatch):
    monkeypatch.setattr(scrape_mod, "write_compact", lambda a, b: shutil.copy(a, b))


def _cfg(tmp_path):
    return Config(custom_levels=tmp_path / "cl", data_dir=tmp_path / "data", base_dir=tmp_path)


def _run(tmp_path, client, **kw):
    cfg = _cfg(tmp_path)
    conn = open_manifest(cfg.manifest_path)
    kw.setdefault("free_bytes", lambda p: MIN_FREE_BYTES * 2)
    return cfg, conn, scrape(conn, client, cfg, **kw)


def test_select_candidates_union_and_priority():
    c = FakeClient(ranked=[info("a", ranked=True), info("b", ranked=True)],
                   bl=[info("b", bl=True), info("c", bl=True)],
                   curated=[info("d", curated=True)],
                   rated=[info("e", score=0.9, upvotes=200)])
    assert [m.key for m in select_candidates(c, 100)] == ["a", "b", "c", "d", "e"]


def test_select_skips_automapper():
    c = FakeClient(ranked=[info("a", ranked=True, auto=True), info("b", ranked=True)])
    assert [m.key for m in select_candidates(c, 100)] == ["b"]


def test_select_applies_score_and_upvote_floor():
    c = FakeClient(rated=[info("a0", score=0.86, upvotes=99), info("a1", score=0.85, upvotes=100),
                          info("a2", score=0.84, upvotes=500)])
    assert [m.key for m in select_candidates(c, 100)] == ["a1"]


def test_select_is_lazy():
    c = FakeClient(ranked=[info(f"{i:x}", ranked=True) for i in range(100)], page_size=10)
    got = list(select_candidates(c, 5))
    assert len(got) == 5
    assert c.pages == [("ranked", 0)]


def test_select_dedupes_case_insensitively():
    c = FakeClient(ranked=[info("ab", ranked=True)], curated=[info("AB", curated=True)])
    assert len(list(select_candidates(c, 10))) == 1


def test_scrape_downloads_and_marks_done(tmp_path):
    c = FakeClient(ranked=[info("1a", ranked=True), info("2b", ranked=True)])
    cfg, conn, s = _run(tmp_path, c, max_maps=2)
    assert (s.downloaded, s.skipped_existing, s.failed, s.stopped_low_disk) == (2, 0, 0, False)
    for k in ("1a", "2b"):
        folder = cfg.raw_dir / k
        assert {p.name for p in folder.iterdir()} == {"Info.dat", "ExpertStandard.dat", "audio.ogg"}
        r = conn.execute("SELECT * FROM maps WHERE key = ?", (k,)).fetchone()
        assert r["download_state"] == "done" and r["source"] == "beatsaver"
        assert r["bs_hash"] == "h" + k and r["bs_ranked"] == 1 and json.loads(r["bs_tags"]) == ["tech"]
        assert r["title"] and r["folder"].endswith(k)
        assert conn.execute("SELECT COUNT(*) FROM difficulties WHERE map_key = ?", (k,)).fetchone()[0] > 0
    assert conn.execute("SELECT COUNT(*) FROM maps").fetchone()[0] == 2


def test_scrape_skips_done_and_local(tmp_path):
    cfg = _cfg(tmp_path)
    conn = open_manifest(cfg.manifest_path)
    upsert_map(conn, {"key": "1a", "source": "beatsaver", "download_state": "done"})
    upsert_map(conn, {"key": "2b", "source": "local"})
    c = FakeClient(ranked=[info("1A", ranked=True), info("2B", ranked=True), info("3c", ranked=True)])
    s = scrape(conn, c, cfg, max_maps=10, free_bytes=lambda p: MIN_FREE_BYTES * 2)
    assert c.downloads == ["3c"]
    assert (s.downloaded, s.skipped_existing) == (1, 2)
    assert conn.execute("SELECT COUNT(*) FROM maps").fetchone()[0] == 3


def test_scrape_stops_on_low_disk(tmp_path):
    c = FakeClient(ranked=[info("1a", ranked=True)])
    cfg, conn, s = _run(tmp_path, c, free_bytes=lambda p: MIN_FREE_BYTES - 1)
    assert s.stopped_low_disk is True and s.downloaded == 0 and c.downloads == []


def test_partial_folder_removed_on_start_and_redownloaded(tmp_path):
    cfg = _cfg(tmp_path)
    (cfg.raw_dir / "1a.partial").mkdir(parents=True)
    (cfg.raw_dir / "1a.partial" / "junk.bin").write_bytes(b"x")
    (cfg.raw_dir / "9f.partial").mkdir()
    conn = open_manifest(cfg.manifest_path)
    upsert_map(conn, {"key": "1a", "source": "beatsaver", "download_state": "pending"})
    s = scrape(conn, FakeClient(ranked=[info("1a", ranked=True)]), cfg, max_maps=1,
               free_bytes=lambda p: MIN_FREE_BYTES * 2)
    assert s.downloaded == 1
    assert not list(cfg.raw_dir.glob("*.partial"))
    assert conn.execute("SELECT download_state FROM maps WHERE key='1a'").fetchone()[0] == "done"


def test_failed_download_cleans_up_and_marks_failed(tmp_path):
    c = FakeClient(ranked=[info("1a", ranked=True), info("2b", ranked=True)], fail={"1a"})
    cfg, conn, s = _run(tmp_path, c, max_maps=2)
    assert (s.downloaded, s.failed) == (1, 1)
    assert not list(cfg.raw_dir.glob("*.partial")) and not (cfg.raw_dir / "1a").exists()
    assert conn.execute("SELECT download_state FROM maps WHERE key='1a'").fetchone()[0] == "failed"


def test_unsafe_archive_marks_failed(tmp_path):
    c = FakeClient(ranked=[info("1a", ranked=True)], bad_zip={"1a"})
    cfg, conn, s = _run(tmp_path, c)
    assert s.failed == 1
    assert not list(cfg.raw_dir.glob("*.partial"))
    assert conn.execute("SELECT download_state FROM maps WHERE key='1a'").fetchone()[0] == "failed"


def test_missing_named_audio_fails(tmp_path):
    class NoAudio(FakeClient):
        def download(self, url, dest, max_bytes=0):
            with zipfile.ZipFile(dest, "w") as z:
                for p in FIX.iterdir():
                    if p.name != "song.egg":
                        z.write(p, p.name)
    c = NoAudio(ranked=[info("1a", ranked=True)])
    cfg, conn, s = _run(tmp_path, c)
    assert s.failed == 1 and not (cfg.raw_dir / "1a").exists()


def test_audio_already_named_audio_ogg(tmp_path):
    class OggClient(FakeClient):
        def download(self, url, dest, max_bytes=0):
            with zipfile.ZipFile(dest, "w") as z:
                for p in FIX.iterdir():
                    if p.name == "Info.dat":
                        d = json.loads(p.read_text(encoding="utf-8-sig"))
                        d["_songFilename"] = "AUDIO.ogg"
                        z.writestr("Info.dat", json.dumps(d))
                    elif p.name == "song.egg":
                        z.write(p, "AUDIO.ogg")
                    else:
                        z.write(p, p.name)
    cfg = _cfg(tmp_path)
    out = download_one(OggClient(), info("1a"), cfg.raw_dir)
    assert {p.name for p in out.iterdir()} == {"Info.dat", "ExpertStandard.dat", "audio.ogg"}


def test_workers_two_all_done(tmp_path):
    keys = [f"{i:x}a" for i in range(1, 7)]
    c = FakeClient(ranked=[info(k, ranked=True) for k in keys])
    cfg, conn, s = _run(tmp_path, c, max_maps=6, workers=2)
    assert s.downloaded == 6
    assert conn.execute("SELECT COUNT(*) FROM maps WHERE download_state='done'").fetchone()[0] == 6


def test_select_drops_non_hex_keys():
    c = FakeClient(ranked=[info("zz\x1b[31m", ranked=True), info("../x", ranked=True),
                           info("ab12", ranked=True)])
    assert [m.key for m in select_candidates(c, 10)] == ["ab12"]


def test_select_stops_on_repeating_pages():
    class Loop(FakeClient):
        def search(self, page, **f):
            self.pages.append((self._cat(f), page))
            return [info("ab", ranked=True)]
    c = Loop()
    assert [m.key for m in select_candidates(c, 10)] == ["ab"]
    assert len(c.pages) < 40


def test_oversize_info_refused(tmp_path):
    class BigInfo(FakeClient):
        def download(self, url, dest, max_bytes=0):
            with zipfile.ZipFile(dest, "w") as z:
                z.writestr("Info.dat", b" " * (2 * 2**20))
                z.writestr("song.egg", b"x")
    cfg = _cfg(tmp_path)
    with pytest.raises(BeatSaverError):
        download_one(BigInfo(), info("1a"), cfg.raw_dir)
    assert not list(cfg.raw_dir.glob("*.partial"))


def test_search_error_ends_run_cleanly_and_indexes(tmp_path, capsys):
    class Dies(FakeClient):
        def search(self, page, **f):
            if page >= 1:
                raise BeatSaverError("search down")
            return super().search(page, **f)
    c = Dies(ranked=[info("1a", ranked=True), info("2b", ranked=True)])
    cfg, conn, s = _run(tmp_path, c, max_maps=10)
    assert s.downloaded == 2
    assert conn.execute("SELECT download_state FROM maps WHERE key='1a'").fetchone()[0] == "done"
    assert conn.execute("SELECT COUNT(*) FROM difficulties").fetchone()[0] > 0
    assert "search down" in capsys.readouterr().out


def test_keyboard_interrupt_settles_finished(tmp_path):
    class Intr(FakeClient):
        def search(self, page, **f):
            if page >= 1:
                import time
                time.sleep(0.5)
                raise KeyboardInterrupt
            return super().search(page, **f)
    cfg = _cfg(tmp_path)
    conn = open_manifest(cfg.manifest_path)
    c = Intr(ranked=[info("1a", ranked=True), info("2b", ranked=True)])
    with pytest.raises(KeyboardInterrupt):
        scrape(conn, c, cfg, max_maps=10, free_bytes=lambda p: MIN_FREE_BYTES * 2)
    states = {r[0]: r[1] for r in conn.execute("SELECT key, download_state FROM maps")}
    assert states == {"1a": "done", "2b": "done"}


def test_rated_category_continues_past_pages_already_seen():
    shared = [info(f"{i + 16:x}", ranked=True) for i in range(16)]  # 8 pages of 2
    fresh = info("fff", score=0.9, upvotes=500)
    rated = [info(m.key, score=0.9, upvotes=500) for m in shared] + [fresh]
    c = FakeClient(ranked=shared, rated=rated)
    assert [m.key for m in select_candidates(c, 100)][-1] == "fff"


def test_key_with_trailing_newline_rejected():
    c = FakeClient(ranked=[info("ab\n", ranked=True), info("cd", ranked=True)])
    assert [m.key for m in select_candidates(c, 10)] == ["cd"]
    with pytest.raises(BeatSaverError):
        download_one(FakeClient(), info("ab\n"), Path("unused"))
