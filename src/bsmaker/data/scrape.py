"""Curated BeatSaver scraper: select, download, extract safely, convert audio, index."""
from __future__ import annotations

import json
import re
import shutil
from collections.abc import Callable, Iterator
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path

from bsmaker.audio.decode import write_compact
from bsmaker.config import Config
from bsmaker.data.beatsaver import BeatSaverClient, BeatSaverError, MapInfo
from bsmaker.data.extract import safe_extract
from bsmaker.data.index_local import index_folder
from bsmaker.data.manifest import upsert_map
from bsmaker.formats.reader import _find_ci, _is_plain_name, _load
from bsmaker.playability.config import load_playability_config

MIN_FREE_BYTES = 20 * 2**30
MIN_SCORE = 0.85
MIN_UPVOTES = 100
_KEY_RE = re.compile(r"^[0-9a-f]{1,16}$")

# (name, search filters, client-side predicate) in priority order
_CATEGORIES: tuple[tuple[str, dict, Callable[[MapInfo], bool]], ...] = (
    ("ranked", {"leaderboard": "Ranked", "order": "Rating"}, lambda m: True),
    ("beatleader", {"leaderboard": "BeatLeader", "order": "Rating"}, lambda m: True),
    ("curated", {"curated": "true", "order": "Curated"}, lambda m: True),
    ("rated", {"minRating": MIN_SCORE, "minUpVotes": MIN_UPVOTES, "order": "Rating"},
     lambda m: m.score >= MIN_SCORE and m.upvotes >= MIN_UPVOTES),
)


@dataclass
class ScrapeSummary:
    downloaded: int = 0
    skipped_existing: int = 0
    failed: int = 0
    stopped_low_disk: bool = False


def select_candidates(client: BeatSaverClient, max_maps: int) -> Iterator[MapInfo]:
    """Lazily yield up to `max_maps` distinct, non-automapper maps in category priority order."""
    seen: set[str] = set()
    yielded = 0
    if max_maps <= 0:
        return
    for _name, filters, keep in _CATEGORIES:
        page = 0
        while True:
            docs = client.search(page, **filters)
            if not docs:
                break
            for m in docs:
                k = m.key.lower()
                if k in seen or m.automapper or not keep(m):
                    continue
                seen.add(k)
                yield m
                yielded += 1
                if yielded >= max_maps:
                    return
            page += 1


def _audio_name(folder: Path) -> str | None:
    info_path = _find_ci(folder, "Info.dat")
    if info_path is None:
        return None
    info = _load(info_path)
    if not isinstance(info, dict):
        return None
    name = info.get("_songFilename")
    if name is None and isinstance(info.get("audio"), dict):
        name = info["audio"].get("songFilename")
    return name if _is_plain_name(name) else None


def download_one(client: BeatSaverClient, info: MapInfo, raw_dir: Path) -> Path:
    """Download, extract and convert one map into raw_dir/<key>; returns the final folder."""
    key = info.key.lower()
    if not _KEY_RE.match(key):
        raise BeatSaverError(f"refusing unsafe map key {info.key!r}")
    raw_dir = Path(raw_dir)
    partial = raw_dir / f"{key}.partial"
    final = raw_dir / key
    try:
        shutil.rmtree(partial, ignore_errors=True)
        partial.mkdir(parents=True)
        zip_path = partial / "map.zip"
        client.download(info.download_url, zip_path)
        content = partial / "content"
        safe_extract(zip_path, content)
        zip_path.unlink()
        name = _audio_name(content)
        if name is None:
            raise BeatSaverError(f"map {key}: Info names no usable audio file")
        audio = next((p for p in content.iterdir() if p.name.lower() == name.lower()
                      and p.suffix.lower() in (".egg", ".ogg", ".wav")), None)
        if audio is None:
            raise BeatSaverError(f"map {key}: audio file {name!r} not in archive")
        tmp = partial / "audio.tmp"
        write_compact(audio, tmp)
        for p in content.iterdir():
            if p.suffix.lower() != ".dat":
                p.unlink()
        tmp.replace(content / "audio.ogg")
        if final.exists():
            shutil.rmtree(final)
        content.replace(final)
        shutil.rmtree(partial, ignore_errors=True)
        return final
    except BaseException:
        shutil.rmtree(partial, ignore_errors=True)
        raise


def _default_free(path: Path) -> int:
    return shutil.disk_usage(path).free


def scrape(conn, client: BeatSaverClient, cfg: Config, *, max_maps: int = 20000, workers: int = 2,
           free_bytes: Callable[[Path], int] = _default_free) -> ScrapeSummary:
    """Download curated maps and index them. All SQLite writes happen on the calling thread."""
    raw_dir = cfg.raw_dir
    raw_dir.mkdir(parents=True, exist_ok=True)
    for p in raw_dir.glob("*.partial"):
        shutil.rmtree(p, ignore_errors=True)
    summary = ScrapeSummary()
    running: dict[Future, str] = {}

    def settle(done: set[Future]) -> None:
        for fut in done:
            key = running.pop(fut)
            try:
                fut.result()
            except Exception as exc:  # noqa: BLE001 - any failure of one map is just that map failing
                print(f"  failed {key}: {type(exc).__name__}: {str(exc)[:120]}", flush=True)
                summary.failed += 1
                upsert_map(conn, {"key": key, "download_state": "failed"})
            else:
                summary.downloaded += 1
                upsert_map(conn, {"key": key, "download_state": "done"})
            n = summary.downloaded + summary.failed
            if n % 10 == 0:
                print(f"  {n} processed ({summary.downloaded} ok, {summary.failed} failed)", flush=True)

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        for info in select_candidates(client, max_maps):
            key = info.key.lower()
            row = conn.execute("SELECT source, download_state FROM maps WHERE lower(key) = ?",
                               (key,)).fetchone()
            if row is not None and (row["download_state"] == "done" or row["source"] == "local"):
                summary.skipped_existing += 1
                continue
            while len(running) >= max(1, workers):
                settle(wait(running, return_when=FIRST_COMPLETED).done)
            if free_bytes(raw_dir) < MIN_FREE_BYTES:
                summary.stopped_low_disk = True
                break
            upsert_map(conn, {
                "key": key, "source": "beatsaver", "title": info.title, "mapper": info.mapper,
                "bs_hash": info.hash, "bs_tags": json.dumps(list(info.tags)),
                "bs_score": info.score, "bs_upvotes": info.upvotes, "bs_ranked": int(info.ranked),
                "bs_bl_ranked": int(info.bl_ranked), "bs_curated": int(info.curated),
                "bs_automapper": int(info.automapper), "bs_fetched": 1,
                "download_state": "pending"})
            running[pool.submit(download_one, client, info, raw_dir)] = key
        while running:
            settle(wait(running, return_when=FIRST_COMPLETED).done)

    index_folder(conn, raw_dir, load_playability_config(cfg.playability_path), source="beatsaver")
    return summary

