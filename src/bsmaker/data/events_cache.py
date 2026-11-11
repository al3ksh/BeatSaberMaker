"""Packed training cache of token events for every accepted human difficulty."""
from __future__ import annotations

import json
import os
import sqlite3
from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from bsmaker.data.manifest import iter_accepted
from bsmaker.data.styles import styles_from_tags
from bsmaker.formats.events import Event, to_events
from bsmaker.formats.model import Note
from bsmaker.formats.reader import read_map
from bsmaker.formats.tokens import NONE_TOKEN
from bsmaker.playability.checker import drop_slider_tails

SLIDER_GAP_S = 0.12
MIN_EVENTS = 20


def training_events(notes: Sequence[Note]) -> list[Event]:
    """Events as the model sees them: grouped notes with slider tails dropped."""
    events, _ = to_events(notes)
    return drop_slider_tails(events, SLIDER_GAP_S)


@dataclass(frozen=True)
class Entry:
    i: int
    map_key: str
    characteristic: str
    rank: str
    split: str | None
    nps: float
    styles: tuple[str, ...]
    njs: float
    njs_offset: float
    n_frames: int


def _atomic_write(path: Path, writer: Callable[[Path], None]) -> None:
    tmp = path.with_name(path.name + ".tmp")
    writer(tmp)
    os.replace(tmp, path)


def build_event_cache(conn: sqlite3.Connection, events_dir: Path, features_dir: Path,
                      progress_every: int = 0) -> int:
    """Write events.npz and index.json; return the number of difficulties left out."""
    events_dir = Path(events_dir)
    features_dir = Path(features_dir)
    events_dir.mkdir(parents=True, exist_ok=True)
    by_map: dict[str, list] = defaultdict(list)
    for row in iter_accepted(conn):
        by_map[row["map_key"]].append(row)

    times: list[np.ndarray] = []
    lefts: list[np.ndarray] = []
    rights: list[np.ndarray] = []
    lengths: list[int] = []
    index: list[dict] = []
    left_out = 0
    unreadable = 0
    for n, (key, rows) in enumerate(by_map.items(), start=1):
        if progress_every and n % progress_every == 0:
            print(f"events: {n}/{len(by_map)} maps, {len(index)} difficulties", flush=True)
        feat = features_dir / f"{key}.npy"
        if not feat.is_file():
            left_out += len(rows)
            continue
        m = conn.execute("SELECT folder, bs_tags FROM maps WHERE key = ?", (key,)).fetchone()
        if m is None:
            left_out += len(rows)
            continue
        try:
            bm = read_map(Path(m["folder"]))
            n_frames = int(np.load(feat, mmap_mode="r").shape[0])
        except Exception:  # noqa: BLE001 - untrusted map data: skip and count
            unreadable += 1
            left_out += len(rows)
            continue
        styles = list(styles_from_tags(json.loads(m["bs_tags"] or "[]")))
        diffs = {(d.characteristic, d.rank): d for d in bm.difficulties}
        for row in rows:
            d = diffs.get((row["characteristic"], row["rank"]))
            if d is None:
                left_out += 1
                continue
            ev = training_events(d.notes)
            if len(ev) < MIN_EVENTS:
                left_out += 1
                continue
            t = np.fromiter((e.time_s for e in ev), dtype=np.float32, count=len(ev))
            lf = np.fromiter((e.left for e in ev), dtype=np.uint8, count=len(ev))
            rt = np.fromiter((e.right for e in ev), dtype=np.uint8, count=len(ev))
            span = ev[-1].time_s - ev[0].time_s
            n_notes = int((lf != NONE_TOKEN).sum() + (rt != NONE_TOKEN).sum())
            times.append(t)
            lefts.append(lf)
            rights.append(rt)
            lengths.append(len(ev))
            index.append({
                "map_key": key, "characteristic": row["characteristic"], "rank": row["rank"],
                "split": row["split"], "nps": n_notes / span if span > 0 else 0.0,
                "styles": styles, "njs": row["njs"], "njs_offset": row["njs_offset"],
                "n_frames": n_frames})
    if unreadable:
        print(f"events: {unreadable} maps unreadable, skipped", flush=True)

    offsets = np.zeros(len(lengths) + 1, dtype=np.int64)
    np.cumsum(lengths, out=offsets[1:])
    time_all = np.concatenate(times) if times else np.zeros(0, np.float32)
    left_all = np.concatenate(lefts) if lefts else np.zeros(0, np.uint8)
    right_all = np.concatenate(rights) if rights else np.zeros(0, np.uint8)

    def write_npz(tmp: Path) -> None:
        with tmp.open("wb") as fh:
            np.savez(fh, time_s=time_all, left=left_all, right=right_all, offsets=offsets)

    def write_index(tmp: Path) -> None:
        tmp.write_text(json.dumps(index), encoding="utf-8")

    _atomic_write(events_dir / "events.npz", write_npz)
    _atomic_write(events_dir / "index.json", write_index)
    return left_out


class EventCache:
    def __init__(self, entries: list[Entry], time_s: np.ndarray, left: np.ndarray,
                 right: np.ndarray, offsets: np.ndarray) -> None:
        self.entries = entries
        self._t, self._l, self._r, self._o = time_s, left, right, offsets

    @classmethod
    def load(cls, events_dir: Path) -> EventCache:
        events_dir = Path(events_dir)
        raw = json.loads((events_dir / "index.json").read_text(encoding="utf-8"))
        entries = [Entry(i=i, map_key=r["map_key"], characteristic=r["characteristic"],
                         rank=r["rank"], split=r["split"], nps=r["nps"],
                         styles=tuple(r["styles"]), njs=r["njs"], njs_offset=r["njs_offset"],
                         n_frames=r["n_frames"]) for i, r in enumerate(raw)]
        with np.load(events_dir / "events.npz") as z:
            return cls(entries, z["time_s"], z["left"], z["right"], z["offsets"])

    def events(self, i: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        a, b = int(self._o[i]), int(self._o[i + 1])
        return self._t[a:b], self._l[a:b], self._r[a:b]

    def select(self, split: str) -> list[Entry]:
        return [e for e in self.entries if e.split == split]
