"""Packed layer cache (lights, walls, bombs, note ornaments), aligned with the event cache."""
from __future__ import annotations

import json
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from bsmaker.data.events_cache import EventCache, _atomic_write
from bsmaker.data.layer_classes import wall_class
from bsmaker.formats.events import FRAMES_PER_SECOND
from bsmaker.formats.layer_tokens import (
    SUPPORTED_ENVIRONMENTS, arc_length_bucket, cells_to_mask, light_event_to_token, wall_to_token)
from bsmaker.formats.model import Difficulty
from bsmaker.formats.reader import read_map
from bsmaker.formats.tokens import encode_token

MATCH_S = 0.001
MIN_LIGHTS_PER_MIN = 100
_ARRAYS = {
    "light_t": np.float32, "light_ch": np.uint8, "light_tok": np.uint8,
    "wall_t": np.float32, "wall_side": np.uint8, "wall_tok": np.uint8,
    "bomb_t": np.float32, "bomb_mask": np.uint16,
    "orn_t": np.float32, "orn_hand": np.uint8, "orn_token": np.uint8, "orn_chain": np.uint8,
    "orn_arc_len": np.uint8, "orn_arc_cell": np.uint8, "orn_arc_dir": np.uint8}
_GROUPS = {"light": ("light_t", "light_ch", "light_tok"), "wall": ("wall_t", "wall_side", "wall_tok"),
           "bomb": ("bomb_t", "bomb_mask"),
           "orn": ("orn_t", "orn_hand", "orn_token", "orn_chain", "orn_arc_len", "orn_arc_cell",
                   "orn_arc_dir")}
_SUMMED = ("n_lights", "n_walls", "n_bomb_groups", "n_bombs", "n_arcs", "n_chains",
           "walls_unsupported", "lights_skipped", "lights_duplicate", "arcs_unmatched",
           "chains_unmatched", "dropped")


def _empty_record(entry_idx: int) -> dict:
    return {"i": entry_idx, "schema": "", "environment": "", "lights_ok": False, "has_v3": False,
            "duration_s": 0.0, "bpm": 0.0, "n_lights": 0, "n_walls": 0, "n_bomb_groups": 0,
            "n_bombs": 0, "n_arcs": 0, "n_chains": 0, "n_notes": 0,
            "wall_classes": {"flash": 0, "side": 0, "centre": 0}, "walls_unsupported": 0,
            "lights_skipped": 0, "lights_duplicate": 0, "arcs_unmatched": 0,
            "chains_unmatched": 0, "dropped": 0, "readable": False}


def _empty_arrays() -> dict[str, np.ndarray]:
    return {k: np.zeros(0, dt) for k, dt in _ARRAYS.items()}


def _pack(rows: dict[str, list], keys: tuple[str, ...]) -> dict[str, np.ndarray]:
    return {k: np.asarray(rows[k], dtype=_ARRAYS[k]) for k in keys}


def _lights(d: Difficulty, rec: dict) -> dict[str, list]:
    last: dict[tuple[int, int], tuple[float, int, int]] = {}
    for ev in d.layers.lights:
        mapped = light_event_to_token(ev.type, ev.value)
        if mapped is None:
            rec["lights_skipped"] += 1
            continue
        ch, tok = mapped
        key = (round(ev.time_s * 1000), ch)
        if key in last:
            rec["lights_duplicate"] += 1
        last[key] = (ev.time_s, ch, tok)
    items = sorted(last.values(), key=lambda r: (r[0], r[1]))
    return {"light_t": [r[0] for r in items], "light_ch": [r[1] for r in items],
            "light_tok": [r[2] for r in items]}


def _walls(d: Difficulty, rec: dict, bpm: float) -> dict[str, list]:
    items = []
    for ob in d.layers.obstacles:
        cls = wall_class(ob)
        rec["wall_classes"][cls] += 1
        mapped = wall_to_token(ob, bpm)
        if mapped is None:
            if cls != "centre":
                rec["walls_unsupported"] += 1
            continue
        items.append((ob.time_s, mapped[0], mapped[1]))
    items.sort()
    return {"wall_t": [r[0] for r in items], "wall_side": [r[1] for r in items],
            "wall_tok": [r[2] for r in items]}


def _bombs(d: Difficulty, rec: dict) -> dict[str, list]:
    groups: list[list] = []  # [time, set of cells]
    for b in sorted(d.layers.bombs, key=lambda b: b.time_s):
        if not (0 <= b.x <= 3 and 0 <= b.y <= 2):
            continue
        rec["n_bombs"] += 1
        if groups and abs(b.time_s - groups[-1][0]) <= MATCH_S:
            groups[-1][1].add((b.x, b.y))
        else:
            groups.append([b.time_s, {(b.x, b.y)}])
    return {"bomb_t": [g[0] for g in groups], "bomb_mask": [cells_to_mask(g[1]) for g in groups]}


def _ornaments(d: Difficulty, rec: dict) -> dict[str, list]:
    notes = sorted(d.notes, key=lambda n: (n.time_s, n.color))
    n = len(notes)
    orn: dict[str, list] = {"orn_t": [nt.time_s for nt in notes], "orn_hand": [nt.color for nt in notes],
                            "orn_token": [], "orn_chain": [0] * n, "orn_arc_len": [0] * n,
                            "orn_arc_cell": [0] * n, "orn_arc_dir": [0] * n}
    for nt in notes:
        try:
            orn["orn_token"].append(encode_token(nt.x, nt.y, nt.direction))
        except ValueError:
            orn["orn_token"].append(0)
    by_cell: dict[tuple[int, int, int], list[int]] = defaultdict(list)
    for k, nt in enumerate(notes):
        by_cell[(nt.color, nt.x, nt.y)].append(k)

    def find(color: int, x: int, y: int, t: float) -> int | None:
        for k in by_cell.get((color, x, y), ()):
            if abs(notes[k].time_s - t) <= MATCH_S:
                return k
        return None

    for a in d.layers.arcs:
        k = find(a.color, a.x, a.y, a.time_s)
        if k is None or not (0 <= a.tx <= 3 and 0 <= a.ty <= 2) or not 0 <= a.td <= 255:
            rec["arcs_unmatched"] += 1
            continue
        orn["orn_arc_len"][k] = arc_length_bucket(a.tail_beat - a.beat)
        orn["orn_arc_cell"][k] = a.ty * 4 + a.tx
        orn["orn_arc_dir"][k] = a.td
        rec["n_arcs"] += 1
    for c in d.layers.chains:
        k = find(c.color, c.x, c.y, c.time_s)
        if k is None:
            rec["chains_unmatched"] += 1
            continue
        orn["orn_chain"][k] = min(max(c.segments, 3), 7)
        rec["n_chains"] += 1
    return orn


def _process(d: Difficulty | None, env: str, bpm: float, duration_s: float,
             entry_idx: int) -> tuple[dict, dict[str, np.ndarray]]:
    rec = _empty_record(entry_idx)
    if d is None:
        return rec, _empty_arrays()
    rec.update(schema=d.schema, environment=env, bpm=bpm, duration_s=duration_s, readable=True)
    if d.layers is None:  # v4: skipped
        return rec, _empty_arrays()
    rec.update(has_v3=d.schema == "v3", dropped=d.layers.dropped, n_notes=len(d.notes))
    rows: dict[str, list] = {}
    rows.update(_lights(d, rec))
    rows.update(_walls(d, rec, bpm))
    rows.update(_bombs(d, rec))
    rows.update(_ornaments(d, rec))
    rec["n_lights"] = len(rows["light_t"])
    rec["n_walls"] = len(rows["wall_t"])
    rec["n_bomb_groups"] = len(rows["bomb_t"])
    per_min = rec["n_lights"] / (duration_s / 60.0) if duration_s > 0 else 0.0
    rec["lights_ok"] = (env in SUPPORTED_ENVIRONMENTS and not d.layers.group_lighting
                        and per_min >= MIN_LIGHTS_PER_MIN)
    arrays: dict[str, np.ndarray] = {}
    for keys in _GROUPS.values():
        arrays.update(_pack(rows, keys))
    return rec, arrays


def build_layer_cache(conn: sqlite3.Connection, layers_dir: Path, events_dir: Path,
                      progress_every: int = 0) -> dict:
    """Write layers.npz and index.json, one record per event-cache entry (same order).

    Returns totals over all entries (counts per element, wall classes, skipped objects, ...).
    A map that cannot be read, a missing difficulty or a v4 difficulty gets an empty record.
    """
    layers_dir = Path(layers_dir)
    layers_dir.mkdir(parents=True, exist_ok=True)
    entries = EventCache.load(events_dir).entries
    by_map: dict[str, list] = defaultdict(list)
    for e in entries:
        by_map[e.map_key].append(e)

    records: list[dict] = [{}] * len(entries)
    arrays: list[dict[str, np.ndarray]] = [{}] * len(entries)
    unreadable = 0
    for n, (key, group) in enumerate(by_map.items(), start=1):
        if progress_every and n % progress_every == 0:
            print(f"layers: {n}/{len(by_map)} maps", flush=True)
        bm = None
        row = conn.execute("SELECT folder FROM maps WHERE key = ?", (key,)).fetchone()
        if row is not None:
            try:
                bm = read_map(Path(row["folder"]), layers=True)
            except Exception as exc:  # noqa: BLE001 - untrusted map data: skip and count
                unreadable += 1
                if unreadable <= 5:
                    print(f"layers: skipping unreadable map {key}: {type(exc).__name__}", flush=True)
        diffs = {(d.characteristic, d.rank): d for d in bm.difficulties} if bm else {}
        for e in group:
            d = diffs.get((e.characteristic, e.rank))
            rec, arr = _process(d, bm.environment if bm else "", bm.bpm if bm else 0.0,
                                e.n_frames / FRAMES_PER_SECOND, e.i)
            rec.update(map_key=e.map_key, characteristic=e.characteristic, rank=e.rank)
            records[e.i], arrays[e.i] = rec, arr
    if unreadable:
        print(f"layers: {unreadable} maps unreadable, skipped", flush=True)

    out: dict[str, np.ndarray] = {}
    for group, keys in _GROUPS.items():
        lengths = [len(a[keys[0]]) for a in arrays]
        off = np.zeros(len(lengths) + 1, dtype=np.int64)
        np.cumsum(lengths, out=off[1:])
        out[f"{group}_offsets"] = off
        for k in keys:
            out[k] = np.concatenate([a[k] for a in arrays]) if arrays else np.zeros(0, _ARRAYS[k])

    def write_npz(tmp: Path) -> None:
        with tmp.open("wb") as fh:
            np.savez(fh, **out)

    def write_index(tmp: Path) -> None:
        tmp.write_text(json.dumps(records), encoding="utf-8")

    _atomic_write(layers_dir / "layers.npz", write_npz)
    _atomic_write(layers_dir / "index.json", write_index)

    totals: Counter = Counter()
    for r in records:
        totals["entries"] += 1
        totals["v4"] += r["schema"] == "v4"
        totals["empty"] += not r["readable"]
        totals["lights_ok"] += r["lights_ok"]
        for k in _SUMMED:
            totals[k] += r[k]
        for c, v in r["wall_classes"].items():
            totals[f"wall_{c}"] += v
    totals["unreadable_maps"] = unreadable
    return dict(totals)


class LayerCache:
    def __init__(self, records: list[dict], data: dict[str, np.ndarray]) -> None:
        self.records = records
        self._d = data

    @classmethod
    def load(cls, layers_dir: Path) -> LayerCache:
        layers_dir = Path(layers_dir)
        records = json.loads((layers_dir / "index.json").read_text(encoding="utf-8"))
        with np.load(layers_dir / "layers.npz", allow_pickle=False) as z:
            data = {k: z[k] for k in z.files}
        for group, keys in _GROUPS.items():
            off = data[f"{group}_offsets"]
            if len(off) != len(records) + 1 or any(len(data[k]) != int(off[-1]) for k in keys):
                raise ValueError(
                    f"layer cache in {layers_dir} is inconsistent ({group}); "
                    "rebuild with `bsmaker data build-layers`")
        return cls(records, data)

    def _slice(self, group: str, i: int) -> tuple[np.ndarray, ...]:
        off = self._d[f"{group}_offsets"]
        a, b = int(off[i]), int(off[i + 1])
        return tuple(self._d[k][a:b] for k in _GROUPS[group])

    def lights(self, i: int) -> tuple[np.ndarray, ...]:
        return self._slice("light", i)

    def walls(self, i: int) -> tuple[np.ndarray, ...]:
        return self._slice("wall", i)

    def bombs(self, i: int) -> tuple[np.ndarray, ...]:
        return self._slice("bomb", i)

    def ornaments(self, i: int) -> tuple[np.ndarray, ...]:
        return self._slice("orn", i)

    def meta(self, i: int) -> dict:
        return self.records[i]
