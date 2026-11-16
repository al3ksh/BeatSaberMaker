"""Read Beat Saber map folders (Info + difficulty files, v2/v3/v4 schemas)."""
from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

from typing import Callable

from bsmaker.formats.model import (Arc, BeatMap, Bomb, Chain, Difficulty, DifficultyFlags, Layers,
                                   LightEvent, Note, Obstacle)
from bsmaker.formats.tempo import TempoMap, read_tempo

_RawNote = tuple[float, int, int, int, int, bool]  # beat, x, y, color, direction, has_custom


def map_key(folder_name: str) -> str:
    m = re.match(r"^([0-9a-fA-F]+)(?: |$)", folder_name)
    if m:
        return m.group(1)
    return "local-" + hashlib.sha1(folder_name.encode("utf-8")).hexdigest()[:10]


def _load(path: Path) -> Any:
    with path.open(encoding="utf-8-sig") as fh:
        return json.load(fh)


def _find_ci(folder: Path, name: str) -> Path | None:
    exact = folder / name
    if exact.is_file():
        return exact
    low = name.lower()
    for p in folder.iterdir():
        if p.is_file() and p.name.lower() == low:
            return p
    return None


def _is_plain_name(name: object) -> bool:
    """True for a bare file name: no separators, drive or colon, not '.'/'..'."""
    return (isinstance(name, str) and bool(name) and name not in (".", "..")
            and not any(c in name for c in "/\\:\0"))


def _find_safe(folder: Path, name: object) -> Path | None:
    """_find_ci for a name taken from untrusted Info data; unsafe names count as missing."""
    return _find_ci(folder, name) if _is_plain_name(name) else None


def _detect_schema(data: dict) -> str:
    if "_notes" in data:
        return "v2"
    if "colorNotesData" in data:
        return "v4"
    if "colorNotes" in data:
        return "v3"
    return "v2"  # no notes key at all: treat as an empty v2 map


def _raw_notes(data: dict, schema: str) -> tuple[list[_RawNote], int]:
    out: list[_RawNote] = []
    bombs = 0
    if schema == "v2":
        for n in data.get("_notes") or []:
            t = n.get("_type", 0)
            if t == 3:
                bombs += 1
            elif t in (0, 1):
                out.append((float(n.get("_time", 0)), int(n.get("_lineIndex", 0)),
                            int(n.get("_lineLayer", 0)), t, int(n.get("_cutDirection", 0)), bool(n.get("_customData"))))
    elif schema == "v3":
        for n in data.get("colorNotes") or []:
            out.append((float(n.get("b", 0)), int(n.get("x", 0)), int(n.get("y", 0)),
                        int(n.get("c", 0)), int(n.get("d", 0)), bool(n.get("customData"))))
        bombs = len(data.get("bombNotes") or [])
    else:
        pool = data.get("colorNotesData") or []
        for n in data.get("colorNotes") or []:
            d = pool[n.get("i", 0)]
            out.append((float(n.get("b", 0)), int(d.get("x", 0)), int(d.get("y", 0)),
                        int(d.get("c", 0)), int(d.get("d", 0)),
                        bool(n.get("customData")) or bool(d.get("customData"))))
        bombs = len(data.get("bombNotes") or [])
    return out, bombs


def _counts(data: dict, schema: str) -> tuple[int, int, int]:
    """(obstacles, arcs, chains)."""
    if schema == "v2":
        return len(data.get("_obstacles") or []), 0, 0
    if schema == "v3":
        return (len(data.get("obstacles") or []), len(data.get("sliders") or []),
                len(data.get("burstSliders") or []))
    return (len(data.get("obstacles") or []), len(data.get("arcs") or []),
            len(data.get("chains") or []))


class _Bad(Exception):
    pass


def _f(v: object) -> float:
    """A finite, sane number or _Bad."""
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise _Bad
    f = float(v)
    if not math.isfinite(f) or abs(f) > 1e6:
        raise _Bad
    return f


def _i(v: object) -> int:
    return int(round(_f(v)))


def _entries(data: dict, key: str) -> list:
    v = data.get(key)
    return v if isinstance(v, list) else []


def _parse_layers(data: dict, schema: str, to_s: Callable[[float], float]) -> Layers | None:
    if schema not in ("v2", "v3"):
        return None
    bombs: list[Bomb] = []
    obst: list[Obstacle] = []
    arcs: list[Arc] = []
    chains: list[Chain] = []
    lights: list[LightEvent] = []
    dropped = 0

    def run(items: list, fn: Callable[[dict], None]) -> None:
        nonlocal dropped
        for e in items:
            try:
                if not isinstance(e, dict):
                    raise _Bad
                fn(e)
            except (_Bad, KeyError, OverflowError):
                dropped += 1

    def v2_wall(e: dict) -> None:
        t = _f(e["_time"])
        typ = e["_type"]
        if typ not in (0, 1) or isinstance(typ, bool):
            raise _Bad
        dur = _f(e["_duration"])
        x = _i(e["_lineIndex"])
        w = _i(e["_width"])
        obst.append(Obstacle(to_s(t), t, to_s(t + dur) - to_s(t), dur, x,
                             0 if typ == 0 else 2, w, 5 if typ == 0 else 3))

    def v2_bomb(e: dict) -> None:
        t = _f(e["_time"])
        bombs.append(Bomb(to_s(t), t, _i(e["_lineIndex"]), _i(e["_lineLayer"])))

    def v2_event(e: dict) -> None:
        t = _f(e["_time"])
        typ = _i(e["_type"])
        fv = _f(e["_floatValue"]) if "_floatValue" in e else 1.0
        lights.append(LightEvent(to_s(t), t, typ, _i(e["_value"]), fv))

    def v3_wall(e: dict) -> None:
        t = _f(e["b"])
        dur = _f(e["d"])
        obst.append(Obstacle(to_s(t), t, to_s(t + dur) - to_s(t), dur, _i(e["x"]), _i(e["y"]),
                             _i(e["w"]), _i(e["h"])))

    def v3_bomb(e: dict) -> None:
        t = _f(e["b"])
        bombs.append(Bomb(to_s(t), t, _i(e["x"]), _i(e["y"])))

    def v3_arc(e: dict) -> None:
        t = _f(e["b"])
        tb = _f(e["tb"])
        arcs.append(Arc(to_s(t), t, _i(e["c"]), _i(e["x"]), _i(e["y"]), _i(e["d"]), to_s(tb), tb,
                        _i(e["tx"]), _i(e["ty"]), _i(e["tc"])))

    def v3_chain(e: dict) -> None:
        t = _f(e["b"])
        tb = _f(e["tb"])
        chains.append(Chain(to_s(t), t, _i(e["c"]), _i(e["x"]), _i(e["y"]), _i(e["d"]), to_s(tb),
                            tb, _i(e["tx"]), _i(e["ty"]), _i(e["sc"]), _f(e["s"])))

    def v3_event(e: dict) -> None:
        t = _f(e["b"])
        fv = _f(e["f"]) if "f" in e else 1.0
        lights.append(LightEvent(to_s(t), t, _i(e["et"]), _i(e["i"]), fv))

    group = False
    if schema == "v2":
        run([n for n in _entries(data, "_notes") if isinstance(n, dict) and n.get("_type") == 3],
            v2_bomb)
        run(_entries(data, "_obstacles"), v2_wall)
        run([e for e in _entries(data, "_events")
             if not (isinstance(e, dict) and e.get("_type") == 100)], v2_event)
    else:
        run(_entries(data, "bombNotes"), v3_bomb)
        run(_entries(data, "obstacles"), v3_wall)
        run(_entries(data, "sliders"), v3_arc)
        run(_entries(data, "burstSliders"), v3_chain)
        run(_entries(data, "basicBeatmapEvents"), v3_event)
        group = any(_entries(data, k) for k in ("lightColorEventBoxGroups",
                    "lightRotationEventBoxGroups", "lightTranslationEventBoxGroups"))
    return Layers(tuple(sorted(bombs, key=lambda o: (o.time_s, o.x, o.y))),
                  tuple(sorted(obst, key=lambda o: (o.time_s, o.x, o.y))),
                  tuple(sorted(arcs, key=lambda o: (o.time_s, o.color, o.x, o.y))),
                  tuple(sorted(chains, key=lambda o: (o.time_s, o.color, o.x, o.y))),
                  tuple(sorted(lights, key=lambda o: (o.time_s, o.type, o.value))),
                  group, dropped)


def _read_difficulty(path: Path, folder: Path, characteristic: str, rank: str, njs: float,
                     njs_offset: float, bpm: float, time_offset: float,
                     tempos: dict | None = None, layers: bool = False) -> Difficulty:
    data = _load(path)
    if not isinstance(data, dict):
        raise ValueError("difficulty file is not a JSON object")
    schema = _detect_schema(data)
    raw, bombs = _raw_notes(data, schema)
    kind, tempo = read_tempo(data, schema, bpm, folder)
    if tempos is not None:
        tempos[(characteristic, rank)] = tempo
    use_map = kind == "events" and all(math.isfinite(b) and b > 0 for _, b in tempo.points)
    bpm_min, bpm_max = tempo.bpm_range()
    offgrid = False
    custom = False
    notes: list[Note] = []
    for beat, x, y, c, d, has_custom in raw:
        custom = custom or has_custom
        if not (0 <= x <= 3 and 0 <= y <= 2 and 0 <= d <= 8):
            offgrid = True
            continue
        notes.append(Note(tempo.to_seconds(beat) if use_map else beat * 60.0 / bpm, beat, x, y, c, d))
    notes.sort(key=lambda n: (n.time_s, n.color, n.x, n.y))
    n_obst, n_arcs, n_chains = _counts(data, schema)
    flags = DifficultyFlags(kind != "none", offgrid, custom, time_offset != 0,
                            kind, bpm_min, bpm_max)
    lay = None
    if layers:
        lay = _parse_layers(data, schema,
                            (lambda b: tempo.to_seconds(b)) if use_map else (lambda b: b * 60.0 / bpm))
    return Difficulty(characteristic, rank, schema, njs, njs_offset, tuple(notes), bombs,
                      n_obst, n_arcs, n_chains, flags, lay)


def read_map(folder: Path, *, layers: bool = False) -> BeatMap:
    return read_map_with_tempo(folder, layers=layers)[0]


def read_map_with_tempo(folder: Path, *, layers: bool = False) -> tuple[BeatMap, dict[tuple[str, str], TempoMap]]:
    """read_map plus the TempoMap recorded in each difficulty file, keyed (characteristic, rank)."""
    folder = Path(folder)
    tempos: dict[tuple[str, str], TempoMap] = {}
    problems: list[str] = []
    info_path = _find_ci(folder, "Info.dat")
    if info_path is None:
        raise FileNotFoundError(f"no Info.dat in {folder}")
    info = _load(info_path)

    reqs: list[str] = []
    sugs: list[str] = []
    specs: list[tuple[str, str, str, float, float]] = []  # characteristic, rank, file, njs, offset
    time_offset = 0.0
    environment = ""
    if "_difficultyBeatmapSets" in info or "_songName" in info:
        title = info.get("_songName", "")
        artist = info.get("_songAuthorName", "")
        mapper = info.get("_levelAuthorName", "")
        bpm = float(info.get("_beatsPerMinute", 0) or 0)
        audio_name = info.get("_songFilename")
        env = info.get("_environmentName")
        environment = env if isinstance(env, str) else ""
        time_offset = float(info.get("_songTimeOffset", 0) or 0)
        for s in info.get("_difficultyBeatmapSets") or []:
            ch = s.get("_beatmapCharacteristicName", "")
            for d in s.get("_difficultyBeatmaps") or []:
                cd = d.get("_customData") or {}
                for r in cd.get("_requirements") or []:
                    if r not in reqs:
                        reqs.append(r)
                for r in cd.get("_suggestions") or []:
                    if r not in sugs:
                        sugs.append(r)
                specs.append((ch, str(d.get("_difficulty", "")), d.get("_beatmapFilename", ""),
                              float(d.get("_noteJumpMovementSpeed", 0) or 0),
                              float(d.get("_noteJumpStartBeatOffset", 0) or 0)))
    else:
        song = info.get("song") or {}
        audio = info.get("audio") or {}
        title = song.get("title", "")
        artist = song.get("author", "")
        bpm = float(audio.get("bpm", 0) or 0)
        audio_name = audio.get("songFilename")
        envs = info.get("environmentNames")
        if isinstance(envs, list) and envs and isinstance(envs[0], str):
            environment = envs[0]
        mappers: list[str] = []
        for d in info.get("difficultyBeatmaps") or []:
            for m in (d.get("beatmapAuthors") or {}).get("mappers") or []:
                if m not in mappers:
                    mappers.append(m)
            specs.append((d.get("characteristic", ""), str(d.get("difficulty", "")),
                          d.get("beatmapDataFilename", ""),
                          float(d.get("noteJumpMovementSpeed", 0) or 0),
                          float(d.get("noteJumpStartBeatOffset", 0) or 0)))
        mapper = ", ".join(mappers)

    if bpm <= 0:
        problems.append("Info has no usable BPM")

    diffs: list[Difficulty] = []
    for ch, rank, fname, njs, off in specs:
        p = _find_safe(folder, fname)
        if p is None:
            problems.append(f"missing difficulty file: {fname!r} ({ch} {rank})")
            continue
        if bpm <= 0:
            problems.append(f"skipped {fname!r}: no BPM")
            continue
        try:
            diffs.append(_read_difficulty(p, folder, ch, rank, njs, off, bpm, time_offset,
                                          tempos, layers))
        except (OSError, ValueError, KeyError, IndexError, TypeError, AttributeError) as e:
            problems.append(f"cannot read {fname!r}: {type(e).__name__}: {e}")

    audio_path = _find_safe(folder, audio_name)
    if audio_path is None and (folder / "audio.ogg").is_file():
        audio_path = folder / "audio.ogg"
    bm = BeatMap(folder, map_key(folder.name), title, artist, mapper, bpm, audio_path,
                 tuple(reqs), tuple(sugs), tuple(diffs), tuple(problems), environment)
    return bm, tempos
